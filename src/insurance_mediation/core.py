"""
Main MediationAnalysis class — the primary entry point for users.

Design rationale: the API spec (KB 1468) calls for a MediationAnalysis class
that takes model type strings and fits everything internally from a DataFrame.
We also support pre-fitted statsmodels objects for users who have complex
production models.

The fit() method returns a MediationResult object which has lazy-evaluated
estimands. CDE, NDE, NIE, and TE are all computed at fit time because the
bootstrap CI computation is the expensive step — we want a single fit call
that handles all the computation.

Default cde_levels: if not specified, we evaluate CDE at the 10th, 25th, 50th,
75th, 90th percentiles of the observed mediator distribution. This gives a
useful picture of how the direct effect varies across the mediator range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


OutcomeModelType = Literal["poisson", "gamma", "tweedie", "gaussian"]
MediatorModelType = Literal["linear", "logistic"]


class MediationAnalysis:
    """Causal mediation analysis for insurance pricing fairness.

    Decomposes treatment effects (e.g., postcode price differentials) into
    direct and mediated components. Three estimands are available:

    - CDE (Controlled Direct Effect): effect when mediator is fixed at a
      specific level. Most defensible for FCA compliance.
    - NDE/NIE (Natural Direct/Indirect Effects): full decomposition requiring
      cross-world counterfactual assumptions.
    - Total Effect: the overall treatment -> outcome causal effect.

    All effects for GLMs with log link are on the log-mean scale (log rate
    ratios). Use EffectEstimate.ratio to get the relativty scale.

    Parameters
    ----------
    outcome_model : str or statsmodels GLM result
        GLM family for the outcome. One of "poisson", "gamma", "tweedie",
        "gaussian". Or pass a pre-fitted statsmodels GLMResultsWrapper.
    mediator_model : str or statsmodels result
        Model type for the mediator. One of "linear" (OLS) or "logistic"
        (binary mediator). Or pass a pre-fitted statsmodels result.
    exposure_col : str or None
        Column name for the exposure offset (e.g. "exposure" for policies).
        Used as log(exposure) offset in Poisson/Tweedie models. If None,
        no offset is applied.
    n_mc_samples : int
        Monte Carlo samples for NDE/NIE estimation. Higher gives lower
        simulation variance but slower computation. Default 1000.
    n_bootstrap : int
        Bootstrap replicates for confidence intervals. Default 200.
    ci_level : float
        Confidence interval coverage. Default 0.95.
    tweedie_var_power : float
        Variance power for Tweedie family. Default 1.5 (compound
        Poisson-Gamma, typical for burning cost).
    include_interaction : bool
        Whether to include a treatment × mediator interaction term in the
        outcome model. Default True. Including the interaction is important
        if the mediator effect on the outcome varies by treatment group.
    seed : int or None
        Random seed for reproducibility.

    Examples
    --------
    Basic usage with a Poisson frequency model::

        from insurance_mediation import MediationAnalysis

        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=1000,
        )
        results = ma.fit(
            data=df,
            treatment="postcode_group",
            mediator="imd_decile",
            outcome="claim_count",
            covariates=["vehicle_age", "driver_age", "cover_type"],
            treatment_value="E1",
            control_value="SW1",
        )
        print(results.cde(mediator_level=5))
        print(results.nde())
        print(results.nie())
        results.report(output="report.html")

    With pre-fitted models (e.g., production pricing GLM)::

        import statsmodels.api as sm

        # Your production GLM
        fitted_outcome_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

        ma = MediationAnalysis(
            outcome_model=fitted_outcome_model,
            mediator_model="linear",
            exposure_col="exposure",
        )
    """

    def __init__(
        self,
        outcome_model: str | object = "poisson",
        mediator_model: str | object = "linear",
        exposure_col: str | None = None,
        n_mc_samples: int = 1000,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        tweedie_var_power: float = 1.5,
        include_interaction: bool = True,
        seed: int | None = 42,
    ) -> None:
        self.outcome_model_spec = outcome_model
        self.mediator_model_spec = mediator_model
        self.exposure_col = exposure_col
        self.n_mc_samples = n_mc_samples
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.tweedie_var_power = tweedie_var_power
        self.include_interaction = include_interaction
        self.seed = seed

    def fit(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: list[str] | None = None,
        treatment_value=1,
        control_value=0,
        cde_levels: list[float] | None = None,
        compute_sensitivity: bool = False,
        rho_range: tuple[float, float] = (-0.5, 0.5),
    ) -> "MediationResult":
        """Fit mediation models and estimate all causal effects.

        Parameters
        ----------
        data : pd.DataFrame
            The analysis dataset. Must contain treatment, mediator, outcome
            columns plus any covariates.
        treatment : str
            Column name of the treatment variable (e.g., "postcode_group").
        mediator : str
            Column name of the mediator variable (e.g., "imd_decile").
        outcome : str
            Column name of the outcome variable (e.g., "claim_count").
        covariates : list[str] or None
            Covariate column names to include as confounders. These are
            included in both outcome and mediator models. If None, no
            covariates are included.
        treatment_value : any
            The treatment group value. Default 1.
        control_value : any
            The control/reference group value. Default 0.
        cde_levels : list[float] or None
            Mediator levels at which to evaluate the CDE. If None, uses the
            10th, 25th, 50th, 75th, 90th percentiles of the mediator in the
            data.
        compute_sensitivity : bool
            Whether to run sensitivity analysis. Takes extra time. Default False.
        rho_range : tuple[float, float]
            rho range for sensitivity analysis if compute_sensitivity=True.

        Returns
        -------
        MediationResult

        Raises
        ------
        ValueError
            If required columns are missing from data.
        """
        from insurance_mediation.models import (
            OutcomeModel, MediatorModel,
            build_outcome_model, build_mediator_model,
        )
        from insurance_mediation.estimators import (
            estimate_cde, estimate_nde_nie, estimate_total_effect,
        )
        from insurance_mediation.estimands import MediationResult

        if covariates is None:
            covariates = []

        # Validate columns
        required = {treatment, mediator, outcome} | set(covariates)
        if self.exposure_col is not None:
            required.add(self.exposure_col)
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        rng = np.random.default_rng(self.seed)

        # Determine model types for pre-fitted vs string specs
        if isinstance(self.outcome_model_spec, str):
            outcome_model_type = self.outcome_model_spec
            outcome_model = build_outcome_model(
                model_type=outcome_model_type,
                data=data,
                outcome_col=outcome,
                treatment_col=treatment,
                mediator_col=mediator,
                covariates=covariates,
                exposure_col=self.exposure_col,
                interaction=self.include_interaction,
                tweedie_var_power=self.tweedie_var_power,
            )
        else:
            # Pre-fitted statsmodels result
            outcome_model_type = _detect_glm_family(self.outcome_model_spec)
            outcome_model = OutcomeModel.from_fit_result(
                self.outcome_model_spec,
                model_type=outcome_model_type,
                exposure_col=self.exposure_col,
            )

        if isinstance(self.mediator_model_spec, str):
            mediator_model_type = self.mediator_model_spec
            mediator_model = build_mediator_model(
                model_type=mediator_model_type,
                data=data,
                mediator_col=mediator,
                treatment_col=treatment,
                covariates=covariates,
            )
        else:
            mediator_model_type = _detect_mediator_type(self.mediator_model_spec)
            mediator_model = MediatorModel.from_fit_result(
                self.mediator_model_spec,
                model_type=mediator_model_type,
            )

        # Determine CDE evaluation levels
        if cde_levels is None:
            cde_levels = list(np.percentile(
                data[mediator].dropna(),
                [10, 25, 50, 75, 90]
            ))

        # Estimate all effects
        # Total Effect
        te = estimate_total_effect(
            data=data,
            outcome_model=outcome_model,
            treatment_col=treatment,
            mediator_col=mediator,
            treatment_value=treatment_value,
            control_value=control_value,
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
            rng=rng,
        )

        # NDE / NIE
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col=treatment,
            mediator_col=mediator,
            treatment_value=treatment_value,
            control_value=control_value,
            n_mc_samples=self.n_mc_samples,
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
            rng=rng,
        )

        # CDE at each specified level
        cde_estimates = {}
        for m_level in cde_levels:
            m_level_f = float(m_level)
            est = estimate_cde(
                data=data,
                outcome_model=outcome_model,
                mediator_model=mediator_model,
                treatment_col=treatment,
                mediator_col=mediator,
                treatment_value=treatment_value,
                control_value=control_value,
                mediator_level=m_level_f,
                n_bootstrap=self.n_bootstrap,
                ci_level=self.ci_level,
                rng=rng,
            )
            cde_estimates[m_level_f] = est

        # Build result object
        result = MediationResult(
            treatment=treatment,
            mediator=mediator,
            outcome=outcome,
            treatment_values=(treatment_value, control_value),
            outcome_model_type=outcome_model_type,
            mediator_model_type=mediator_model_type,
            n_obs=len(data),
            _cde_estimates=cde_estimates,
            _nde=nde,
            _nie=nie,
            _total_effect=te,
            _outcome_fit=outcome_model._fit_result,
            _mediator_fit=mediator_model._fit_result,
            _data=data,
        )

        # Optional sensitivity analysis
        if compute_sensitivity:
            from insurance_mediation.sensitivity import compute_sensitivity as _sens
            sens = _sens(
                data=data,
                outcome_model=outcome_model,
                mediator_model=mediator_model,
                treatment_col=treatment,
                mediator_col=mediator,
                treatment_value=treatment_value,
                control_value=control_value,
                nie_estimate=nie,
                rho_range=rho_range,
                rng=rng,
            )
            result._sensitivity = sens

        return result


def _detect_glm_family(fit_result) -> str:
    """Detect the GLM family type from a fitted statsmodels result."""
    try:
        family = fit_result.model.family
        family_name = type(family).__name__.lower()
        if "poisson" in family_name:
            return "poisson"
        elif "gamma" in family_name:
            return "gamma"
        elif "tweedie" in family_name:
            return "tweedie"
        elif "gaussian" in family_name:
            return "gaussian"
        elif "binomial" in family_name:
            return "gaussian"  # treat as quasi-Gaussian for mediation
    except AttributeError:
        pass
    # OLS result — check for mse_resid
    if hasattr(fit_result, "mse_resid"):
        return "gaussian"
    return "gaussian"


def _detect_mediator_type(fit_result) -> str:
    """Detect whether a mediator model is linear or logistic."""
    try:
        family = fit_result.model.family
        family_name = type(family).__name__.lower()
        if "binomial" in family_name:
            return "logistic"
    except AttributeError:
        pass
    return "linear"
