"""
VanderWeele regression-based mediation estimators.

This is the primary estimation approach for insurance-mediation. It follows
VanderWeele (2015) "Explanation in Causal Inference" but extends it to
GLM outcomes (Poisson, Gamma, Tweedie) via Monte Carlo integration.

CDE ESTIMATION
--------------
Straightforward plug-in. Fit outcome model E[Y | A, M, C]. Then:
    CDE(m) = E_C[ E[Y | A=a, M=m, C] - E[Y | A=a*, M=m, C] ]

For log-link GLMs, this difference is on the log scale. The ratio
exp(CDE) is the price relativity "with deprivation fixed at level m".

NDE/NIE ESTIMATION (GLMs)
--------------------------
Cannot use the simple product-of-coefficients formula. For non-linear
models, indirect effects require Monte Carlo integration:

NDE = E_C[ sum_m E[Y | A=a, M=m, C] * p(M=m | A=a*, C) ]
      - E_C[ E[Y | A=a*, C] ]

NIE = E_C[ E[Y | A=a*, C] ]
      - E_C[ sum_m E[Y | A=a*, M=m, C] * p(M=m | A=a, C) ]

(These are not the standard NDE/NIE formulas but are algebraically
equivalent under sequential ignorability. Using this formulation avoids
the need for cross-world potential outcomes in the computation step.)

Wait — using the standard VanderWeele definition:
NDE = E[Y(a, M(a*))] - E[Y(a*, M(a*))]
NIE = E[Y(a*, M(a))] - E[Y(a*, M(a*))]
TE  = E[Y(a)] - E[Y(a*)] = NDE + NIE

The Monte Carlo estimator samples M from p(M | A=a*, C) and averages
E[Y | A=a, M, C] over those samples. The VanderWeele (2015) regression
estimator for GLMs (pp. 43-49) is:

For NDE (averaging over C and sampling M from control condition):
    NDE_hat = (1/n) sum_i { (1/K) sum_k E[Y | A=a, M=m_ik, C_i] }
              - (1/n) sum_i E[Y | A=a*, C_i]
where m_ik ~ p(M | A=a*, C_i).

For NIE:
    NIE_hat = (1/n) sum_i { (1/K) sum_k E[Y | A=a*, M=m_ik, C_i] }
              - (1/n) sum_i E[Y | A=a*, C_i]
where m_ik ~ p(M | A=a, C_i).

Wait, this gives NDE + NIE != TE in general. The standard approach:
TE = E[Y(a)] - E[Y(a*)]
NDE = TE - NIE
NIE = E[Y(a, M(a))] - E[Y(a, M(a*))]

We use: NIE = integral E[Y|A=a, M=m, C] p(M=m|A=a, C) dm
             - integral E[Y|A=a, M=m, C] p(M=m|A=a*, C) dm
         averaged over C.

This yields NDE = TE - NIE by construction.

BOOTSTRAP STANDARD ERRORS
--------------------------
We use the nonparametric bootstrap. Each bootstrap iteration:
1. Resample the data with replacement.
2. Refit the outcome and mediator models.
3. Compute the estimand.
4. Collect bootstrap estimates.

95% CI is the 2.5th-97.5th percentile of the bootstrap distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_mediation.models import OutcomeModel, MediatorModel
    from insurance_mediation.estimands import EffectEstimate

# Identification assumptions for each estimand
_CDE_ASSUMPTIONS = [
    "C1: No unmeasured treatment-outcome confounding given covariates C.",
    "C2: No unmeasured mediator-outcome confounding given (A, C).",
    "C3: No unmeasured treatment-mediator confounding given C.",
    "(C4 NOT required for CDE — this is its advantage over NDE/NIE.)",
]

_NDE_NIE_ASSUMPTIONS = [
    "C1: No unmeasured treatment-outcome confounding given covariates C.",
    "C2: No unmeasured mediator-outcome confounding given (A, C).",
    "C3: No unmeasured treatment-mediator confounding given C.",
    "C4: No treatment-induced mediator-outcome confounders. This means no"
    " variable on the treatment -> mediator-outcome path that is affected by"
    " treatment. For postcode->IMD analysis: no other postcode-driven variable"
    " that confounds the IMD -> premium relationship.",
]

_TE_ASSUMPTIONS = [
    "C1: No unmeasured treatment-outcome confounding given covariates C.",
]


def estimate_cde(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    mediator_model: "MediatorModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    mediator_level: float,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> "EffectEstimate":
    """Estimate the Controlled Direct Effect.

    CDE(m) = E_C[E[Y|A=a, M=m, C] - E[Y|A=a*, M=m, C]]

    For GLMs with log link, the effect is on the log-mean scale (a
    log rate ratio for Poisson). Use EffectEstimate.ratio to get the
    relativty (e.g., 1.15 means 15% higher claim rate).

    Parameters
    ----------
    data : pd.DataFrame
    outcome_model : OutcomeModel (fitted)
    mediator_model : MediatorModel (fitted) — not used for CDE itself but
        required for bootstrap (must be refittable).
    treatment_col : str
    mediator_col : str
    treatment_value : any
    control_value : any
    mediator_level : float
        The fixed mediator level m.
    n_bootstrap : int
        Bootstrap replicates for CI. Default 200.
    ci_level : float
        Coverage level. Default 0.95.
    rng : np.random.Generator or None

    Returns
    -------
    EffectEstimate
    """
    from insurance_mediation.estimands import EffectEstimate

    if rng is None:
        rng = np.random.default_rng(42)

    # Point estimate
    point = _cde_point(
        data, outcome_model, treatment_col, mediator_col,
        treatment_value, control_value, mediator_level
    )

    # Bootstrap
    boot_estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(data), size=len(data))
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            # Refit models on bootstrap sample
            boot_outcome = _refit_outcome(outcome_model, boot_data)
            boot_est = _cde_point(
                boot_data, boot_outcome, treatment_col, mediator_col,
                treatment_value, control_value, mediator_level
            )
            boot_estimates.append(boot_est)
        except Exception:
            continue

    alpha = 1 - ci_level
    if len(boot_estimates) >= 10:
        ci_lower = float(np.percentile(boot_estimates, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))
    else:
        # Fall back to no CI if bootstrap fails
        ci_lower = np.nan
        ci_upper = np.nan

    return EffectEstimate(
        estimand="CDE",
        effect=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        scale="difference",
        mediator_level=mediator_level,
        treatment_values=(treatment_value, control_value),
        n_bootstrap=len(boot_estimates),
        assumptions=_CDE_ASSUMPTIONS,
    )


def _cde_point(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    mediator_level: float,
) -> float:
    """Compute CDE point estimate on log-mean or mean scale."""
    # Predict under (treatment, m)
    y_a_m = outcome_model.predict(
        data, treatment_col, treatment_value,
        mediator_col, mediator_value=mediator_level
    )
    # Predict under (control, m)
    y_a0_m = outcome_model.predict(
        data, treatment_col, control_value,
        mediator_col, mediator_value=mediator_level
    )

    if outcome_model.has_log_link:
        # Log ratio: E[log Y(a,m) - log Y(a*,m)]
        # Use log of mean ratio: log(mean(y_a_m) / mean(y_a0_m))
        return float(np.log(np.mean(y_a_m)) - np.log(np.mean(y_a0_m)))
    else:
        # Mean difference for Gaussian
        return float(np.mean(y_a_m) - np.mean(y_a0_m))


def estimate_nde_nie(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    mediator_model: "MediatorModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    n_mc_samples: int = 1000,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple["EffectEstimate", "EffectEstimate", "EffectEstimate"]:
    """Estimate Natural Direct Effect and Natural Indirect Effect.

    Uses Monte Carlo integration over the mediator distribution.

    For NDE: sample M ~ p(M | A=a*, C), compute E[Y|A=a, M, C] averaged
    over samples. NIE is derived as TE - NDE.

    For log-link GLMs, all effects are on the log-mean scale.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_model : OutcomeModel (fitted)
    mediator_model : MediatorModel (fitted)
    treatment_col : str
    mediator_col : str
    treatment_value : any
    control_value : any
    n_mc_samples : int
        Number of Monte Carlo samples per observation. Default 1000.
    n_bootstrap : int
    ci_level : float
    rng : np.random.Generator or None

    Returns
    -------
    tuple[EffectEstimate, EffectEstimate]
        (nde, nie)
    """
    from insurance_mediation.estimands import EffectEstimate

    if rng is None:
        rng = np.random.default_rng(42)

    # Point estimates (te computed consistently with NDE/NIE)
    nde_point, nie_point, te_point = _nde_nie_point(
        data, outcome_model, mediator_model,
        treatment_col, mediator_col,
        treatment_value, control_value,
        n_mc_samples, rng
    )

    # Bootstrap
    nde_boot = []
    nie_boot = []
    te_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(data), size=len(data))
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            boot_outcome = _refit_outcome(outcome_model, boot_data)
            boot_mediator = _refit_mediator(mediator_model, boot_data)
            boot_rng = np.random.default_rng(rng.integers(0, 2**31))
            b_nde, b_nie, b_te = _nde_nie_point(
                boot_data, boot_outcome, boot_mediator,
                treatment_col, mediator_col,
                treatment_value, control_value,
                n_mc_samples // 2,  # fewer MC samples per bootstrap iter for speed
                boot_rng,
            )
            nde_boot.append(b_nde)
            nie_boot.append(b_nie)
            te_boot.append(b_te)
        except Exception:
            continue

    alpha = 1 - ci_level
    if len(nde_boot) >= 10:
        nde_ci = (
            float(np.percentile(nde_boot, 100 * alpha / 2)),
            float(np.percentile(nde_boot, 100 * (1 - alpha / 2))),
        )
        nie_ci = (
            float(np.percentile(nie_boot, 100 * alpha / 2)),
            float(np.percentile(nie_boot, 100 * (1 - alpha / 2))),
        )
    else:
        nde_ci = (np.nan, np.nan)
        nie_ci = (np.nan, np.nan)

    if len(te_boot) >= 10:
        te_ci = (
            float(np.percentile(te_boot, 100 * alpha / 2)),
            float(np.percentile(te_boot, 100 * (1 - alpha / 2))),
        )
    else:
        te_ci = (np.nan, np.nan)

    nde = EffectEstimate(
        estimand="NDE",
        effect=nde_point,
        ci_lower=nde_ci[0],
        ci_upper=nde_ci[1],
        scale="difference",
        treatment_values=(treatment_value, control_value),
        n_bootstrap=len(nde_boot),
        assumptions=_NDE_NIE_ASSUMPTIONS,
    )
    nie = EffectEstimate(
        estimand="NIE",
        effect=nie_point,
        ci_lower=nie_ci[0],
        ci_upper=nie_ci[1],
        scale="difference",
        treatment_values=(treatment_value, control_value),
        n_bootstrap=len(nie_boot),
        assumptions=_NDE_NIE_ASSUMPTIONS,
    )
    te = EffectEstimate(
        estimand="TE",
        effect=te_point,
        ci_lower=te_ci[0],
        ci_upper=te_ci[1],
        scale="difference",
        treatment_values=(treatment_value, control_value),
        n_bootstrap=len(te_boot),
        assumptions=_TE_ASSUMPTIONS,
    )

    return nde, nie, te


def _nde_nie_point(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    mediator_model: "MediatorModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Compute NDE and NIE point estimates via Monte Carlo.

    Strategy (VanderWeele regression approach for GLMs):

    TE = E[Y(a)] - E[Y(a*)]
       = mean(E[Y|A=a, M_obs, C]) - mean(E[Y|A=a*, M_obs, C])
       (using observed mediator values, which integrate over p(M|A,C) naturally)

    NDE = E[Y(a, M(a*))] - E[Y(a*, M(a*))]
       = mean over C of: { avg over M~p(M|A=a*,C) of E[Y|A=a, M, C] }
         - mean(E[Y|A=a*, M_obs, C])

    NIE = TE - NDE
    """
    n_obs = len(data)

    # E[Y | A=a*, M_obs, C] — baseline
    y_control_obs = outcome_model.predict(
        data, treatment_col, control_value,
        mediator_col, mediator_value=None
    )

    # E[Y | A=a, M_obs, C] — treatment with observed mediator
    y_treat_obs = outcome_model.predict(
        data, treatment_col, treatment_value,
        mediator_col, mediator_value=None
    )

    # Sample M from p(M | A=a*, C) for NDE computation
    m_samples_control = mediator_model.sample(
        data, treatment_col, control_value,
        n_samples=n_mc_samples, rng=rng
    )  # shape: (n_mc_samples, n_obs)

    # Sample M from p(M | A=a, C) for TE computation (marginal)
    m_samples_treat = mediator_model.sample(
        data, treatment_col, treatment_value,
        n_samples=n_mc_samples, rng=rng
    )  # shape: (n_mc_samples, n_obs)

    # For NDE: average E[Y|A=a, M=m, C] over M~p(M|A=a*, C)
    y_treat_given_control_m = np.zeros(n_obs)
    for k in range(n_mc_samples):
        m_k = m_samples_control[k]
        cf_data = data.copy()
        cf_data[mediator_col] = m_k
        cf_data[treatment_col] = treatment_value
        preds = _predict_with_mediator_array(outcome_model, cf_data, treatment_col, mediator_col)
        y_treat_given_control_m += preds
    y_treat_given_control_m /= n_mc_samples

    # For TE: E[Y(a)] = average E[Y|A=a, M~p(M|A=a,C), C] over population
    # This correctly marginalises over the treatment-specific mediator distribution
    y_treat_marginal = np.zeros(n_obs)
    for k in range(n_mc_samples):
        m_k = m_samples_treat[k]
        cf_data = data.copy()
        cf_data[mediator_col] = m_k
        cf_data[treatment_col] = treatment_value
        preds = _predict_with_mediator_array(outcome_model, cf_data, treatment_col, mediator_col)
        y_treat_marginal += preds
    y_treat_marginal /= n_mc_samples

    # E[Y(a*)] = average E[Y|A=a*, M~p(M|A=a*,C), C]
    # Reuse m_samples_control for efficiency
    y_ctrl_marginal = np.zeros(n_obs)
    for k in range(n_mc_samples):
        m_k = m_samples_control[k]
        cf_data = data.copy()
        cf_data[mediator_col] = m_k
        cf_data[treatment_col] = control_value
        preds = _predict_with_mediator_array(outcome_model, cf_data, treatment_col, mediator_col)
        y_ctrl_marginal += preds
    y_ctrl_marginal /= n_mc_samples

    if outcome_model.has_log_link:
        # All effects on log-mean scale
        te = float(np.log(np.mean(y_treat_marginal)) - np.log(np.mean(y_ctrl_marginal)))
        nde = float(np.log(np.mean(y_treat_given_control_m)) - np.log(np.mean(y_ctrl_marginal)))
        nie = te - nde
    else:
        te = float(np.mean(y_treat_marginal) - np.mean(y_ctrl_marginal))
        nde = float(np.mean(y_treat_given_control_m) - np.mean(y_ctrl_marginal))
        nie = te - nde

    return nde, nie, te


def _predict_with_mediator_array(
    outcome_model: "OutcomeModel",
    cf_data: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
) -> np.ndarray:
    """Predict outcome using the mediator values already set in cf_data."""
    # The data already has the mediator set — just predict
    if outcome_model.exposure_col is not None:
        offset = np.log(cf_data[outcome_model.exposure_col].clip(lower=1e-9))
        return outcome_model._fit_result.predict(cf_data, offset=offset)
    return outcome_model._fit_result.predict(cf_data)


def estimate_total_effect(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> "EffectEstimate":
    """Estimate the total causal effect of treatment on outcome.

    TE = E[Y(a)] - E[Y(a*)]

    For GLMs with log link: log(E[Y|A=a, M_obs, C]) - log(E[Y|A=a*, M_obs, C])
    averaged over the empirical distribution of (M, C).

    Parameters
    ----------
    data : pd.DataFrame
    outcome_model : OutcomeModel
    treatment_col : str
    mediator_col : str
    treatment_value : any
    control_value : any
    n_bootstrap : int
    ci_level : float
    rng : np.random.Generator or None

    Returns
    -------
    EffectEstimate
    """
    from insurance_mediation.estimands import EffectEstimate

    if rng is None:
        rng = np.random.default_rng(42)

    point = _te_point(data, outcome_model, treatment_col, mediator_col,
                      treatment_value, control_value)

    boot_ests = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(data), size=len(data))
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            boot_outcome = _refit_outcome(outcome_model, boot_data)
            boot_ests.append(_te_point(
                boot_data, boot_outcome, treatment_col, mediator_col,
                treatment_value, control_value
            ))
        except Exception:
            continue

    alpha = 1 - ci_level
    if len(boot_ests) >= 10:
        ci_lower = float(np.percentile(boot_ests, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_ests, 100 * (1 - alpha / 2)))
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return EffectEstimate(
        estimand="TE",
        effect=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        scale="difference",
        treatment_values=(treatment_value, control_value),
        n_bootstrap=len(boot_ests),
        assumptions=_TE_ASSUMPTIONS,
    )


def _te_point(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
) -> float:
    y_a = outcome_model.predict(data, treatment_col, treatment_value, mediator_col)
    y_a0 = outcome_model.predict(data, treatment_col, control_value, mediator_col)
    if outcome_model.has_log_link:
        return float(np.log(np.mean(y_a)) - np.log(np.mean(y_a0)))
    return float(np.mean(y_a) - np.mean(y_a0))


def _refit_outcome(outcome_model: "OutcomeModel", data: pd.DataFrame) -> "OutcomeModel":
    """Refit the outcome model on new data (for bootstrap)."""
    from insurance_mediation.models import OutcomeModel

    if outcome_model.formula is None:
        raise RuntimeError(
            "Cannot refit outcome model: no formula stored. "
            "Pass formula to OutcomeModel or use from_fit_result() with "
            "a model that stores the formula."
        )
    new_model = OutcomeModel(
        model_type=outcome_model.model_type,
        formula=outcome_model.formula,
        exposure_col=outcome_model.exposure_col,
        tweedie_var_power=outcome_model.tweedie_var_power,
    )
    # Determine outcome column from formula (left side of ~)
    outcome_col = outcome_model.formula.split("~")[0].strip()
    new_model.fit(data, outcome_col)
    return new_model


def _refit_mediator(mediator_model: "MediatorModel", data: pd.DataFrame) -> "MediatorModel":
    """Refit the mediator model on new data (for bootstrap)."""
    from insurance_mediation.models import MediatorModel

    if mediator_model.formula is None:
        raise RuntimeError(
            "Cannot refit mediator model: no formula stored."
        )
    new_model = MediatorModel(
        model_type=mediator_model.model_type,
        formula=mediator_model.formula,
    )
    mediator_col = mediator_model.formula.split("~")[0].strip()
    new_model.fit(data, mediator_col)
    return new_model
