"""
Data classes for mediation analysis estimands and results.

Each estimand has precise causal identification assumptions. These are
surfaced explicitly in every EffectEstimate so that FCA reports can
state clearly what is required for the number to be trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EstimandType = Literal["CDE", "NDE", "NIE", "TE", "IIE_direct", "IIE_indirect"]


@dataclass
class EffectEstimate:
    """A single causal effect estimate with confidence interval and assumptions.

    Attributes
    ----------
    estimand : str
        The estimand type — one of CDE, NDE, NIE, TE, IIE_direct, IIE_indirect.
    effect : float
        Point estimate. For ratio scale (Poisson/Gamma/Tweedie outcomes) this is
        a log-ratio (difference on the log scale), unless scale="ratio" in which
        case it is the ratio itself. For Gaussian this is a mean difference.
    ci_lower : float
        Lower bound of bootstrap confidence interval.
    ci_upper : float
        Upper bound of bootstrap confidence interval.
    scale : str
        "difference" (additive) or "ratio" (multiplicative). For GLMs with log
        link, "ratio" gives exp(effect) — the relativty familiar to pricing
        teams.
    mediator_level : float or None
        For CDE only — the mediator level at which the direct effect was
        evaluated. E.g. mediator_level=5 means "if IMD decile were 5 for all".
    treatment_values : tuple[any, any]
        (treatment, control) values used. E.g. ("E1", "SW1").
    n_bootstrap : int
        Number of bootstrap replicates used for the CI.
    assumptions : list[str]
        Causal identification assumptions required for this estimand to have
        a valid causal interpretation.
    notes : str
        Any analyst notes (e.g., "exposure-mediator interaction detected").
    """

    estimand: EstimandType
    effect: float
    ci_lower: float
    ci_upper: float
    scale: Literal["difference", "ratio"] = "difference"
    mediator_level: float | None = None
    treatment_values: tuple = (1, 0)
    n_bootstrap: int = 0
    assumptions: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def ratio(self) -> float:
        """Effect on ratio scale — exp(effect) for log-link models.

        For Gaussian (identity link) models this is not meaningful. Check
        scale attribute first.
        """
        import math
        if self.scale == "ratio":
            return self.effect
        return math.exp(self.effect)

    @property
    def ratio_ci(self) -> tuple[float, float]:
        """Confidence interval on ratio scale."""
        import math
        if self.scale == "ratio":
            return (self.ci_lower, self.ci_upper)
        return (math.exp(self.ci_lower), math.exp(self.ci_upper))

    @property
    def significant(self) -> bool:
        """True if the CI does not include zero (difference) or one (ratio)."""
        if self.scale == "ratio":
            return not (self.ci_lower <= 1.0 <= self.ci_upper)
        return not (self.ci_lower <= 0.0 <= self.ci_upper)

    def __repr__(self) -> str:
        sig = " *" if self.significant else ""
        if self.scale == "difference":
            return (
                f"EffectEstimate({self.estimand}: {self.effect:.4f} "
                f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]{sig})"
            )
        else:
            return (
                f"EffectEstimate({self.estimand}: {self.effect:.4f} "
                f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]{sig}, ratio)"
            )


@dataclass
class MediationResult:
    """Full results from a MediationAnalysis.fit() call.

    Provides access to all estimands. The most commonly used are:
    - cde(mediator_level) — controlled direct effect
    - nde() — natural direct effect
    - nie() — natural indirect effect
    - total_effect() — total causal effect
    - sensitivity() — sensitivity analysis results

    All EffectEstimate objects include the causal assumptions required for
    valid interpretation.

    Attributes
    ----------
    treatment : str
        Column name of the treatment variable.
    mediator : str
        Column name of the mediator variable.
    outcome : str
        Column name of the outcome variable.
    treatment_values : tuple
        (treatment, control) values — e.g., ("E1", "SW1").
    outcome_model_type : str
        GLM family used for outcome model — "poisson", "gamma", "tweedie",
        "gaussian".
    mediator_model_type : str
        Model type for mediator — "linear", "logistic".
    n_obs : int
        Number of observations used in estimation.
    """

    treatment: str
    mediator: str
    outcome: str
    treatment_values: tuple
    outcome_model_type: str
    mediator_model_type: str
    n_obs: int
    # Internal storage of pre-computed estimates
    _cde_estimates: dict[float, EffectEstimate] = field(default_factory=dict)
    _nde: EffectEstimate | None = None
    _nie: EffectEstimate | None = None
    _total_effect: EffectEstimate | None = None
    _sensitivity: SensitivityResult | None = None
    # Fitted model objects (statsmodels results)
    _outcome_fit: object = None
    _mediator_fit: object = None
    _data: object = None  # the original DataFrame

    def cde(self, mediator_level: float | None = None) -> EffectEstimate:
        """Controlled Direct Effect at a specific mediator level.

        The CDE answers: "If we intervened to set everyone's mediator to the
        same level m, what would the treatment effect on the outcome be?"

        For FCA purposes this is the most defensible estimand. CDE ≈ 0 means
        the treatment effect is fully mediated through the specified mediator
        (i.e., if we equalised deprivation across all postcodes, the price
        differential would vanish).

        Parameters
        ----------
        mediator_level : float, optional
            The level m at which to fix the mediator. If None, uses the
            population mean of the mediator.

        Returns
        -------
        EffectEstimate
            Effect with CI, assumptions, and notes.

        Raises
        ------
        RuntimeError
            If this CDE level was not pre-computed. Call MediationAnalysis.fit()
            with cde_levels parameter.
        """
        # Check estimates exist first
        if not self._cde_estimates:
            raise RuntimeError(
                "No CDE estimates available. Call MediationAnalysis.fit() first."
            )

        if mediator_level is None:
            # Use population mean
            if self._data is not None:
                import numpy as np
                mediator_level = float(np.mean(self._data[self.mediator]))
            else:
                raise RuntimeError("mediator_level required when data not stored")
        nearest = min(self._cde_estimates.keys(), key=lambda k: abs(k - mediator_level))
        return self._cde_estimates[nearest]

    def nde(self) -> EffectEstimate:
        """Natural Direct Effect — effect not operating through the mediator.

        NDE = E[Y(a, M(a*))] - E[Y(a*, M(a*))]

        The component of the total treatment effect that would remain if the
        mediator were held at the level it would naturally take under the
        control condition. Requires cross-world counterfactual assumptions.

        Returns
        -------
        EffectEstimate

        Raises
        ------
        RuntimeError
            If not yet estimated.
        """
        if self._nde is None:
            raise RuntimeError("NDE not available. Call MediationAnalysis.fit() first.")
        return self._nde

    def nie(self) -> EffectEstimate:
        """Natural Indirect Effect — effect operating through the mediator.

        NIE = E[Y(a*, M(a))] - E[Y(a*, M(a*))]

        The component of the total effect that operates through the mediator
        pathway. For fairness analysis: if NIE is large and the mediator is
        a legitimate risk factor, this portion of the price differential is
        defensible.

        Returns
        -------
        EffectEstimate

        Raises
        ------
        RuntimeError
            If not yet estimated.
        """
        if self._nie is None:
            raise RuntimeError("NIE not available. Call MediationAnalysis.fit() first.")
        return self._nie

    def total_effect(self) -> EffectEstimate:
        """Total causal effect of treatment on outcome.

        TE = E[Y(a)] - E[Y(a*)]

        For well-identified models, NDE + NIE ≈ TE (subject to Monte Carlo
        integration error).

        Returns
        -------
        EffectEstimate
        """
        if self._total_effect is None:
            raise RuntimeError(
                "Total effect not available. Call MediationAnalysis.fit() first."
            )
        return self._total_effect

    def sensitivity(
        self,
        rho_range: tuple[float, float] = (-0.5, 0.5),
        n_rho: int = 21,
    ) -> "SensitivityResult":
        """Sensitivity analysis for unmeasured mediator-outcome confounding.

        Implements Imai et al. (2010) sensitivity analysis via the rho
        parameter — the correlation between the residuals of the outcome
        model and the mediator model. Under sequential ignorability, rho = 0.
        Varying rho shows how robust the NIE estimate is to violations of
        this assumption.

        Also computes E-value (VanderWeele & Ding 2017): the minimum
        unmeasured confounder effect size needed to explain away the NIE.

        Parameters
        ----------
        rho_range : tuple[float, float]
            Range of rho to evaluate. Default (-0.5, 0.5).
        n_rho : int
            Number of rho values to evaluate. Default 21.

        Returns
        -------
        SensitivityResult
        """
        if self._sensitivity is not None:
            return self._sensitivity
        raise RuntimeError(
            "Sensitivity not computed. Call MediationAnalysis.fit() first, "
            "or call MediationAnalysis with compute_sensitivity=True."
        )

    def report(
        self,
        output: str = "mediation_report.html",
        title: str = "Mediation Analysis Report",
        protected_attribute: str | None = None,
    ) -> str:
        """Generate an FCA-ready HTML mediation report.

        The report includes: causal DAG, total effect decomposition table,
        CDE, NDE/NIE with 95% CIs, sensitivity analysis bounds, and a
        template Section 19 proportionality statement.

        Parameters
        ----------
        output : str
            Output file path. Defaults to "mediation_report.html".
        title : str
            Report title.
        protected_attribute : str, optional
            Name of the protected characteristic being proxied (e.g.,
            "ethnicity", "religion"). Appears in the proportionality statement.

        Returns
        -------
        str
            Path to the generated report file.
        """
        from insurance_mediation.report import generate_report
        return generate_report(self, output=output, title=title,
                               protected_attribute=protected_attribute)

    def summary(self) -> str:
        """Human-readable summary of all computed estimands."""
        lines = [
            f"Mediation Analysis Summary",
            f"Treatment: {self.treatment}  ({self.treatment_values[0]} vs {self.treatment_values[1]})",
            f"Mediator:  {self.mediator}",
            f"Outcome:   {self.outcome}  ({self.outcome_model_type})",
            f"N obs:     {self.n_obs:,}",
            "",
        ]
        try:
            te = self.total_effect()
            lines.append(f"Total Effect:              {te.effect:+.4f}  [{te.ci_lower:+.4f}, {te.ci_upper:+.4f}]")
        except RuntimeError:
            pass
        try:
            nde = self.nde()
            lines.append(f"Natural Direct Effect:     {nde.effect:+.4f}  [{nde.ci_lower:+.4f}, {nde.ci_upper:+.4f}]")
        except RuntimeError:
            pass
        try:
            nie = self.nie()
            lines.append(f"Natural Indirect Effect:   {nie.effect:+.4f}  [{nie.ci_lower:+.4f}, {nie.ci_upper:+.4f}]")
        except RuntimeError:
            pass
        if self._cde_estimates:
            for m_level, est in sorted(self._cde_estimates.items()):
                lines.append(f"CDE (m={m_level:.2f}):              {est.effect:+.4f}  [{est.ci_lower:+.4f}, {est.ci_upper:+.4f}]")
        return "\n".join(lines)


@dataclass
class SensitivityResult:
    """Results of Imai et al. sensitivity analysis and E-value computation.

    Under sequential ignorability, rho = 0 and the NIE is identified. This
    object shows how the NIE estimate changes as rho varies, and reports the
    rho at which the NIE crosses zero (i.e., would be explained away by
    unmeasured confounding).

    Attributes
    ----------
    rho_values : list[float]
        The rho values evaluated.
    nie_estimates : list[float]
        NIE point estimates at each rho.
    nie_ci_lower : list[float]
        Lower CI at each rho.
    nie_ci_upper : list[float]
        Upper CI at each rho.
    rho_at_zero : float or None
        The rho at which NIE = 0. None if NIE does not cross zero in the
        evaluated range.
    e_value : float
        E-value: minimum risk ratio of unmeasured confounder with both
        treatment and outcome needed to explain away the NIE.
    e_value_ci : float
        E-value for the confidence interval bound closest to zero.
    nie_at_rho0 : EffectEstimate
        The NIE estimate at rho = 0 (the baseline, assuming sequential
        ignorability).
    """

    rho_values: list[float]
    nie_estimates: list[float]
    nie_ci_lower: list[float]
    nie_ci_upper: list[float]
    rho_at_zero: float | None
    e_value: float
    e_value_ci: float
    nie_at_rho0: EffectEstimate

    def __repr__(self) -> str:
        rho_str = (
            f"rho_at_zero={self.rho_at_zero:.3f}" if self.rho_at_zero is not None
            else "effect does not cross zero in range"
        )
        return (
            f"SensitivityResult("
            f"E-value={self.e_value:.3f}, "
            f"E-value(CI)={self.e_value_ci:.3f}, "
            f"{rho_str})"
        )
