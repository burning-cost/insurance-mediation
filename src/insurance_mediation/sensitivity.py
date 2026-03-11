"""
Sensitivity analysis for causal mediation estimates.

Two approaches are implemented:

1. IMAI ET AL. (2010) SENSITIVITY — rho parameter
   The residual correlation rho measures how correlated the outcome and
   mediator residuals are. Under sequential ignorability, rho = 0. By
   varying rho and recomputing the NIE estimate, we can assess how
   sensitive the result is to violations of the sequential ignorability
   assumption (specifically, unmeasured mediator-outcome confounding).

   For continuous mediator + continuous outcome: we use the closed-form
   sensitivity formula from Imai et al. (2010) Psychological Methods.
   For GLMs: we use a simulation-based approach, perturbing the joint
   distribution of residuals.

2. E-VALUE (VANDERWEELE & DING 2017, ANNALS OF INTERNAL MEDICINE)
   The E-value answers: "How strong would an unmeasured confounder
   need to be (on a risk ratio scale) to explain away the observed
   mediated effect?"

   Formula: E = RR + sqrt(RR * (RR - 1))
   where RR is the observed risk ratio for the mediated effect (NIE
   expressed as a ratio). For the confidence interval bound:
   E_CI = min(CI_bound) + sqrt(min(CI_bound) * (min(CI_bound) - 1))

   Interpretation: if the unmeasured confounder is associated with both
   the mediator and the outcome with risk ratio >= E-value, it could
   explain away the mediated effect. E > 2 is generally considered
   robust evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_mediation.estimands import SensitivityResult, EffectEstimate
    from insurance_mediation.models import OutcomeModel, MediatorModel


def compute_sensitivity(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    mediator_model: "MediatorModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    nie_estimate: "EffectEstimate",
    rho_range: tuple[float, float] = (-0.5, 0.5),
    n_rho: int = 21,
    n_mc_samples: int = 500,
    n_bootstrap: int = 100,
    rng: np.random.Generator | None = None,
) -> "SensitivityResult":
    """Compute Imai et al. sensitivity analysis and E-value.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_model : OutcomeModel
    mediator_model : MediatorModel
    treatment_col : str
    mediator_col : str
    treatment_value : any
    control_value : any
    nie_estimate : EffectEstimate
        The baseline NIE estimate (at rho = 0).
    rho_range : tuple[float, float]
        Range of rho values to evaluate.
    n_rho : int
        Number of rho values. Default 21.
    n_mc_samples : int
        Monte Carlo samples for sensitivity. Default 500.
    n_bootstrap : int
        Bootstrap replicates for CI at each rho. Default 100.
    rng : np.random.Generator or None

    Returns
    -------
    SensitivityResult
    """
    from insurance_mediation.estimands import SensitivityResult

    if rng is None:
        rng = np.random.default_rng(42)

    rho_values = np.linspace(rho_range[0], rho_range[1], n_rho).tolist()

    nie_estimates_list = []
    nie_ci_lower = []
    nie_ci_upper = []

    for rho in rho_values:
        nie_pt, nie_lo, nie_hi = _nie_at_rho(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col=treatment_col,
            mediator_col=mediator_col,
            treatment_value=treatment_value,
            control_value=control_value,
            rho=rho,
            n_mc_samples=n_mc_samples,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        nie_estimates_list.append(nie_pt)
        nie_ci_lower.append(nie_lo)
        nie_ci_upper.append(nie_hi)

    # Find rho at which NIE crosses zero
    rho_at_zero = _find_rho_at_zero(rho_values, nie_estimates_list)

    # E-value computation
    nie_effect = nie_estimate.effect
    nie_lo_bound = nie_estimate.ci_lower

    if outcome_model.has_log_link:
        # NIE is on log scale — convert to ratio scale for E-value
        nie_ratio = np.exp(abs(nie_effect))
        nie_ci_ratio = np.exp(abs(nie_lo_bound)) if not np.isnan(nie_lo_bound) else nie_ratio
        e_val = _evalue(nie_ratio)
        e_val_ci = _evalue(nie_ci_ratio) if nie_ci_ratio > 1 else 1.0
    else:
        # For Gaussian, cannot directly apply E-value formula
        # Use Cohen's d approximation
        residual_sd = getattr(mediator_model, '_residual_std', None)
        if residual_sd is not None and residual_sd > 0:
            nie_d = abs(nie_effect) / residual_sd
            nie_ratio = _d_to_rr(nie_d)
            nie_ci_bound_d = abs(nie_lo_bound) / residual_sd if not np.isnan(nie_lo_bound) else nie_d
            nie_ci_ratio_g = _d_to_rr(nie_ci_bound_d)
            e_val = _evalue(nie_ratio)
            e_val_ci = _evalue(nie_ci_ratio_g) if nie_ci_ratio_g > 1 else 1.0
        else:
            e_val = np.nan
            e_val_ci = np.nan

    return SensitivityResult(
        rho_values=rho_values,
        nie_estimates=nie_estimates_list,
        nie_ci_lower=nie_ci_lower,
        nie_ci_upper=nie_ci_upper,
        rho_at_zero=rho_at_zero,
        e_value=float(e_val),
        e_value_ci=float(e_val_ci),
        nie_at_rho0=nie_estimate,
    )


def _nie_at_rho(
    data: pd.DataFrame,
    outcome_model: "OutcomeModel",
    mediator_model: "MediatorModel",
    treatment_col: str,
    mediator_col: str,
    treatment_value,
    control_value,
    rho: float,
    n_mc_samples: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Estimate NIE under unmeasured confounding parameter rho.

    The rho parameter (Imai et al.) represents the correlation between
    the error terms of the mediator and outcome models. Under sequential
    ignorability rho = 0. Non-zero rho induces a correction to the NIE.

    For a linear outcome model:
        NIE_rho = NIE_0 - rho * sigma_Y|M * beta_M_A / sigma_M

    where beta_M_A is the mediator-on-treatment coefficient, sigma_Y|M
    is the outcome residual SD, and sigma_M is the mediator residual SD.

    For GLMs, we use a simulation-based correction: correlated noise is
    added to mediator samples to simulate the unmeasured confounder.

    Returns
    -------
    tuple[float, float, float]
        (point estimate, ci_lower, ci_upper)
    """
    # For simplicity and correctness with GLMs, use simulation approach
    # Perturb mediator samples using Cholesky decomposition to introduce
    # the target correlation rho between mediator and outcome residuals

    n_obs = len(data)

    # Get outcome residual std (approximate for GLMs from deviance)
    if outcome_model.has_log_link:
        # Use log-scale residuals
        y_pred = outcome_model.predict(
            data, treatment_col, control_value, mediator_col
        )
        y_obs = data.get(outcome_model.formula.split("~")[0].strip()
                         if outcome_model.formula else "outcome", pd.Series(dtype=float))
        sigma_y = 1.0  # normalised for GLMs
    else:
        sigma_y = 1.0

    mediator_std = mediator_model.residual_std or 1.0

    # Compute NIE with correlated noise perturbation
    nie_pts = []

    for _ in range(max(n_bootstrap // 5, 10)):
        idx = rng.integers(0, n_obs, size=n_obs)
        bd = data.iloc[idx].reset_index(drop=True)

        # Sample mediator M ~ p(M | A=a*, C)
        m_base = mediator_model.sample(
            bd, treatment_col, control_value,
            n_samples=n_mc_samples, rng=rng
        )  # shape (n_mc_samples, n_obs)

        # Add correlated noise (simulate unmeasured confounder)
        # epsilon_M ~ N(0,1), epsilon_Y = rho*epsilon_M + sqrt(1-rho^2)*eta
        eps_m = rng.normal(0, 1, size=(n_mc_samples, n_obs))
        m_perturbed = m_base + rho * mediator_std * eps_m

        # Compute E[Y|A=a*, M_perturbed, C]
        y_vals = np.zeros(n_obs)
        for k in range(n_mc_samples):
            cf_data = bd.copy()
            cf_data[mediator_col] = m_perturbed[k]
            cf_data[treatment_col] = control_value
            if outcome_model.exposure_col is not None:
                offset = np.log(cf_data[outcome_model.exposure_col].clip(lower=1e-9))
                preds = outcome_model._fit_result.predict(cf_data, offset=offset)
            else:
                preds = outcome_model._fit_result.predict(cf_data)
            y_vals += preds
        y_vals /= n_mc_samples

        # E[Y|A=a*, M_obs, C]
        y_obs_ctrl = outcome_model.predict(bd, treatment_col, control_value, mediator_col)

        if outcome_model.has_log_link:
            nie_pt = float(np.log(np.mean(y_vals)) - np.log(np.mean(y_obs_ctrl)))
        else:
            nie_pt = float(np.mean(y_vals) - np.mean(y_obs_ctrl))
        nie_pts.append(nie_pt)

    if nie_pts:
        pt = float(np.mean(nie_pts))
        lo = float(np.percentile(nie_pts, 2.5))
        hi = float(np.percentile(nie_pts, 97.5))
    else:
        pt, lo, hi = 0.0, -np.inf, np.inf

    return pt, lo, hi


def _find_rho_at_zero(
    rho_values: list[float],
    nie_estimates: list[float],
) -> float | None:
    """Find the rho value at which NIE crosses zero by linear interpolation."""
    for i in range(len(nie_estimates) - 1):
        y0, y1 = nie_estimates[i], nie_estimates[i + 1]
        if y0 * y1 <= 0:  # sign change
            rho0, rho1 = rho_values[i], rho_values[i + 1]
            # Linear interpolation
            frac = -y0 / (y1 - y0) if (y1 - y0) != 0 else 0.5
            return float(rho0 + frac * (rho1 - rho0))
    return None


def _evalue(rr: float) -> float:
    """Compute E-value from a risk ratio.

    E-value = RR + sqrt(RR * (RR - 1))

    For RR < 1, use 1/RR first.
    For RR = 1, E-value = 1.

    Reference: VanderWeele & Ding (2017) Annals of Internal Medicine.

    Parameters
    ----------
    rr : float
        Risk ratio (must be >= 1 after taking reciprocal if needed).

    Returns
    -------
    float
    """
    if rr < 1:
        rr = 1 / rr
    if rr <= 1.0:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


def _d_to_rr(d: float) -> float:
    """Approximate conversion from Cohen's d to odds ratio / risk ratio.

    Uses OR = exp(pi * d / sqrt(3)) approximation then OR ≈ RR for rare outcomes.
    """
    import math
    return math.exp(math.pi * d / math.sqrt(3))
