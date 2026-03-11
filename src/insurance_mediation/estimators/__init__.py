"""
Mediation effect estimators.

Three estimators are provided, each implementing a different estimand:

regression.py — VanderWeele regression-based approach
    CDE via plug-in regression (outcome model evaluated at fixed mediator level).
    NDE/NIE via Monte Carlo integration over mediator distribution.
    Standard errors via bootstrap.

ipw.py — Inverse probability weighting
    CDE via IPTW/IPMT weighting.
    Less model-dependent than regression approach but requires good
    propensity models. Included for robustness checks.

doubly_robust.py — Tchetgen Tchetgen efficient influence function
    Doubly robust: consistent if either the outcome model or the mediator
    propensity model is correctly specified (not both required).
    Most efficient estimator asymptotically.
"""

from insurance_mediation.estimators.regression import (
    estimate_cde,
    estimate_nde_nie,
    estimate_total_effect,
)

__all__ = [
    "estimate_cde",
    "estimate_nde_nie",
    "estimate_total_effect",
]
