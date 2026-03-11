"""
insurance-mediation: Causal mediation analysis for insurance pricing fairness.

Decomposes total causal effects (e.g., postcode price differentials) into
direct and indirect (mediated) components. Primary use case is FCA proxy
discrimination analysis — how much of a postcode-driven price variation runs
through a legitimate mediator like IMD deprivation score versus a direct
postcode effect that cannot be explained by that mediator.

Three estimands are provided:

CDE (Controlled Direct Effect) — default for FCA compliance.
    "What if we set everyone's IMD deprivation to the same level?
    What price differential would remain?"
    Requires weakest causal assumptions. Directly actionable for pricing teams.

NDE/NIE (Natural Direct/Indirect Effects) — academic decomposition.
    Decomposes the total effect into the component that operates through the
    mediator and the component that does not. Requires cross-world
    counterfactual assumptions.

Interventional Effects — stochastic intervention framework.
    Avoids cross-world assumptions. Better for multiple-mediator settings.
    Not NDE+NIE=Total in general when mediators are correlated.

Quick start::

    from insurance_mediation import MediationAnalysis

    ma = MediationAnalysis(
        outcome_model="poisson",
        mediator_model="linear",
        exposure_col="exposure",
    )
    results = ma.fit(
        data=df,
        treatment="postcode_group",
        mediator="imd_decile",
        outcome="claim_count",
        covariates=["vehicle_age", "driver_age", "cover_type"],
    )
    print(results.cde(mediator_level=5))
    print(results.nde())
    print(results.nie())
    results.sensitivity(rho_range=(-0.5, 0.5))
    results.report(output="mediation_report.html")
"""

from insurance_mediation.core import MediationAnalysis
from insurance_mediation.estimands import (
    EffectEstimate,
    MediationResult,
    SensitivityResult,
)

__version__ = "0.1.0"
__all__ = [
    "MediationAnalysis",
    "EffectEstimate",
    "MediationResult",
    "SensitivityResult",
]
