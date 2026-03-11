# insurance-mediation

Causal mediation analysis for insurance pricing fairness.

## The problem

Your motor insurer prices by postcode. The FCA asks whether your postcode ratings
are acting as a proxy for ethnicity or religion. You say no — postcodes reflect
genuine risk differences like crime, flood exposure, and socioeconomic deprivation.
But how much of your postcode differential is actually *explained* by those legitimate
factors, versus being a direct postcode effect that cannot be justified?

That's a mediation question. You want to decompose:

```
Postcode → (IMD deprivation) → Claim rate
    ↘_________________________↗
           direct path
```

The **indirect effect** (through IMD) is defensible: postcodes with worse deprivation
have genuinely different risk profiles and IMD is a legitimate rating factor. The
**direct effect** (postcode on claims, holding IMD fixed) is the part that requires
scrutiny — if it's substantial, postcode is doing something that deprivation alone
doesn't explain.

This library implements that decomposition with proper causal inference methodology
for GLM-based insurance models (Poisson, Gamma, Tweedie) and produces an FCA-ready
audit report.

## Three estimands, because the methodology choice matters

**CDE (Controlled Direct Effect)** — default for FCA compliance.

> "If we intervened to set everyone's IMD deprivation to the national median,
> what would the remaining postcode price differential be?"

CDE requires the weakest causal assumptions (no cross-world counterfactuals needed).
If CDE ≈ 0, the postcode effect is fully explained by deprivation. If CDE is large,
postcode is doing something beyond deprivation.

**NDE/NIE (Natural Direct/Indirect Effects)** — academic decomposition.

Decomposes the total postcode effect into:
- NIE: the part that operates *through* IMD
- NDE: the part that does not

NDE + NIE = Total Effect. Requires stronger assumptions (sequential ignorability
plus no treatment-induced mediator-outcome confounders). More informative but more
assumptions to defend.

**Total Effect** — the overall A → Y causal effect, estimated from the fitted
outcome model.

The library is opinionated: CDE is the right default for regulatory work because
you can actually defend the assumptions in front of the FCA. NDE/NIE are provided
for completeness and academic reporting.

## Quick start

```python
from insurance_mediation import MediationAnalysis

ma = MediationAnalysis(
    outcome_model="poisson",   # or "gamma", "tweedie", "gaussian"
    mediator_model="linear",   # or "logistic" for binary mediator (flood zone)
    exposure_col="exposure",   # for Poisson/Tweedie: log(exposure) offset
    n_mc_samples=1000,         # Monte Carlo samples for NDE/NIE
    n_bootstrap=500,           # bootstrap replicates for CIs
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

# Controlled Direct Effect: what if IMD were equalised to decile 5?
print(results.cde(mediator_level=5))
# EffectEstimate(CDE: +0.0821 [+0.0312, +0.1330] *)

# Natural effects decomposition
print(results.nde())   # EffectEstimate(NDE: +0.0834 [...])
print(results.nie())   # EffectEstimate(NIE: +0.0624 [...])
print(results.total_effect())  # EffectEstimate(TE: +0.1458 [...])

# How much is explained by IMD?
te = results.total_effect().effect
nie = results.nie().effect
print(f"{nie/te:.0%} of the price differential operates through IMD")

# Sensitivity: how strong must unmeasured confounding be to explain away the NIE?
results.sensitivity(rho_range=(-0.5, 0.5))

# FCA-ready HTML report
results.report(
    protected_attribute="ethnicity",
    title="Postcode-IMD Mediation Analysis — Motor 2024",
    output="mediation_report.html"
)
```

## GLM families

The library handles the non-linearity of insurance GLMs correctly. For Poisson,
Gamma, and Tweedie models the indirect effect is *not* computed as a product
of regression coefficients (the Baron & Kenny approach) — that only works for
linear models. Instead, NDE/NIE are estimated via Monte Carlo integration over
the mediator distribution (VanderWeele 2015, extended to GLMs).

| Outcome | Family | Link | Use case |
|---------|--------|------|----------|
| `"poisson"` | Poisson | log | Claim frequency |
| `"gamma"` | Gamma | log | Claim severity |
| `"tweedie"` | Tweedie (var_power=1.5) | log | Burning cost |
| `"gaussian"` | Gaussian | identity | Benchmarking, continuous outcomes |

Effects for log-link models are on the log-mean scale (log rate ratios). Use
`result.nde().ratio` to get the multiplicative relativity (e.g., 1.09 = 9% higher
claim rate).

## Pre-fitted models

You can pass a pre-fitted statsmodels GLM instead of a family string. This is useful
when you have a production pricing model with interactions, splines, and offsets that
you want to use directly:

```python
import statsmodels.api as sm

# Your production GLM (already fitted)
fitted_glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

ma = MediationAnalysis(
    outcome_model=fitted_glm,
    mediator_model="linear",
    exposure_col="exposure",
)
```

Note: bootstrap confidence intervals require the model to be refittable from a
formula string. Pre-fitted models without a stored formula will produce point
estimates only.

## Sensitivity analysis

Sequential ignorability — the key assumption for causal mediation — cannot be
verified from data. The sensitivity analysis shows how robust the NIE is to
violations of this assumption.

Two tools are provided:

**Imai et al. (2010) rho sensitivity**: varies the residual correlation between
outcome and mediator models (rho). Under sequential ignorability, rho = 0. The
analysis reports the rho at which the NIE would cross zero.

**E-value (VanderWeele & Ding 2017)**: the minimum risk ratio of an unmeasured
confounder (with both treatment assignment and the outcome) that would be needed
to explain away the NIE. E > 2 is generally considered a robust finding.

```python
sens = results.sensitivity(rho_range=(-0.5, 0.5))
print(sens)
# SensitivityResult(E-value=2.847, E-value(CI)=1.923, rho_at_zero=0.312)
```

## FCA report

The HTML report is designed to be attached to a pricing actuarial function report
or submitted as evidence for FCA data ethics review. It includes:

- Causal DAG (SVG, self-contained)
- Effect decomposition table (CDE, NDE, NIE, TE) with 95% CIs on log and ratio scale
- Fairness interpretation (percentage of differential explained by mediator)
- Sensitivity analysis table
- Identification assumptions (plain English, per estimand)
- Template Section 19 proportionality statement

No binary dependencies (wkhtmltopdf, headless Chrome). Pure Python / Jinja2.

## Causal identification assumptions

Be honest about what this requires:

**For CDE** (weakest):
- C1: No unmeasured treatment-outcome confounders given covariates
- C2: No unmeasured mediator-outcome confounders given (treatment, covariates)
- C3: No unmeasured treatment-mediator confounders given covariates

C4 (no treatment-induced mediator-outcome confounders) is **not** required for CDE.
This is why CDE is the preferred estimand for regulatory use.

**For NDE/NIE** (stronger, adds):
- C4: No treatment-induced mediator-outcome confounders. In practice: no other
  postcode-level variable (beyond IMD) that also confounds the IMD → claim rate
  relationship. This is harder to defend when postcodes affect crime, flood,
  housing density, and all of these correlate with IMD.

If you can't defend C4, report CDE and acknowledge it.

## Installation

```bash
pip install insurance-mediation
```

Dependencies: numpy, scipy, pandas, statsmodels, jinja2. No heavy ML deps.

Optional visualisation: `pip install insurance-mediation[viz]` adds matplotlib
and networkx for causal DAG plots.

## Scope

v0.1.0: single mediator. Multiple simultaneous mediators (e.g., IMD *and* flood zone
*and* crime rate) are deferred to v0.2.0 where we'll use interventional indirect
effects (VanderWeele & Vansteelandt 2009) which avoid the cross-world counterfactual
issues that arise with multiple mediators.

## Methodology references

- VanderWeele, T.J. (2015). *Explanation in Causal Inference*. Oxford University Press.
- Imai, K., Keele, L., Tingley, D. (2010). A general approach to causal mediation
  analysis. *Psychological Methods* 15(4): 309–334.
- VanderWeele, T.J. & Ding, P. (2017). Sensitivity analysis in observational research:
  introducing the E-value. *Annals of Internal Medicine* 167(4): 268–274.
- Robins, J.M. & Greenland, S. (1992). Identifiability and exchangeability for
  direct and indirect effects. *Epidemiology* 3(2): 143–155.
- Jackson, J.W. (2021). Meaningful causal decompositions in health equity research.
  *Epidemiology* 32(2): 230–239.
