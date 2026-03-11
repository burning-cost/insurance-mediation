# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-mediation: FCA Proxy Discrimination Analysis
# MAGIC
# MAGIC **Scenario**: A motor insurer prices by postcode district. The FCA is asking
# MAGIC whether postcode acts as a proxy for ethnicity. The insurer argues that postcode
# MAGIC captures genuine risk differences, primarily through IMD (Index of Multiple
# MAGIC Deprivation) — a legitimate rating factor.
# MAGIC
# MAGIC This notebook demonstrates how to use `insurance-mediation` to:
# MAGIC 1. Decompose the postcode effect into direct vs. IMD-mediated components
# MAGIC 2. Quantify what fraction of the price differential is "legitimate"
# MAGIC 3. Assess sensitivity to unmeasured confounding
# MAGIC 4. Generate an FCA-ready HTML report

# COMMAND ----------

# MAGIC %pip install insurance-mediation statsmodels jinja2

# COMMAND ----------

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from insurance_mediation import MediationAnalysis

print(f"insurance-mediation loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic insurance data
# MAGIC
# MAGIC We simulate a realistic motor insurance portfolio with known causal structure.
# MAGIC
# MAGIC **True DGP**:
# MAGIC - `postcode_group` (binary: deprived vs affluent area) is the treatment
# MAGIC - `imd_decile` is the mediator — correlated with postcode, affects claims
# MAGIC - `vehicle_age`, `driver_age`, `cover_type` are confounders
# MAGIC - `claim_count` ~ Poisson(mu * exposure)
# MAGIC
# MAGIC **True effects (log scale)**:
# MAGIC - Direct A → Y: +0.15 (postcode effect not through IMD)
# MAGIC - Indirect A → M → Y: IMD coefficient × treatment-on-IMD = 0.08 × 2.0 = +0.16
# MAGIC - Total Effect ≈ +0.31

# COMMAND ----------

np.random.seed(42)
n = 10_000

# Confounders
vehicle_age = np.random.choice([1, 2, 3, 4, 5, 6, 7], n, p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1]).astype(float)
driver_age = np.random.normal(40, 12, n).clip(18, 75)
cover_type = np.random.choice([0, 1], n, p=[0.6, 0.4]).astype(float)  # 0=TPO, 1=Comprehensive

# Treatment: postcode group (0=affluent, 1=deprived)
p_deprived = 1 / (1 + np.exp(-(0.0 + 0.02 * vehicle_age - 0.005 * driver_age)))
postcode_group = np.random.binomial(1, p_deprived)

# Mediator: IMD decile (1=most deprived, 10=least deprived)
# Deprived postcodes have lower IMD decile (higher deprivation)
imd_noise = np.random.normal(0, 1.5, n)
imd_decile = (8 - 2.0 * postcode_group - 0.05 * vehicle_age + 0.02 * driver_age + imd_noise).clip(1, 10)

# Exposure (years at risk)
exposure = np.random.uniform(0.3, 2.0, n)

# Claim frequency model
# True coefficients:
#   postcode_group: +0.15 (direct effect)
#   imd_decile:     -0.08 (lower IMD = higher deprivation = more claims)
#   vehicle_age:    +0.04
#   driver_age:     -0.015
#   cover_type:     +0.20
log_mu = (
    -3.0
    + 0.15 * postcode_group
    - 0.08 * imd_decile
    + 0.04 * vehicle_age
    - 0.015 * driver_age
    + 0.20 * cover_type
    + np.log(exposure)
)
mu = np.exp(log_mu)
claim_count = np.random.poisson(mu)

df = pd.DataFrame({
    "postcode_group": postcode_group.astype(float),
    "imd_decile": imd_decile,
    "claim_count": claim_count,
    "vehicle_age": vehicle_age,
    "driver_age": driver_age,
    "cover_type": cover_type,
    "exposure": exposure,
})

print(f"Dataset: {len(df):,} policies")
print(f"Claim rate: {df.claim_count.sum() / df.exposure.sum():.4f} per policy-year")
print(f"Postcode 1 (deprived) claim rate: {df.query('postcode_group==1').claim_count.sum() / df.query('postcode_group==1').exposure.sum():.4f}")
print(f"Postcode 0 (affluent) claim rate: {df.query('postcode_group==0').claim_count.sum() / df.query('postcode_group==0').exposure.sum():.4f}")
print(f"Raw rate ratio: {(df.query('postcode_group==1').claim_count.sum() / df.query('postcode_group==1').exposure.sum()) / (df.query('postcode_group==0').claim_count.sum() / df.query('postcode_group==0').exposure.sum()):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run mediation analysis
# MAGIC
# MAGIC We use the default CDE estimand (strongest regulatory justification)
# MAGIC plus NDE/NIE for the full decomposition.

# COMMAND ----------

ma = MediationAnalysis(
    outcome_model="poisson",
    mediator_model="linear",
    exposure_col="exposure",
    n_mc_samples=1000,
    n_bootstrap=300,
    seed=42,
)

results = ma.fit(
    data=df,
    treatment="postcode_group",
    mediator="imd_decile",
    outcome="claim_count",
    covariates=["vehicle_age", "driver_age", "cover_type"],
    treatment_value=1,     # deprived postcode
    control_value=0,       # affluent postcode
    cde_levels=[3.0, 5.5, 8.0],  # low, median, high IMD decile
)

print(results.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Effect decomposition

# COMMAND ----------

te = results.total_effect()
nde = results.nde()
nie = results.nie()

print("=" * 60)
print("EFFECT DECOMPOSITION (log rate ratio scale)")
print("=" * 60)
print(f"Total Effect (TE):           {te.effect:+.4f}  [{te.ci_lower:+.4f}, {te.ci_upper:+.4f}]")
print(f"Natural Direct Effect (NDE): {nde.effect:+.4f}  [{nde.ci_lower:+.4f}, {nde.ci_upper:+.4f}]")
print(f"Natural Indirect Effect:     {nie.effect:+.4f}  [{nie.ci_lower:+.4f}, {nie.ci_upper:+.4f}]")
print()
print(f"On ratio scale:")
print(f"  TE  = {te.ratio:.3f}x  ({(te.ratio-1)*100:.1f}% higher claim rate)")
print(f"  NDE = {nde.ratio:.3f}x  (direct postcode effect)")
print(f"  NIE = {nie.ratio:.3f}x  (through IMD pathway)")
print()
proportion_mediated = abs(nie.effect) / abs(te.effect)
print(f"Proportion mediated through IMD: {proportion_mediated:.1%}")
print(f"  → {'Most' if proportion_mediated > 0.5 else 'Less than half'} of the postcode differential is explained by IMD deprivation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Controlled Direct Effect at different IMD levels
# MAGIC
# MAGIC The CDE answers: "If we fixed everyone's IMD deprivation to the same level,
# MAGIC what postcode price differential would remain?"
# MAGIC
# MAGIC This is directly actionable for pricing: if CDE ≈ 0 at any IMD level,
# MAGIC the postcode effect is fully absorbed by IMD.

# COMMAND ----------

print("CONTROLLED DIRECT EFFECTS BY MEDIATOR LEVEL")
print("-" * 50)
print(f"{'IMD Decile':<15} {'CDE (log RR)':<15} {'95% CI':<25} {'Ratio':<10} {'Sig?'}")
print("-" * 75)
for m_level in [3.0, 5.5, 8.0]:
    cde = results.cde(mediator_level=m_level)
    sig = "*" if cde.significant else ""
    print(f"{m_level:<15.1f} {cde.effect:<+15.4f} [{cde.ci_lower:+.4f}, {cde.ci_upper:+.4f}]  {cde.ratio:<10.3f} {sig}")

print()
print("Note: CDE is relatively stable across IMD levels (no treatment:mediator")
print("interaction in this dataset). In real data with interactions, CDE can")
print("vary substantially across the mediator range.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sensitivity analysis
# MAGIC
# MAGIC Sequential ignorability (the key mediation assumption) can't be verified from data.
# MAGIC We assess robustness via:
# MAGIC - **rho**: residual correlation between outcome and mediator models (Imai et al. 2010)
# MAGIC - **E-value**: minimum unmeasured confounder effect to explain away the NIE

# COMMAND ----------

from insurance_mediation.sensitivity import compute_sensitivity
from insurance_mediation.models import build_outcome_model, build_mediator_model

# Refit models for sensitivity analysis
outcome_model = build_outcome_model(
    model_type="poisson",
    data=df,
    outcome_col="claim_count",
    treatment_col="postcode_group",
    mediator_col="imd_decile",
    covariates=["vehicle_age", "driver_age", "cover_type"],
    exposure_col="exposure",
    interaction=False,
)
mediator_model = build_mediator_model(
    model_type="linear",
    data=df,
    mediator_col="imd_decile",
    treatment_col="postcode_group",
    covariates=["vehicle_age", "driver_age", "cover_type"],
)

sens = compute_sensitivity(
    data=df,
    outcome_model=outcome_model,
    mediator_model=mediator_model,
    treatment_col="postcode_group",
    mediator_col="imd_decile",
    treatment_value=1,
    control_value=0,
    nie_estimate=nie,
    rho_range=(-0.5, 0.5),
    n_rho=11,
    n_mc_samples=200,
    n_bootstrap=50,
    rng=np.random.default_rng(99),
)

print(sens)
print()
print(f"E-value: {sens.e_value:.3f}")
print(f"  → An unmeasured confounder would need RR ≥ {sens.e_value:.2f} with BOTH")
print(f"    postcode assignment AND claim rate to explain away the IMD-mediated effect.")

if sens.rho_at_zero is not None:
    print(f"\nNIE crosses zero at rho = {sens.rho_at_zero:.3f}")
    print(f"  → Residual correlation of {abs(sens.rho_at_zero):.2f} between outcome and mediator")
    print(f"    errors would be needed to reverse the conclusion.")
else:
    print("\nNIE does not cross zero in the evaluated rho range.")
    print("The mediated effect is robust to the confounding levels examined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate FCA report

# COMMAND ----------

report_path = "/tmp/postcode_imd_mediation_report.html"
results.report(
    output=report_path,
    title="Motor Insurance — Postcode × IMD Mediation Analysis",
    protected_attribute="ethnicity / religion",
)
print(f"Report saved to: {report_path}")
print("Download from DBFS and open in browser for review.")

# In Databricks, copy to DBFS:
# dbutils.fs.cp(f"file:{report_path}", "dbfs:/mediation_reports/postcode_imd_2024.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. JSON export for audit trail

# COMMAND ----------

from insurance_mediation.report import generate_json_report

json_path = "/tmp/mediation_results.json"
generate_json_report(results, output=json_path)
print(f"JSON results saved to: {json_path}")

import json
with open(json_path) as f:
    audit_data = json.load(f)

print("\nKey metrics for audit trail:")
print(f"  Total Effect:  {audit_data['estimates']['total_effect']['effect']:.4f}")
print(f"  NDE:           {audit_data['estimates']['nde']['effect']:.4f}")
print(f"  NIE:           {audit_data['estimates']['nie']['effect']:.4f}")
print(f"  N bootstrap:   {audit_data['estimates']['nie']['n_bootstrap']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Using a production GLM (pre-fitted model)
# MAGIC
# MAGIC If you already have a fitted production pricing GLM, you can pass it directly
# MAGIC rather than re-specifying a formula. The library wraps the result and uses it
# MAGIC for counterfactual predictions.

# COMMAND ----------

# Fit a production-style GLM with more complexity
import numpy as np
production_offset = np.log(df["exposure"].clip(lower=1e-9))

production_glm = smf.glm(
    "claim_count ~ postcode_group + imd_decile + C(vehicle_age) + driver_age + cover_type",
    data=df,
    family=sm.families.Poisson(),
    offset=production_offset,
).fit(disp=False)

print("Production GLM fitted.")
print(production_glm.summary().tables[1])

# COMMAND ----------

# Now use this pre-fitted model in mediation analysis
# (bootstrap will be skipped for pre-fitted models without stored formula)
ma_prod = MediationAnalysis(
    outcome_model=production_glm,   # pre-fitted statsmodels result
    mediator_model="linear",
    exposure_col="exposure",
    n_mc_samples=500,
    n_bootstrap=0,  # No bootstrap for pre-fitted models
)

# Point estimates only (no CIs without bootstrap)
results_prod = ma_prod.fit(
    data=df,
    treatment="postcode_group",
    mediator="imd_decile",
    outcome="claim_count",
    covariates=["vehicle_age", "driver_age", "cover_type"],
    treatment_value=1,
    control_value=0,
)

te_prod = results_prod.total_effect()
print(f"Production model TE: {te_prod.effect:+.4f} (ratio: {te_prod.ratio:.3f}x)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated the full insurance-mediation workflow:
# MAGIC
# MAGIC | Step | Code |
# MAGIC |------|------|
# MAGIC | 1. Fit analysis | `ma.fit(data, treatment, mediator, outcome, covariates)` |
# MAGIC | 2. CDE (FCA default) | `results.cde(mediator_level=5.5)` |
# MAGIC | 3. NDE/NIE decomposition | `results.nde()`, `results.nie()` |
# MAGIC | 4. Sensitivity | `compute_sensitivity(...)` |
# MAGIC | 5. HTML report | `results.report(output="report.html")` |
# MAGIC | 6. JSON export | `generate_json_report(results, output="results.json")` |
# MAGIC
# MAGIC **For FCA compliance**: lead with the CDE. It makes the weakest assumptions
# MAGIC and has the clearest regulatory interpretation: "if we equalised deprivation
# MAGIC across all postcodes, what price differential would remain?"
# MAGIC
# MAGIC **Key finding from this analysis**: ~51% of the postcode differential operates
# MAGIC through the IMD deprivation pathway (legitimate). ~49% is a direct postcode
# MAGIC effect that requires further justification or remediation.
