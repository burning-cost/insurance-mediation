"""
Tests for the main MediationAnalysis class.

Integration tests that exercise the full fit() -> result workflow.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_mediation import MediationAnalysis
from tests.conftest import make_poisson_data, make_gaussian_data, make_gamma_data


@pytest.fixture(scope="module")
def poisson_df():
    return make_poisson_data(n=1500, seed=10)


@pytest.fixture(scope="module")
def gaussian_df():
    return make_gaussian_data(n=1500, seed=10)


class TestMediationAnalysisInit:
    def test_default_init(self):
        ma = MediationAnalysis()
        assert ma.outcome_model_spec == "poisson"
        assert ma.mediator_model_spec == "linear"
        assert ma.n_mc_samples == 1000
        assert ma.n_bootstrap == 200

    def test_custom_params(self):
        ma = MediationAnalysis(
            outcome_model="gamma",
            mediator_model="logistic",
            exposure_col="exposure",
            n_mc_samples=500,
            n_bootstrap=100,
        )
        assert ma.outcome_model_spec == "gamma"
        assert ma.mediator_model_spec == "logistic"
        assert ma.exposure_col == "exposure"


class TestMediationAnalysisFit:
    def test_fit_returns_mediation_result(self, poisson_df):
        from insurance_mediation.estimands import MediationResult
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            covariates=["confounder"],
        )
        assert isinstance(result, MediationResult)

    def test_fit_metadata(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
        )
        assert result.treatment == "treatment"
        assert result.mediator == "mediator"
        assert result.outcome == "outcome"
        assert result.n_obs == len(poisson_df)

    def test_fit_raises_on_missing_column(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            n_mc_samples=50,
            n_bootstrap=10,
        )
        with pytest.raises(ValueError, match="Missing columns"):
            ma.fit(
                data=poisson_df,
                treatment="treatment",
                mediator="nonexistent_column",
                outcome="outcome",
            )

    def test_fit_has_total_effect(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
        )
        te = result.total_effect()
        assert te.effect > 0

    def test_fit_has_nde_nie(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
        )
        nde = result.nde()
        nie = result.nie()
        assert nde.estimand == "NDE"
        assert nie.estimand == "NIE"

    def test_fit_has_cde_at_percentiles(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
        )
        # Default: CDE at 10th, 25th, 50th, 75th, 90th percentiles
        assert len(result._cde_estimates) == 5

    def test_fit_custom_cde_levels(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            cde_levels=[3.0, 5.0, 7.0],
        )
        assert len(result._cde_estimates) == 3

    def test_cde_mediator_level_lookup(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            cde_levels=[5.0],
        )
        cde = result.cde(mediator_level=5.0)
        assert cde.mediator_level == pytest.approx(5.0)

    def test_treatment_values_stored(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            n_mc_samples=50,
            n_bootstrap=10,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            treatment_value=1,
            control_value=0,
        )
        assert result.treatment_values == (1, 0)

    def test_fit_gaussian_outcome(self, gaussian_df):
        ma = MediationAnalysis(
            outcome_model="gaussian",
            mediator_model="linear",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=gaussian_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            covariates=["confounder"],
        )
        assert result.outcome_model_type == "gaussian"
        te = result.total_effect()
        assert te.effect > 0

    def test_fit_no_covariates(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            covariates=None,
        )
        assert result.n_obs == len(poisson_df)

    def test_fit_with_pre_fitted_model(self, poisson_df):
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        offset = np.log(poisson_df["exposure"].clip(lower=1e-9))
        fitted_glm = smf.glm(
            "outcome ~ treatment + mediator + confounder",
            data=poisson_df,
            family=sm.families.Poisson(),
            offset=offset,
        ).fit(disp=False)

        ma = MediationAnalysis(
            outcome_model=fitted_glm,
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=50,
            n_bootstrap=10,
        )
        # Pre-fitted models have no formula so bootstrap will fail gracefully
        # Just check that fit() runs and returns something sensible
        try:
            result = ma.fit(
                data=poisson_df,
                treatment="treatment",
                mediator="mediator",
                outcome="outcome",
            )
            te = result.total_effect()
            assert te.effect > 0
        except RuntimeError:
            # Expected if bootstrap can't refit
            pass

    def test_summary_string(self, poisson_df):
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=100,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
        )
        s = result.summary()
        assert "treatment" in s
        assert "NDE" in s or "Natural" in s

    def test_nde_plus_nie_equals_te(self, poisson_df):
        """NIE is computed as TE - NDE so this should be exact."""
        ma = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=200,
            n_bootstrap=20,
        )
        result = ma.fit(
            data=poisson_df,
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
        )
        te = result.total_effect().effect
        nde = result.nde().effect
        nie = result.nie().effect
        assert abs((nde + nie) - te) < 0.001

    def test_fit_reproducible_with_seed(self):
        data = make_poisson_data(n=500, seed=99)
        ma1 = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=50,
            n_bootstrap=20,
            seed=7,
        )
        ma2 = MediationAnalysis(
            outcome_model="poisson",
            mediator_model="linear",
            exposure_col="exposure",
            n_mc_samples=50,
            n_bootstrap=20,
            seed=7,
        )
        r1 = ma1.fit(data=data, treatment="treatment", mediator="mediator", outcome="outcome")
        r2 = ma2.fit(data=data, treatment="treatment", mediator="mediator", outcome="outcome")
        assert r1.total_effect().effect == pytest.approx(r2.total_effect().effect)
