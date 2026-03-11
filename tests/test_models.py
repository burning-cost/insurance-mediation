"""
Tests for GLM model wrappers (OutcomeModel, MediatorModel).
"""
import numpy as np
import pandas as pd
import pytest

from insurance_mediation.models import (
    OutcomeModel,
    MediatorModel,
    build_outcome_model,
    build_mediator_model,
)
from tests.conftest import make_poisson_data, make_gaussian_data, make_gamma_data


class TestOutcomeModel:
    def test_fit_poisson(self):
        data = make_poisson_data(n=500)
        model = OutcomeModel(
            model_type="poisson",
            formula="outcome ~ treatment + mediator + confounder",
            exposure_col="exposure",
        )
        model.fit(data, "outcome")
        assert model.is_fitted

    def test_fit_gaussian(self):
        data = make_gaussian_data(n=500)
        model = OutcomeModel(
            model_type="gaussian",
            formula="outcome ~ treatment + mediator + confounder",
        )
        model.fit(data, "outcome")
        assert model.is_fitted

    def test_fit_gamma(self):
        data = make_gamma_data(n=500)
        model = OutcomeModel(
            model_type="gamma",
            formula="outcome ~ treatment + mediator + confounder",
        )
        model.fit(data, "outcome")
        assert model.is_fitted

    def test_predict_returns_correct_shape(self):
        data = make_poisson_data(n=500)
        model = OutcomeModel(
            model_type="poisson",
            formula="outcome ~ treatment + mediator + confounder",
            exposure_col="exposure",
        )
        model.fit(data, "outcome")
        preds = model.predict(data, "treatment", 1, "mediator")
        assert preds.shape == (500,)

    def test_predict_counterfactual_treatment(self):
        data = make_poisson_data(n=500)
        model = OutcomeModel(
            model_type="poisson",
            formula="outcome ~ treatment + mediator + confounder",
            exposure_col="exposure",
        )
        model.fit(data, "outcome")
        preds_treat = model.predict(data, "treatment", 1, "mediator")
        preds_ctrl = model.predict(data, "treatment", 0, "mediator")
        # Treatment should generally produce higher predictions (true effect > 0)
        assert np.mean(preds_treat) > np.mean(preds_ctrl)

    def test_predict_counterfactual_mediator(self):
        data = make_poisson_data(n=500)
        model = OutcomeModel(
            model_type="poisson",
            formula="outcome ~ treatment + mediator + confounder",
            exposure_col="exposure",
        )
        model.fit(data, "outcome")
        preds_m5 = model.predict(data, "treatment", 0, "mediator", mediator_value=5.0)
        preds_m10 = model.predict(data, "treatment", 0, "mediator", mediator_value=10.0)
        # Higher mediator -> higher outcomes (positive coefficient)
        assert np.mean(preds_m10) > np.mean(preds_m5)

    def test_predict_positive_values(self):
        data = make_poisson_data(n=500)
        model = OutcomeModel(
            model_type="poisson",
            formula="outcome ~ treatment + mediator + confounder",
            exposure_col="exposure",
        )
        model.fit(data, "outcome")
        preds = model.predict(data, "treatment", 1, "mediator")
        assert np.all(preds > 0)

    def test_has_log_link(self):
        for mtype in ("poisson", "gamma", "tweedie"):
            m = OutcomeModel(model_type=mtype)
            assert m.has_log_link is True

    def test_no_log_link_gaussian(self):
        m = OutcomeModel(model_type="gaussian")
        assert m.has_log_link is False

    def test_not_fitted_raises(self):
        model = OutcomeModel(model_type="poisson", formula="y ~ x")
        data = make_poisson_data(n=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(data, "treatment", 1, "mediator")

    def test_fit_without_formula_raises(self):
        model = OutcomeModel(model_type="poisson")
        data = make_poisson_data(n=10)
        with pytest.raises(ValueError, match="formula must be set"):
            model.fit(data, "outcome")

    def test_from_fit_result(self):
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        data = make_poisson_data(n=300)
        offset = np.log(data["exposure"].clip(lower=1e-9))
        fit = smf.glm(
            "outcome ~ treatment + mediator + confounder",
            data=data,
            family=sm.families.Poisson(),
            offset=offset,
        ).fit(disp=False)

        model = OutcomeModel.from_fit_result(fit, model_type="poisson", exposure_col="exposure")
        assert model.is_fitted
        preds = model.predict(data, "treatment", 1, "mediator")
        assert preds.shape == (300,)

    def test_predict_linear_returns_log_scale(self):
        data = make_poisson_data(n=300)
        model = OutcomeModel(
            model_type="poisson",
            formula="outcome ~ treatment + mediator + confounder",
            exposure_col="exposure",
        )
        model.fit(data, "outcome")
        eta = model.predict_linear(data, "treatment", 1, "mediator")
        # Log scale values should be negative (log of small rate)
        assert np.mean(eta) < 0


class TestMediatorModel:
    def test_fit_linear(self):
        data = make_poisson_data(n=500)
        model = MediatorModel(
            model_type="linear",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        assert model.is_fitted

    def test_fit_logistic(self):
        data = pd.DataFrame({
            "treatment": np.random.binomial(1, 0.5, 500).astype(float),
            "mediator": np.random.binomial(1, 0.3, 500).astype(float),
            "confounder": np.random.uniform(0, 10, 500),
        })
        model = MediatorModel(
            model_type="logistic",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        assert model.is_fitted

    def test_predict_mean_shape(self):
        data = make_poisson_data(n=500)
        model = MediatorModel(
            model_type="linear",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        mu = model.predict_mean(data, "treatment", 1)
        assert mu.shape == (500,)

    def test_predict_mean_treatment_effect(self):
        data = make_poisson_data(n=500)
        model = MediatorModel(
            model_type="linear",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        mu_1 = model.predict_mean(data, "treatment", 1)
        mu_0 = model.predict_mean(data, "treatment", 0)
        # True beta_A_M = 1.5 > 0
        assert np.mean(mu_1) > np.mean(mu_0)
        assert np.mean(mu_1) - np.mean(mu_0) == pytest.approx(1.5, abs=0.3)

    def test_sample_shape(self):
        data = make_poisson_data(n=100)
        model = MediatorModel(
            model_type="linear",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        rng = np.random.default_rng(0)
        samples = model.sample(data, "treatment", 0, n_samples=5, rng=rng)
        assert samples.shape == (5, 100)

    def test_sample_logistic_binary(self):
        data = pd.DataFrame({
            "treatment": np.random.binomial(1, 0.5, 500).astype(float),
            "mediator": np.random.binomial(1, 0.3, 500).astype(float),
            "confounder": np.random.uniform(0, 10, 500),
        })
        model = MediatorModel(
            model_type="logistic",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        rng = np.random.default_rng(0)
        samples = model.sample(data, "treatment", 0, n_samples=10, rng=rng)
        # Samples should be 0 or 1
        assert set(np.unique(samples)).issubset({0.0, 1.0})

    def test_residual_std_positive(self):
        data = make_poisson_data(n=500)
        model = MediatorModel(
            model_type="linear",
            formula="mediator ~ treatment + confounder",
        )
        model.fit(data, "mediator")
        assert model.residual_std > 0

    def test_fit_without_formula_raises(self):
        model = MediatorModel(model_type="linear")
        data = make_poisson_data(n=10)
        with pytest.raises(ValueError, match="formula must be set"):
            model.fit(data, "mediator")


class TestBuildModels:
    def test_build_outcome_poisson(self):
        data = make_poisson_data(n=500)
        model = build_outcome_model(
            model_type="poisson",
            data=data,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
            covariates=["confounder"],
            exposure_col="exposure",
        )
        assert model.is_fitted

    def test_build_outcome_no_covariates(self):
        data = make_poisson_data(n=500)
        model = build_outcome_model(
            model_type="poisson",
            data=data,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
            covariates=[],
            exposure_col="exposure",
        )
        assert model.is_fitted

    def test_build_mediator_linear(self):
        data = make_poisson_data(n=500)
        model = build_mediator_model(
            model_type="linear",
            data=data,
            mediator_col="mediator",
            treatment_col="treatment",
            covariates=["confounder"],
        )
        assert model.is_fitted

    def test_build_outcome_formula_contains_interaction(self):
        data = make_poisson_data(n=500)
        model = build_outcome_model(
            model_type="poisson",
            data=data,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
            covariates=["confounder"],
            interaction=True,
        )
        assert "treatment:mediator" in model.formula or "mediator:treatment" in model.formula

    def test_build_outcome_formula_no_interaction(self):
        data = make_poisson_data(n=500)
        model = build_outcome_model(
            model_type="poisson",
            data=data,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
            covariates=["confounder"],
            interaction=False,
        )
        assert ":" not in model.formula or "treatment:mediator" not in model.formula
