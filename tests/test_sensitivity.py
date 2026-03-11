"""
Tests for sensitivity analysis (Imai et al. rho + E-value).
"""
import math
import numpy as np
import pytest

from insurance_mediation.sensitivity import compute_sensitivity, _evalue, _find_rho_at_zero
from insurance_mediation.estimands import EffectEstimate


class TestEValue:
    def test_evalue_rr_equals_one(self):
        """RR = 1 means no effect — E-value should be 1."""
        assert _evalue(1.0) == pytest.approx(1.0)

    def test_evalue_rr_two(self):
        """RR = 2: E = 2 + sqrt(2*1) = 2 + sqrt(2) ≈ 3.414."""
        expected = 2 + math.sqrt(2)
        assert _evalue(2.0) == pytest.approx(expected)

    def test_evalue_rr_less_than_one(self):
        """E-value for RR < 1 should use 1/RR."""
        e_05 = _evalue(0.5)
        e_2 = _evalue(2.0)
        assert e_05 == pytest.approx(e_2)

    def test_evalue_increases_with_rr(self):
        evalues = [_evalue(r) for r in [1.5, 2.0, 3.0, 5.0]]
        assert all(evalues[i] < evalues[i + 1] for i in range(len(evalues) - 1))

    def test_evalue_large_rr(self):
        """For large RR, E-value ≈ 2 * RR."""
        rr = 10.0
        e = _evalue(rr)
        assert e < 2 * rr  # approximate bound
        assert e > rr  # always larger than RR

    def test_evalue_rr_near_one(self):
        """Just above 1."""
        e = _evalue(1.001)
        assert e > 1.0


class TestFindRhoAtZero:
    def test_finds_zero_crossing(self):
        rho_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
        # NIE goes from positive to negative
        nie_estimates = [0.3, 0.15, 0.0, -0.15, -0.3]
        rho_z = _find_rho_at_zero(rho_values, nie_estimates)
        assert rho_z == pytest.approx(0.0, abs=0.01)

    def test_finds_positive_rho_crossing(self):
        rho_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
        nie_estimates = [0.2, 0.15, 0.1, 0.0, -0.05]
        rho_z = _find_rho_at_zero(rho_values, nie_estimates)
        assert rho_z == pytest.approx(0.25, abs=0.05)

    def test_returns_none_when_no_crossing(self):
        rho_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
        nie_estimates = [0.1, 0.12, 0.15, 0.18, 0.2]
        rho_z = _find_rho_at_zero(rho_values, nie_estimates)
        assert rho_z is None

    def test_returns_none_all_negative(self):
        rho_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
        nie_estimates = [-0.3, -0.2, -0.1, -0.05, -0.02]
        rho_z = _find_rho_at_zero(rho_values, nie_estimates)
        assert rho_z is None

    def test_handles_two_point_crossing(self):
        rho_values = [0.0, 0.5]
        nie_estimates = [0.1, -0.1]
        rho_z = _find_rho_at_zero(rho_values, nie_estimates)
        assert rho_z == pytest.approx(0.25, abs=0.01)


class TestComputeSensitivity:
    @pytest.fixture(scope="class")
    def fitted_models_and_nie(self):
        from tests.conftest import make_poisson_data
        from insurance_mediation.models import build_outcome_model, build_mediator_model
        from insurance_mediation.estimators import estimate_nde_nie

        data = make_poisson_data(n=1000, seed=5)
        outcome_model = build_outcome_model(
            model_type="poisson",
            data=data,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
            covariates=["confounder"],
            exposure_col="exposure",
            interaction=False,
        )
        mediator_model = build_mediator_model(
            model_type="linear",
            data=data,
            mediator_col="mediator",
            treatment_col="treatment",
            covariates=["confounder"],
        )
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=100,
            n_bootstrap=10,
        )
        return data, outcome_model, mediator_model, nie

    def test_sensitivity_returns_result(self, fitted_models_and_nie):
        from insurance_mediation.estimands import SensitivityResult
        data, outcome_model, mediator_model, nie = fitted_models_and_nie

        result = compute_sensitivity(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            nie_estimate=nie,
            rho_range=(-0.3, 0.3),
            n_rho=5,
            n_mc_samples=50,
            n_bootstrap=10,
        )
        assert isinstance(result, SensitivityResult)

    def test_sensitivity_rho_values_count(self, fitted_models_and_nie):
        data, outcome_model, mediator_model, nie = fitted_models_and_nie

        result = compute_sensitivity(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            nie_estimate=nie,
            rho_range=(-0.3, 0.3),
            n_rho=7,
            n_mc_samples=50,
            n_bootstrap=10,
        )
        assert len(result.rho_values) == 7
        assert len(result.nie_estimates) == 7

    def test_sensitivity_evalue_positive(self, fitted_models_and_nie):
        data, outcome_model, mediator_model, nie = fitted_models_and_nie

        result = compute_sensitivity(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            nie_estimate=nie,
            rho_range=(-0.3, 0.3),
            n_rho=5,
            n_mc_samples=50,
            n_bootstrap=10,
        )
        assert result.e_value > 1.0

    def test_sensitivity_nie_at_rho0_matches_input(self, fitted_models_and_nie):
        data, outcome_model, mediator_model, nie = fitted_models_and_nie

        result = compute_sensitivity(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            nie_estimate=nie,
            rho_range=(-0.3, 0.3),
            n_rho=5,
            n_mc_samples=50,
            n_bootstrap=10,
        )
        assert result.nie_at_rho0.effect == pytest.approx(nie.effect)
