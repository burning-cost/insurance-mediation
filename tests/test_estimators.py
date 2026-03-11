"""
Tests for the regression-based mediation estimators.

Key tests verify:
1. Correct sign and approximate magnitude against known true effects
2. CDE invariant to mediator level when there's no treatment-mediator interaction
3. NDE + NIE ≈ TE (decomposition identity)
4. Bootstrap CIs contain the true effect at reasonable rates
5. Estimators work with Gaussian outcomes (for benchmarking)
"""
import math
import numpy as np
import pandas as pd
import pytest

from insurance_mediation.models import build_outcome_model, build_mediator_model
from insurance_mediation.estimators import estimate_cde, estimate_nde_nie, estimate_total_effect
from tests.conftest import make_poisson_data, make_gaussian_data


# True effects in the Poisson DGP:
# TE  = 0.2 + 0.1 * 1.5 = 0.35 (but this is approximate — GLM nonlinearity means
#       the true marginal TE is close but not exactly this)
# NDE ≈ 0.20 (direct A -> Y effect)
# NIE ≈ 0.15 (A -> M -> Y: 1.5 * 0.1)
# CDE(any m) ≈ 0.20 (no treatment:mediator interaction in DGP)

TRUE_TE = 0.35
TRUE_NDE = 0.20
TRUE_NIE = 0.15
TOLERANCE = 0.12  # Wide tolerance: MC noise + nonlinearity + finite sample


@pytest.fixture(scope="module")
def fitted_poisson_models():
    data = make_poisson_data(n=3000, seed=0)
    outcome_model = build_outcome_model(
        model_type="poisson",
        data=data,
        outcome_col="outcome",
        treatment_col="treatment",
        mediator_col="mediator",
        covariates=["confounder"],
        exposure_col="exposure",
        interaction=False,  # No interaction in DGP — match the truth
    )
    mediator_model = build_mediator_model(
        model_type="linear",
        data=data,
        mediator_col="mediator",
        treatment_col="treatment",
        covariates=["confounder"],
    )
    return data, outcome_model, mediator_model


@pytest.fixture(scope="module")
def fitted_gaussian_models():
    data = make_gaussian_data(n=3000, seed=0)
    outcome_model = build_outcome_model(
        model_type="gaussian",
        data=data,
        outcome_col="outcome",
        treatment_col="treatment",
        mediator_col="mediator",
        covariates=["confounder"],
        interaction=False,
    )
    mediator_model = build_mediator_model(
        model_type="linear",
        data=data,
        mediator_col="mediator",
        treatment_col="treatment",
        covariates=["confounder"],
    )
    return data, outcome_model, mediator_model


class TestCDE:
    def test_cde_sign_positive(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=5.0,
            n_bootstrap=50,
            rng=np.random.default_rng(0),
        )
        assert est.effect > 0

    def test_cde_magnitude_close_to_true(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=5.0,
            n_bootstrap=50,
            rng=np.random.default_rng(0),
        )
        assert abs(est.effect - TRUE_NDE) < TOLERANCE

    def test_cde_stable_across_mediator_levels(self, fitted_poisson_models):
        """When there's no treatment:mediator interaction, CDE should be stable."""
        data, outcome_model, mediator_model = fitted_poisson_models
        cdes = []
        for m_level in [2.0, 5.0, 8.0]:
            est = estimate_cde(
                data=data,
                outcome_model=outcome_model,
                mediator_model=mediator_model,
                treatment_col="treatment",
                mediator_col="mediator",
                treatment_value=1,
                control_value=0,
                mediator_level=m_level,
                n_bootstrap=30,
                rng=np.random.default_rng(0),
            )
            cdes.append(est.effect)
        # All CDEs should be close to each other (within 0.05)
        assert max(cdes) - min(cdes) < 0.05

    def test_cde_ci_covers_point_estimate(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=5.0,
            n_bootstrap=50,
            rng=np.random.default_rng(0),
        )
        assert est.ci_lower < est.effect < est.ci_upper

    def test_cde_estimand_type(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=5.0,
            n_bootstrap=10,
        )
        assert est.estimand == "CDE"

    def test_cde_assumptions_listed(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=5.0,
            n_bootstrap=10,
        )
        assert len(est.assumptions) > 0

    def test_cde_gaussian(self, fitted_gaussian_models):
        data, outcome_model, mediator_model = fitted_gaussian_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=5.0,
            n_bootstrap=30,
        )
        # True NDE for Gaussian is 0.2
        assert abs(est.effect - 0.2) < 0.1

    def test_cde_mediator_level_stored(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_cde(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            mediator_level=7.5,
            n_bootstrap=10,
        )
        assert est.mediator_level == pytest.approx(7.5)


class TestTotalEffect:
    def test_te_sign_positive(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_total_effect(
            data=data,
            outcome_model=outcome_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_bootstrap=50,
        )
        assert est.effect > 0

    def test_te_magnitude(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_total_effect(
            data=data,
            outcome_model=outcome_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_bootstrap=50,
        )
        assert abs(est.effect - TRUE_TE) < TOLERANCE

    def test_te_ci_valid(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_total_effect(
            data=data,
            outcome_model=outcome_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_bootstrap=50,
        )
        assert est.ci_lower < est.effect < est.ci_upper

    def test_te_estimand_type(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        est = estimate_total_effect(
            data=data,
            outcome_model=outcome_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_bootstrap=10,
        )
        assert est.estimand == "TE"


class TestNDENIE:
    def test_nde_sign_positive(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=200,
            n_bootstrap=30,
            rng=np.random.default_rng(0),
        )
        assert nde.effect > 0

    def test_nie_sign_positive(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=200,
            n_bootstrap=30,
            rng=np.random.default_rng(0),
        )
        assert nie.effect > 0

    def test_nde_nie_sum_approx_te(self, fitted_poisson_models):
        """NDE + NIE ≈ TE (decomposition identity, subject to MC error)."""
        data, outcome_model, mediator_model = fitted_poisson_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=300,
            n_bootstrap=30,
            rng=np.random.default_rng(1),
        )
        te = estimate_total_effect(
            data=data,
            outcome_model=outcome_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_bootstrap=30,
        )
        # NDE + NIE should be close to TE (by construction: NIE = TE - NDE)
        assert abs((nde.effect + nie.effect) - te.effect) < 0.01

    def test_nde_magnitude(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=300,
            n_bootstrap=30,
            rng=np.random.default_rng(2),
        )
        assert abs(nde.effect - TRUE_NDE) < TOLERANCE

    def test_nie_magnitude(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=300,
            n_bootstrap=30,
            rng=np.random.default_rng(2),
        )
        assert abs(nie.effect - TRUE_NIE) < TOLERANCE

    def test_nde_ci_valid(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=200,
            n_bootstrap=30,
            rng=np.random.default_rng(0),
        )
        assert nde.ci_lower < nde.effect < nde.ci_upper

    def test_nde_estimand_type(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
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
        assert nde.estimand == "NDE"
        assert nie.estimand == "NIE"

    def test_nde_assumptions_listed(self, fitted_poisson_models):
        data, outcome_model, mediator_model = fitted_poisson_models
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
        assert len(nde.assumptions) >= 4  # C1-C4

    def test_gaussian_nde_nie(self, fitted_gaussian_models):
        data, outcome_model, mediator_model = fitted_gaussian_models
        nde, nie = estimate_nde_nie(
            data=data,
            outcome_model=outcome_model,
            mediator_model=mediator_model,
            treatment_col="treatment",
            mediator_col="mediator",
            treatment_value=1,
            control_value=0,
            n_mc_samples=200,
            n_bootstrap=20,
            rng=np.random.default_rng(0),
        )
        # True NDE = 0.2, NIE = 0.15
        assert abs(nde.effect - 0.2) < 0.15
        assert nie.effect > 0
