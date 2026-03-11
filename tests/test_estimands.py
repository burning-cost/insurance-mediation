"""
Tests for EffectEstimate and MediationResult data classes.
"""
import math
import pytest

from insurance_mediation.estimands import EffectEstimate, MediationResult, SensitivityResult


class TestEffectEstimate:
    def test_basic_creation(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=0.2,
            ci_lower=0.05,
            ci_upper=0.35,
        )
        assert est.estimand == "CDE"
        assert est.effect == pytest.approx(0.2)

    def test_ratio_property_log_scale(self):
        est = EffectEstimate(
            estimand="NDE",
            effect=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
            scale="difference",
        )
        assert est.ratio == pytest.approx(math.exp(0.2))

    def test_ratio_property_ratio_scale(self):
        est = EffectEstimate(
            estimand="TE",
            effect=1.5,
            ci_lower=1.1,
            ci_upper=2.0,
            scale="ratio",
        )
        assert est.ratio == pytest.approx(1.5)

    def test_significant_difference_scale_positive(self):
        est = EffectEstimate(
            estimand="NIE",
            effect=0.15,
            ci_lower=0.05,
            ci_upper=0.25,
        )
        assert est.significant is True

    def test_significant_difference_scale_crosses_zero(self):
        est = EffectEstimate(
            estimand="NIE",
            effect=0.05,
            ci_lower=-0.02,
            ci_upper=0.12,
        )
        assert est.significant is False

    def test_significant_ratio_scale_above_one(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=1.3,
            ci_lower=1.1,
            ci_upper=1.5,
            scale="ratio",
        )
        assert est.significant is True

    def test_significant_ratio_scale_includes_one(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=1.05,
            ci_lower=0.95,
            ci_upper=1.15,
            scale="ratio",
        )
        assert est.significant is False

    def test_ratio_ci_log_scale(self):
        est = EffectEstimate(
            estimand="NDE",
            effect=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
        )
        lo, hi = est.ratio_ci
        assert lo == pytest.approx(math.exp(0.1))
        assert hi == pytest.approx(math.exp(0.3))

    def test_repr_contains_estimand(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=0.2,
            ci_lower=0.05,
            ci_upper=0.35,
        )
        assert "CDE" in repr(est)

    def test_repr_marks_significant(self):
        est = EffectEstimate(
            estimand="NDE",
            effect=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
        )
        assert "*" in repr(est)

    def test_repr_no_star_not_significant(self):
        est = EffectEstimate(
            estimand="NDE",
            effect=0.05,
            ci_lower=-0.02,
            ci_upper=0.12,
        )
        assert "*" not in repr(est)

    def test_assumptions_default_empty(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=0.1,
            ci_lower=0.0,
            ci_upper=0.2,
        )
        assert est.assumptions == []

    def test_assumptions_stored(self):
        assumptions = ["C1: no unmeasured confounding", "C2: no interaction"]
        est = EffectEstimate(
            estimand="CDE",
            effect=0.1,
            ci_lower=0.0,
            ci_upper=0.2,
            assumptions=assumptions,
        )
        assert len(est.assumptions) == 2

    def test_mediator_level_stored(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=0.1,
            ci_lower=0.0,
            ci_upper=0.2,
            mediator_level=5.0,
        )
        assert est.mediator_level == 5.0

    def test_treatment_values_stored(self):
        est = EffectEstimate(
            estimand="CDE",
            effect=0.1,
            ci_lower=0.0,
            ci_upper=0.2,
            treatment_values=("E1", "SW1"),
        )
        assert est.treatment_values == ("E1", "SW1")


class TestMediationResult:
    def _make_result(self):
        te = EffectEstimate("TE", 0.35, 0.20, 0.50)
        nde = EffectEstimate("NDE", 0.20, 0.05, 0.35)
        nie = EffectEstimate("NIE", 0.15, 0.05, 0.25)
        cde = EffectEstimate("CDE", 0.18, 0.04, 0.32, mediator_level=5.0)

        return MediationResult(
            treatment="treatment",
            mediator="mediator",
            outcome="outcome",
            treatment_values=(1, 0),
            outcome_model_type="poisson",
            mediator_model_type="linear",
            n_obs=2000,
            _cde_estimates={5.0: cde},
            _nde=nde,
            _nie=nie,
            _total_effect=te,
        )

    def test_total_effect(self):
        r = self._make_result()
        te = r.total_effect()
        assert te.effect == pytest.approx(0.35)

    def test_nde(self):
        r = self._make_result()
        nde = r.nde()
        assert nde.estimand == "NDE"

    def test_nie(self):
        r = self._make_result()
        nie = r.nie()
        assert nie.estimand == "NIE"

    def test_cde_with_level(self):
        r = self._make_result()
        cde = r.cde(mediator_level=5.0)
        assert cde.mediator_level == 5.0

    def test_cde_raises_without_estimates(self):
        r = MediationResult(
            treatment="t", mediator="m", outcome="y",
            treatment_values=(1, 0),
            outcome_model_type="poisson",
            mediator_model_type="linear",
            n_obs=100,
        )
        with pytest.raises(RuntimeError, match="No CDE estimates"):
            r.cde()

    def test_nde_raises_without_fit(self):
        r = MediationResult(
            treatment="t", mediator="m", outcome="y",
            treatment_values=(1, 0),
            outcome_model_type="poisson",
            mediator_model_type="linear",
            n_obs=100,
        )
        with pytest.raises(RuntimeError, match="NDE not available"):
            r.nde()

    def test_summary_contains_treatment(self):
        r = self._make_result()
        s = r.summary()
        assert "treatment" in s

    def test_summary_contains_effect_values(self):
        r = self._make_result()
        s = r.summary()
        assert "0.35" in s or "0.3500" in s

    def test_cde_nearest_level(self):
        r = self._make_result()
        # Should find the nearest pre-computed level (5.0) even with 4.9
        cde = r.cde(mediator_level=4.9)
        assert cde.mediator_level == 5.0


class TestSensitivityResult:
    def _make_sensitivity(self):
        nie_est = EffectEstimate("NIE", 0.15, 0.05, 0.25)
        return SensitivityResult(
            rho_values=[-0.5, 0.0, 0.5],
            nie_estimates=[0.05, 0.15, 0.25],
            nie_ci_lower=[-0.02, 0.05, 0.12],
            nie_ci_upper=[0.12, 0.25, 0.38],
            rho_at_zero=None,
            e_value=2.5,
            e_value_ci=1.8,
            nie_at_rho0=nie_est,
        )

    def test_creation(self):
        s = self._make_sensitivity()
        assert s.e_value == pytest.approx(2.5)

    def test_repr_contains_evalue(self):
        s = self._make_sensitivity()
        assert "E-value" in repr(s)

    def test_rho_at_zero_none(self):
        s = self._make_sensitivity()
        assert s.rho_at_zero is None
