"""
Tests for HTML/JSON report generation.
"""
import json
import os
import tempfile
import pytest

from insurance_mediation.estimands import EffectEstimate, MediationResult


def _make_full_result() -> MediationResult:
    """Create a MediationResult with all estimands populated."""
    te = EffectEstimate("TE", 0.35, 0.20, 0.50, n_bootstrap=100)
    nde = EffectEstimate("NDE", 0.20, 0.05, 0.35, n_bootstrap=100)
    nie = EffectEstimate("NIE", 0.15, 0.05, 0.25, n_bootstrap=100)
    cde = EffectEstimate("CDE", 0.18, 0.04, 0.32, mediator_level=5.0, n_bootstrap=100)

    return MediationResult(
        treatment="postcode_group",
        mediator="imd_decile",
        outcome="claim_count",
        treatment_values=("E1", "SW1"),
        outcome_model_type="poisson",
        mediator_model_type="linear",
        n_obs=50000,
        _cde_estimates={5.0: cde},
        _nde=nde,
        _nie=nie,
        _total_effect=te,
    )


class TestHTMLReport:
    def test_generates_html_file(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            output = result.report(output=path, title="Test Report")
            assert os.path.exists(output)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_title(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path, title="My Mediation Report")
            with open(path, "r") as f:
                html = f.read()
            assert "My Mediation Report" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_treatment_name(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            assert "postcode_group" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_mediator_name(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            assert "imd_decile" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_effect_values(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            # Should contain the TE effect (0.35)
            assert "0.35" in html or "0.3500" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_proportionality_section(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            assert "proportionality" in html.lower() or "Proportionality" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_protected_attribute(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path, protected_attribute="ethnicity")
            with open(path, "r") as f:
                html = f.read()
            assert "ethnicity" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_assumptions(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            # Should mention C1, C2 assumptions
            assert "C1" in html or "unmeasured" in html.lower()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_contains_dag(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            assert "<svg" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_html_is_valid_html(self):
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            result.report(output=path)
            with open(path, "r") as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "</html>" in html
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestJSONReport:
    def test_generates_json_file(self):
        from insurance_mediation.report import generate_json_report
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            output = generate_json_report(result, output=path)
            assert os.path.exists(output)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_json_is_valid(self):
        from insurance_mediation.report import generate_json_report
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            generate_json_report(result, output=path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "estimates" in data
            assert "metadata" in data
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_json_contains_nde_nie(self):
        from insurance_mediation.report import generate_json_report
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            generate_json_report(result, output=path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "nde" in data["estimates"]
            assert "nie" in data["estimates"]
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_json_effect_values_correct(self):
        from insurance_mediation.report import generate_json_report
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            generate_json_report(result, output=path)
            with open(path, "r") as f:
                data = json.load(f)
            assert data["estimates"]["total_effect"]["effect"] == pytest.approx(0.35)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_json_metadata_correct(self):
        from insurance_mediation.report import generate_json_report
        result = _make_full_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            generate_json_report(result, output=path)
            with open(path, "r") as f:
                data = json.load(f)
            assert data["metadata"]["treatment"] == "postcode_group"
            assert data["metadata"]["n_obs"] == 50000
        finally:
            if os.path.exists(path):
                os.unlink(path)
