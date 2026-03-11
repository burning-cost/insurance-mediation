"""
FCA-ready HTML mediation analysis report generator.

The report is designed to be attached to a pricing actuarial function
report or submitted as supporting evidence for FCA data ethics review.
It documents:
  1. The causal DAG assumed by the analysis
  2. Total effect decomposition (CDE, NDE, NIE, TE)
  3. Sensitivity analysis — how robust is the mediated effect?
  4. Identification assumptions required for each estimand
  5. A template Section 19 proportionality statement

Design choice: Pure Python / Jinja2, no external binary dependencies.
No wkhtmltopdf, no headless Chrome. The HTML output is self-contained
and renders correctly in all browsers.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_mediation.estimands import MediationResult


# Inline CSS keeps the report self-contained
_CSS = """
body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto;
       color: #333; line-height: 1.5; }
h1 { color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }
h2 { color: #16213e; margin-top: 32px; }
h3 { color: #0f3460; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th { background: #1a1a2e; color: white; padding: 8px 12px; text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f8f8f8; }
.significant { font-weight: bold; color: #0f3460; }
.not-significant { color: #888; }
.assumption { background: #fff3cd; border-left: 4px solid #ffc107;
              padding: 8px 12px; margin: 8px 0; font-size: 0.9em; }
.warning { background: #f8d7da; border-left: 4px solid #dc3545;
           padding: 8px 12px; margin: 8px 0; }
.info { background: #d4edda; border-left: 4px solid #28a745;
        padding: 8px 12px; margin: 8px 0; }
.proportionality { background: #e8f4f8; border: 1px solid #bee5eb;
                   padding: 16px; border-radius: 4px; }
.evalue-box { background: #f0f0ff; border: 1px solid #9999dd;
              padding: 12px; border-radius: 4px; display: inline-block; }
svg { font-size: 12px; }
.footer { margin-top: 40px; font-size: 0.8em; color: #888;
          border-top: 1px solid #ddd; padding-top: 12px; }
"""


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>{{ css }}</style>
</head>
<body>
<h1>{{ title }}</h1>
<p><strong>Generated:</strong> {{ generated }}</p>
<p><strong>Treatment:</strong> {{ treatment }} &nbsp;&nbsp;
   <strong>Mediator:</strong> {{ mediator }} &nbsp;&nbsp;
   <strong>Outcome:</strong> {{ outcome }}</p>
<p><strong>Comparison:</strong> {{ treat_val }} vs {{ ctrl_val }} &nbsp;&nbsp;
   <strong>N observations:</strong> {{ n_obs }}</p>
{% if protected_attribute %}
<p><strong>Protected characteristic proxied:</strong> {{ protected_attribute }}</p>
{% endif %}

<h2>1. Causal DAG</h2>
<p>The analysis assumes the following directed acyclic graph (DAG):</p>
{{ dag_svg }}
<p>The mediator ({{ mediator }}) is a legitimate risk-rating factor.
The question is: what proportion of the treatment ({{ treatment }}) effect on
{{ outcome }} operates <em>through</em> the mediator pathway versus directly?</p>

<h2>2. Effect Decomposition</h2>
<p>Effects are on the <strong>{{ scale_label }}</strong> scale.
{% if log_link %}
Ratios &gt; 1 indicate higher outcomes for the treatment group.
{% else %}
Positive values indicate higher outcomes for the treatment group.
{% endif %}
</p>

<table>
<tr>
  <th>Estimand</th>
  <th>Effect ({{ scale_label }})</th>
  <th>95% CI</th>
  <th>Ratio</th>
  <th>Ratio 95% CI</th>
  <th>Significant?</th>
</tr>
{% for row in effect_rows %}
<tr>
  <td><strong>{{ row.name }}</strong></td>
  <td class="{{ 'significant' if row.sig else 'not-significant' }}">{{ row.effect }}</td>
  <td>{{ row.ci }}</td>
  <td>{{ row.ratio }}</td>
  <td>{{ row.ratio_ci }}</td>
  <td>{{ 'Yes *' if row.sig else 'No' }}</td>
</tr>
{% endfor %}
</table>

<p><em>CDE: Controlled Direct Effect (mediator fixed at population mean).
NDE: Natural Direct Effect. NIE: Natural Indirect Effect.
TE: Total Effect. Proportionality: NIE / TE (share mediated through {{ mediator }}).</em></p>

<h2>3. FCA Fairness Interpretation</h2>
{% if nie_proportion is not none %}
<div class="{{ 'info' if nie_proportion > 0.5 else 'warning' }}">
<strong>{{ "%.1f"|format(nie_proportion * 100) }}% of the {{ treatment }} price differential
is explained by the {{ mediator }} pathway.</strong>
{% if nie_proportion > 0.8 %}
The large majority of the price differential appears to be mediated through
a legitimate risk factor. The direct (unexplained) effect is small.
{% elif nie_proportion > 0.5 %}
More than half of the price differential is mediated through a legitimate
risk factor, but a substantial direct effect remains.
{% else %}
Less than half of the price differential is mediated through the specified
risk factor. The direct effect is the dominant component, which may warrant
further investigation.
{% endif %}
</div>
{% endif %}

{% if cde_rows %}
<h3>Controlled Direct Effects (by mediator level)</h3>
<table>
<tr><th>Mediator Level</th><th>CDE (log-ratio)</th><th>95% CI</th><th>Ratio</th><th>Significant?</th></tr>
{% for row in cde_rows %}
<tr>
  <td>{{ row.m_level }}</td>
  <td>{{ row.effect }}</td>
  <td>{{ row.ci }}</td>
  <td>{{ row.ratio }}</td>
  <td>{{ 'Yes *' if row.sig else 'No' }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

<h2>4. Sensitivity Analysis</h2>
{% if sensitivity %}
<div class="evalue-box">
<strong>E-value: {{ sensitivity.e_value }}</strong><br>
E-value (CI bound): {{ sensitivity.e_value_ci }}<br>
<small>Minimum unmeasured confounder risk ratio needed to explain away the NIE.</small>
</div>
<p></p>
{% if sensitivity.rho_at_zero is not none %}
<p>The NIE crosses zero at <strong>rho = {{ sensitivity.rho_at_zero }}</strong>.
Values of |rho| above this represent confounding strong enough to reverse the conclusion.</p>
{% else %}
<p>The NIE does not cross zero within the evaluated rho range
({{ sensitivity.rho_values[0] }} to {{ sensitivity.rho_values[-1] }}).
The mediated effect is robust to the levels of confounding considered.</p>
{% endif %}
<table>
<tr><th>rho</th><th>NIE estimate</th><th>95% CI</th></tr>
{% for i in range(sensitivity.rho_values|length) %}
<tr>
  <td>{{ "%.3f"|format(sensitivity.rho_values[i]) }}</td>
  <td>{{ "%.4f"|format(sensitivity.nie_estimates[i]) }}</td>
  <td>[{{ "%.4f"|format(sensitivity.nie_ci_lower[i]) }}, {{ "%.4f"|format(sensitivity.nie_ci_upper[i]) }}]</td>
</tr>
{% endfor %}
</table>
{% else %}
<p>Sensitivity analysis not computed. Call <code>results.sensitivity()</code> first.</p>
{% endif %}

<h2>5. Identification Assumptions</h2>
<p>The following assumptions are required for each estimand to have a valid
causal interpretation. These cannot be verified from data alone.</p>

<h3>For CDE (most defensible)</h3>
<div class="assumption">
<ul>
<li>C1: No unmeasured confounders of the treatment–outcome relationship given the covariates.</li>
<li>C2: No unmeasured confounders of the mediator–outcome relationship given (treatment, covariates).</li>
<li>C3: No unmeasured confounders of the treatment–mediator relationship given the covariates.</li>
<li><em>Assumption C4 is NOT required for CDE — this is its primary advantage.</em></li>
</ul>
</div>

<h3>For NDE/NIE (additional requirement)</h3>
<div class="assumption">
<ul>
<li>C4: No treatment-induced mediator–outcome confounders. No variable on the
treatment → confounders → mediator path exists that also affects the outcome.
In practice: no other postcode-level variable (beyond {{ mediator }}) that
both varies with {{ treatment }} and confounds the {{ mediator }}–{{ outcome }}
relationship.</li>
</ul>
</div>

<h2>6. Proportionality Statement (Template)</h2>
<div class="proportionality">
<h3>Template — adapt for your specific product/portfolio</h3>
<p>This analysis was conducted in relation to [<em>product name</em>] sold to
[<em>customer segment</em>] in the UK. The rating factor under review is
<strong>{{ treatment }}</strong>, which may act as a proxy for
<strong>{{ protected_attribute or "[protected characteristic]" }}</strong>.</p>

<p>A causal mediation analysis was performed to decompose the {{ treatment }}
price differential into:</p>
<ul>
<li>A <em>direct effect</em> that cannot be explained by legitimate risk factors
    (specifically <strong>{{ mediator }}</strong>)</li>
<li>An <em>indirect effect</em> that operates through {{ mediator }}, a rating
    factor that reflects genuine differences in expected claim costs</li>
</ul>

{% if nie_proportion is not none %}
<p>The analysis found that approximately <strong>{{ "%.0f"|format(nie_proportion * 100) }}%</strong>
of the {{ treatment }} price differential is attributable to differences in
{{ mediator }} between {{ treatment }} groups. The remaining
<strong>{{ "%.0f"|format((1 - nie_proportion) * 100) }}%</strong> represents
a direct {{ treatment }} effect.</p>
{% endif %}

<p>[<em>Pricing actuarial function to complete: explanation of direct effect,
whether it reflects additional legitimate risk factors not included as mediators,
and assessment of proportionality under ICOBS 4 / Consumer Duty.</em>]</p>
</div>

<div class="footer">
<p>Generated by insurance-mediation v{{ version }} |
   Analysis date: {{ generated }} |
   Burning Cost</p>
</div>
</body>
</html>
"""


def _make_dag_svg(treatment: str, mediator: str, outcome: str, covariates: list[str]) -> str:
    """Generate a simple SVG causal DAG."""
    # Positions: Treatment left, Mediator centre, Outcome right
    # Covariates below
    svg_parts = [
        '<svg width="600" height="180" xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="7" '
        'refX="10" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#1a1a2e"/></marker></defs>',
    ]

    # Nodes
    nodes = {
        "T": (90, 80, treatment[:20]),
        "M": (300, 80, mediator[:20]),
        "Y": (510, 80, outcome[:20]),
        "C": (300, 150, "Covariates"),
    }

    # Arrows
    arrows = [
        # T -> M
        (165, 80, 220, 80),
        # M -> Y
        (380, 80, 435, 80),
        # T -> Y (direct path, curved)
        (165, 68, 435, 68),
        # C -> M
        (300, 133, 300, 103),
        # C -> Y
        (370, 142, 490, 95),
        # C -> T
        (230, 148, 115, 95),
    ]

    for x1, y1, x2, y2 in arrows:
        svg_parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#1a1a2e" stroke-width="1.5" '
            f'marker-end="url(#arrow)"/>'
        )

    # T -> Y curved arrow (direct path)
    svg_parts.append(
        '<path d="M 165 65 C 300 30 300 30 435 65" '
        'fill="none" stroke="#666" stroke-width="1.5" stroke-dasharray="4,3" '
        'marker-end="url(#arrow)"/>'
    )

    # Draw node boxes
    for key, (cx, cy, label) in nodes.items():
        color = "#1a1a2e" if key in ("T", "Y") else "#16213e"
        svg_parts.append(
            f'<rect x="{cx-70}" y="{cy-18}" width="140" height="36" '
            f'rx="4" fill="{color}" opacity="0.85"/>'
        )
        svg_parts.append(
            f'<text x="{cx}" y="{cy+5}" text-anchor="middle" '
            f'fill="white" font-size="11">{label}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _fmt_effect(val: float, precision: int = 4) -> str:
    if math.isnan(val):
        return "N/A"
    return f"{val:+.{precision}f}"


def _fmt_ratio(val: float, precision: int = 3) -> str:
    if math.isnan(val):
        return "N/A"
    try:
        return f"{math.exp(val):.{precision}f}"
    except Exception:
        return "N/A"


def generate_report(
    result: "MediationResult",
    output: str = "mediation_report.html",
    title: str = "Mediation Analysis Report",
    protected_attribute: str | None = None,
) -> str:
    """Generate an HTML FCA-ready mediation report.

    Parameters
    ----------
    result : MediationResult
    output : str
    title : str
    protected_attribute : str or None

    Returns
    -------
    str
        Path to the generated report.
    """
    from jinja2 import Environment
    from insurance_mediation import __version__

    env = Environment(autoescape=True)
    template = env.from_string(_REPORT_TEMPLATE)

    log_link = result.outcome_model_type in ("poisson", "gamma", "tweedie")
    scale_label = "log-ratio" if log_link else "mean difference"

    # Build effect rows
    effect_rows = []
    try:
        te = result.total_effect()
        effect_rows.append({
            "name": "Total Effect (TE)",
            "effect": _fmt_effect(te.effect),
            "ci": f"[{_fmt_effect(te.ci_lower)}, {_fmt_effect(te.ci_upper)}]",
            "ratio": _fmt_ratio(te.effect) if log_link else "N/A",
            "ratio_ci": (
                f"[{_fmt_ratio(te.ci_lower)}, {_fmt_ratio(te.ci_upper)}]"
                if log_link else "N/A"
            ),
            "sig": te.significant,
        })
    except RuntimeError:
        pass

    try:
        nde = result.nde()
        effect_rows.append({
            "name": "Natural Direct Effect (NDE)",
            "effect": _fmt_effect(nde.effect),
            "ci": f"[{_fmt_effect(nde.ci_lower)}, {_fmt_effect(nde.ci_upper)}]",
            "ratio": _fmt_ratio(nde.effect) if log_link else "N/A",
            "ratio_ci": (
                f"[{_fmt_ratio(nde.ci_lower)}, {_fmt_ratio(nde.ci_upper)}]"
                if log_link else "N/A"
            ),
            "sig": nde.significant,
        })
    except RuntimeError:
        pass

    try:
        nie = result.nie()
        effect_rows.append({
            "name": "Natural Indirect Effect (NIE)",
            "effect": _fmt_effect(nie.effect),
            "ci": f"[{_fmt_effect(nie.ci_lower)}, {_fmt_effect(nie.ci_upper)}]",
            "ratio": _fmt_ratio(nie.effect) if log_link else "N/A",
            "ratio_ci": (
                f"[{_fmt_ratio(nie.ci_lower)}, {_fmt_ratio(nie.ci_upper)}]"
                if log_link else "N/A"
            ),
            "sig": nie.significant,
        })
    except RuntimeError:
        pass

    # CDE rows
    cde_rows = []
    for m_level, est in sorted(result._cde_estimates.items()):
        cde_rows.append({
            "m_level": f"{m_level:.2f}",
            "effect": _fmt_effect(est.effect),
            "ci": f"[{_fmt_effect(est.ci_lower)}, {_fmt_effect(est.ci_upper)}]",
            "ratio": _fmt_ratio(est.effect) if log_link else "N/A",
            "sig": est.significant,
        })

    # NIE proportion of TE
    nie_proportion = None
    try:
        te = result.total_effect()
        nie = result.nie()
        if abs(te.effect) > 1e-10:
            nie_proportion = abs(nie.effect) / abs(te.effect)
    except RuntimeError:
        pass

    # Sensitivity
    sens = result._sensitivity

    dag_svg = _make_dag_svg(
        result.treatment, result.mediator, result.outcome,
        covariates=[]
    )

    html = template.render(
        title=title,
        css=_CSS,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        treatment=result.treatment,
        mediator=result.mediator,
        outcome=result.outcome,
        treat_val=result.treatment_values[0],
        ctrl_val=result.treatment_values[1],
        n_obs=f"{result.n_obs:,}",
        protected_attribute=protected_attribute,
        dag_svg=dag_svg,
        scale_label=scale_label,
        log_link=log_link,
        effect_rows=effect_rows,
        cde_rows=cde_rows,
        nie_proportion=nie_proportion,
        sensitivity=sens,
        version=__version__,
    )

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    return output


def generate_json_report(
    result: "MediationResult",
    output: str = "mediation_report.json",
) -> str:
    """Generate a JSON representation of the mediation results.

    Suitable for machine consumption or integration into actuarial
    reporting pipelines.

    Parameters
    ----------
    result : MediationResult
    output : str

    Returns
    -------
    str
        Path to the generated JSON file.
    """
    from insurance_mediation import __version__

    data: dict = {
        "generated": datetime.now().isoformat(),
        "version": __version__,
        "metadata": {
            "treatment": result.treatment,
            "mediator": result.mediator,
            "outcome": result.outcome,
            "treatment_values": list(result.treatment_values),
            "outcome_model_type": result.outcome_model_type,
            "n_obs": result.n_obs,
        },
        "estimates": {},
    }

    def _est_to_dict(est):
        return {
            "estimand": est.estimand,
            "effect": est.effect,
            "ci_lower": est.ci_lower,
            "ci_upper": est.ci_upper,
            "scale": est.scale,
            "significant": est.significant,
            "n_bootstrap": est.n_bootstrap,
            "assumptions": est.assumptions,
        }

    try:
        data["estimates"]["total_effect"] = _est_to_dict(result.total_effect())
    except RuntimeError:
        pass
    try:
        data["estimates"]["nde"] = _est_to_dict(result.nde())
    except RuntimeError:
        pass
    try:
        data["estimates"]["nie"] = _est_to_dict(result.nie())
    except RuntimeError:
        pass

    cde_data = {}
    for m_level, est in result._cde_estimates.items():
        cde_data[str(m_level)] = _est_to_dict(est)
    if cde_data:
        data["estimates"]["cde"] = cde_data

    if result._sensitivity:
        s = result._sensitivity
        data["sensitivity"] = {
            "rho_values": s.rho_values,
            "nie_estimates": s.nie_estimates,
            "rho_at_zero": s.rho_at_zero,
            "e_value": s.e_value,
            "e_value_ci": s.e_value_ci,
        }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output
