import numpy as np
import pytest

from src.evaluation.statistics import (
    METRIC_DIRECTION,
    StatisticalAnalyzer,
    holm_adjust,
    rank_biserial_kerby,
)


def _rows(n=12, seed=1, factual=False):
    """Synthetic paired rows; hybrid wins strongly on rerank score (diagnostic) only."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = float(rng.uniform(0.2, 0.8))
        rows.append({
            "type": ("simple" if i % 2 else "multi-hop") if not factual else ("factual" if i % 2 else "comparison"),
            "domain": "finance",
            "naive": {"faithfulness": base, "answer_relevance": base, "context_precision": base,
                      "latency_ms": 1000.0 + i, "top_rerank_score": 0.1 * (i % 5)},
            "hybrid": {"faithfulness": base + float(rng.normal(0, 0.05)),
                       "answer_relevance": base + float(rng.normal(0, 0.05)),
                       "context_precision": base + float(rng.normal(0, 0.05)),
                       "latency_ms": 1000.0 + i + float(rng.normal(0, 5)),
                       "top_rerank_score": 0.1 * (i % 5) + 0.5 + 0.01 * i},
            "delta": {"faithfulness": 0.0, "answer_relevance": 0.01 * (i % 3), "context_precision": 0.0},
        })
    return rows


def test_kerby_rank_biserial_hand_computed():
    # |Δ| ranks 1,2,3,4 -> W+ = 1+2+4 = 7, W- = 3 -> r = (7-3)/10 = 0.4
    assert rank_biserial_kerby(np.array([1.0, 2.0, -3.0, 4.0])) == pytest.approx(0.4)


def test_kerby_uses_average_ranks_and_ignores_zeros():
    # Non-zero |Δ| = 1,1,2 -> ranks 1.5,1.5,3; W+ = 1.5+3 = 4.5, W- = 1.5 -> r = 3/6 = 0.5
    assert rank_biserial_kerby(np.array([1.0, -1.0, 2.0, 0.0])) == pytest.approx(0.5)


def test_kerby_zero_when_no_nonzero_deltas():
    assert rank_biserial_kerby(np.zeros(5)) == 0.0
    res = StatisticalAnalyzer().analyze_metric([0.5] * 5, [0.5] * 5, "Faithfulness")
    assert res["effect_size_name"] == "Rank-Biserial r" and res["effect_size_val"] == 0.0
    assert res["p_value"] == 1.0 and res["variance_note"]


def test_constant_nonzero_delta_is_tested_not_forced_to_p1():
    # Hybrid wins every one of 10 pairs by the same margin: the strongest possible evidence.
    res = StatisticalAnalyzer().analyze_metric([0.5] * 10, [1.0] * 10, "Faithfulness")
    assert res["test_name"].startswith("Wilcoxon Signed-Rank Test")
    assert res["p_value"] < 0.05
    assert not res["variance_note"]
    assert res["improved"] and res["effect_size_val"] == pytest.approx(1.0)


def test_analyze_metric_reports_kerby_effect_size():
    res = StatisticalAnalyzer().analyze_metric([0, 0, 0, 0], [1, 2, -3, 4], "Faithfulness")
    assert res["effect_size_name"] == "Rank-Biserial r"
    assert res["effect_size_val"] == pytest.approx(0.4)
    assert res["test_name"].startswith("Wilcoxon Signed-Rank Test")
    assert "shapiro_p" not in res and "Student" not in res["test_name"]


def test_holm_adjustment_on_known_pvalues():
    # Sorted 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06; returned in input order.
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []


def test_direction_map_latency_lower_is_better():
    assert METRIC_DIRECTION["latency_ms"] == "lower"
    for key in ("faithfulness", "answer_relevance", "context_precision", "top_rerank_score"):
        assert METRIC_DIRECTION[key] == "higher"
    naive = [1000.0 + 10 * i for i in range(10)]
    faster = [x - 200.0 - i for i, x in enumerate(naive)]
    res = StatisticalAnalyzer().analyze_metric(naive, faster, "Latency (ms)")
    assert res["direction"] == "lower"
    assert res["is_significant"] and res["improved"] and res["better_arm"] == "hybrid"


def test_unknown_metric_defaults_to_higher_with_warning(caplog):
    with caplog.at_level("WARNING"):
        res = StatisticalAnalyzer().analyze_metric([0.1] * 4, [0.2, 0.3, 0.4, 0.5], "Mystery Metric")
    assert res["direction"] == "higher"
    assert "Mystery Metric" in caplog.text


def test_ci_is_two_sided_and_contains_mean_delta():
    rng = np.random.default_rng(3)
    naive = rng.uniform(0, 1, 20)
    hybrid = naive + rng.normal(0.1, 0.2, 20)
    res = StatisticalAnalyzer().analyze_metric(list(naive), list(hybrid), "Faithfulness")
    lo, hi = res["ci_95"]
    assert lo < res["mean_delta"] < hi
    assert res["alternative"] == "two-sided"


def test_holm_applied_across_outcome_family_excluding_top1():
    report = StatisticalAnalyzer().generate_full_report(_rows())
    ov = report["overall"]
    family = ["faithfulness", "answer_relevance", "context_precision", "latency_ms"]
    expected = holm_adjust([ov[k]["p_value"] for k in family])
    for key, adj in zip(family, expected):
        assert ov[key]["p_value_adjusted"] == pytest.approx(adj, abs=1e-5)
        assert ov[key]["is_significant"] == (ov[key]["p_value_adjusted"] < 0.05)
        assert not ov[key]["diagnostic"]
    top1 = ov["rerank_score"]
    assert top1["diagnostic"] is True and "diagnostic_note" in top1
    assert top1["p_value_adjusted"] == top1["p_value"]


def test_conclusions_ignore_top1_and_hardcoded_factual_narrative():
    az = StatisticalAnalyzer()
    report = az.generate_full_report(_rows())
    text = "\n".join(az._conclusions(report))
    assert "Cross-Encoder" not in text
    assert "factual" not in text.lower()
    # Even with a real "factual" category, no claim that it favours the vector baseline.
    text_f = "\n".join(az._conclusions(az.generate_full_report(_rows(factual=True))))
    assert "favorecen al baseline" not in text_f


def test_markdown_report_marks_top1_as_diagnostic():
    az = StatisticalAnalyzer()
    md = az.format_markdown_report(az.generate_full_report(_rows()))
    assert "diagn" in md.lower()
    assert "Holm" in md
