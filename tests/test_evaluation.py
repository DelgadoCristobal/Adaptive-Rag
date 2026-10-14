import pytest

from src.evaluation.evaluator import parse_judge_score
from src.evaluation.statistics import StatisticalAnalyzer
from src.utils.llm_client import extract_json, qualify_model


@pytest.mark.parametrize("reply, expected", [
    ('{"relevance_score": 0.5, "reason": "partial"}', (0.5, "partial")),
    ('```json\n{"relevance_score": 1, "reason": "ok"}\n```', (1.0, "ok")),
    ('Sure! Here is my verdict: {"relevance_score": "0.5", "reason": "x"} Hope it helps.', (0.5, "x")),
    ('{"score": 1.0, "reason": "alt key"}', (1.0, "alt key")),
    ('{"relevance_score": 0.5, "reason": "truncated mid-sen', (0.5, "truncated mid-sen")),
    ("relevance_score: 0.0", (0.0, "Evaluated by judge.")),
    ('{"relevance_score": 7}', (1.0, "Evaluated by judge.")),
])
def test_parse_judge_score_is_format_tolerant(reply, expected):
    assert parse_judge_score(reply, "relevance_score") == expected


@pytest.mark.parametrize("reply", ["", "I cannot grade this.", '{"reason": "no score here"}'])
def test_parse_judge_score_never_defaults_to_zero(reply):
    with pytest.raises(ValueError):
        parse_judge_score(reply, "relevance_score")


def test_llm_helpers():
    assert extract_json("noise {\"a\": 1} noise") == {"a": 1}
    assert extract_json("[1, 2]") is None
    assert qualify_model("gemini/x") == "gemini/x"
    assert qualify_model("x", provider="groq") == "groq/x"


def test_zero_variance_yields_p_one():
    res = StatisticalAnalyzer().analyze_metric([1.0] * 5, [1.0] * 5, "Context Precision")
    assert res["p_value"] == 1.0 and not res["is_significant"] and res["variance_note"]


def test_alternative_direction():
    az = StatisticalAnalyzer()
    naive = [0.1, 0.2, 0.3, 0.2, 0.4, 0.1, 0.3]
    better = [x + 0.2 + 0.01 * i for i, x in enumerate(naive)]
    assert az.analyze_metric(naive, better, "Faithfulness")["alternative"] == "greater"
    assert az.analyze_metric(naive, better, "Faithfulness")["is_significant"]
    assert az.analyze_metric(naive, better, "Latency (ms)")["alternative"] == "less"
    assert not az.analyze_metric(naive, better, "Latency (ms)")["is_significant"]


def test_retry_delay_parses_compound_durations():
    from src.evaluation.evaluator import _retry_delay
    assert _retry_delay("please retry in 11h19m35.5s.") == 11 * 3600 + 19 * 60 + 35.5
    assert _retry_delay("retry in 40.8s") == 40.8
    assert _retry_delay("503 unavailable") is None


def test_cli_judge_without_binary_is_unavailable(monkeypatch):
    import pytest
    from src.utils.llm_client import CLIUnavailableError, GeminiCLIClient

    monkeypatch.setenv("JUDGE_CLI", "definitely-not-a-binary")
    client = GeminiCLIClient()
    assert not client.available
    with pytest.raises(CLIUnavailableError):
        client.complete("hi")


def test_cli_judge_roundtrips_utf8_via_stdin(tmp_path, monkeypatch):
    import sys
    from src.utils.llm_client import GeminiCLIClient

    script = tmp_path / "fake_cli.py"
    script.write_text("import sys\nsys.stdout.reconfigure(encoding='utf-8')\nprint(sys.stdin.buffer.read().decode('utf-8'))\n", encoding="utf-8")
    wrapper = tmp_path / "fake_cli.cmd"
    wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n')
    out = GeminiCLIClient(binary=str(wrapper)).complete('Ünïcode "quotes" & <tags> 100%')
    assert out == 'Ünïcode "quotes" & <tags> 100%'
