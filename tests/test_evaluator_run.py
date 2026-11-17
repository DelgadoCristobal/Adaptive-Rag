"""Offline tests for the paired-benchmark run loop: checkpoint keying, resume filtering and arm order.

No real retrieval engine, synthesizer, judge or model is constructed: a fake evaluator replaces them.
"""
import json
from pathlib import Path

import pytest

from src.evaluation import evaluator as ev
from src.evaluation.evaluator import RAGEvaluator, load_checkpoint, run_fingerprint
from src.retrieval.router import RetrievalStrategy

CONFIG = {"generator_model": "ollama/gemma:2b", "judge_model": "gemini/x", "judge_backend": "api"}


class FakeEngine:
    def __init__(self, calls):
        self.calls = calls

    def retrieve(self, query, strategy):
        self.calls.append(("retrieve", query, strategy))
        return [{"text": "ctx", "source_doc_id": "doc1", "rerank_score": 0.5}], None


class FakeSynth:
    def synthesize(self, query, retrieved):
        return {"answer": f"answer to {query}"}

    def context_block(self, items):
        return "\n\n".join(i.get("text", "") for i in items)


def make_evaluator():
    """A RAGEvaluator with no real engine, synthesizer or judge."""
    e = RAGEvaluator.__new__(RAGEvaluator)
    e.model_name = "ollama/gemma:2b"
    e.judge_model_name = "ollama/judge"
    e.judge_fallbacks = []
    e.cli_judge = None
    e.calls = []
    e.hybrid_engine = FakeEngine(e.calls)
    e.synthesizer = FakeSynth()
    e.evaluate_faithfulness = lambda context, answer: (1.0, "ok")
    e.evaluate_relevance = lambda question, answer: (1.0, "ok")
    return e


def write_dataset(tmp_path, n=3):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([{"id": f"q{i}", "question": f"question {i}"} for i in range(n)]), encoding="utf-8")
    return path


def row(q_id):
    arm = {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0, "latency_ms": 1.0, "top_rerank_score": 0.5}
    return {"id": q_id, "type": "general", "naive": arm, "hybrid": arm, "delta": dict.fromkeys(arm, 0.0)}


# --- T2: fingerprint -------------------------------------------------------------------------------

def test_fingerprint_is_stable_and_order_independent():
    a = run_fingerprint(["q1", "q0"], CONFIG)
    assert a == run_fingerprint(["q0", "q1"], dict(reversed(list(CONFIG.items()))))
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)


def test_fingerprint_changes_with_dataset_ids_or_config():
    base = run_fingerprint(["q0", "q1"], CONFIG)
    assert run_fingerprint(["q0", "q2"], CONFIG) != base
    assert run_fingerprint(["q0"], CONFIG) != base
    assert run_fingerprint(["q0", "q1"], {**CONFIG, "judge_model": "gemini/y"}) != base


# --- T2: loading a checkpoint --------------------------------------------------------------------

def write_ckpt(path, fingerprint, rows):
    path.write_text(json.dumps({"fingerprint": fingerprint, "config": CONFIG, "dataset_ids": [], "results": rows}), encoding="utf-8")


def test_matching_checkpoint_is_merged_and_filtered_to_dataset_ids(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    write_ckpt(ckpt, "abc", [row("q0"), row("stale")])
    results, save_to = load_checkpoint(ckpt, "abc", ["q0", "q1"], explicit=True)
    assert [r["id"] for r in results] == ["q0"]
    assert save_to == ckpt


def test_mismatched_fingerprint_is_not_merged_nor_overwritten(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    write_ckpt(ckpt, "other", [row("q0")])
    results, save_to = load_checkpoint(ckpt, "abc", ["q0"], explicit=False)
    assert results == [] and save_to != ckpt and not save_to.exists()


def test_mismatched_fingerprint_on_explicit_path_raises(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    write_ckpt(ckpt, "other", [row("q0")])
    with pytest.raises(ValueError, match="--no-checkpoint"):
        load_checkpoint(ckpt, "abc", ["q0"], explicit=True)


def test_legacy_list_checkpoint_is_not_merged(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(json.dumps([row("q0")]), encoding="utf-8")
    results, save_to = load_checkpoint(ckpt, "abc", ["q0"], explicit=False)
    assert results == [] and save_to != ckpt
    with pytest.raises(ValueError):
        load_checkpoint(ckpt, "abc", ["q0"], explicit=True)


def test_missing_checkpoint_starts_clean(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    assert load_checkpoint(ckpt, "abc", ["q0"], explicit=False) == ([], ckpt)


# --- T2: end-to-end run ------------------------------------------------------------------------------

def test_run_writes_fingerprinted_default_checkpoint_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = write_dataset(tmp_path, n=3)
    e = make_evaluator()
    results = e.run_paired_benchmark(dataset_path=str(dataset), max_samples=2, delay_seconds=0)
    assert [r["id"] for r in results] == ["q0", "q1"]

    ckpt = e.last_checkpoint_path
    assert ckpt.parent == Path("data") and ckpt.name.startswith("eval_checkpoint_") and ckpt.suffix == ".json"
    assert ckpt.with_suffix(".csv").exists()
    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    assert set(saved) == {"fingerprint", "config", "dataset_ids", "results"}
    assert saved["dataset_ids"] == ["q0", "q1"]
    assert ckpt.name == f"eval_checkpoint_{saved['fingerprint']}.json"

    e2 = make_evaluator()
    resumed = e2.run_paired_benchmark(dataset_path=str(dataset), max_samples=2, delay_seconds=0)
    assert [r["id"] for r in resumed] == ["q0", "q1"]
    assert e2.calls == []  # nothing re-run

    e3 = make_evaluator()
    e3.run_paired_benchmark(dataset_path=str(dataset), max_samples=3, delay_seconds=0)
    assert e3.last_checkpoint_path != ckpt  # a different dataset slice never merges into the 2-sample run


def test_explicit_checkpoint_path_is_used(tmp_path):
    dataset = write_dataset(tmp_path, n=1)
    target = tmp_path / "mine.json"
    e = make_evaluator()
    e.run_paired_benchmark(dataset_path=str(dataset), checkpoint_path=str(target), delay_seconds=0)
    assert e.last_checkpoint_path == target and target.exists()


# --- T5: seeded arm order and warm-up ------------------------------------------------------------

def test_arm_order_is_deterministic_for_a_seed_and_mixes_both_orders():
    ids = [f"q{i}" for i in range(20)]
    a = ev.arm_orders(ids, seed=42)
    assert a == ev.arm_orders(ids, seed=42)
    assert set(a.values()) == {"naive_first", "hybrid_first"}
    assert any(ev.arm_orders(ids, seed=s) != a for s in range(1, 5))


def test_seed_changes_fingerprint():
    e = make_evaluator()
    assert run_fingerprint(["q0"], e.run_config(seed=1)) != run_fingerprint(["q0"], e.run_config(seed=2))


def test_run_follows_arm_order_and_records_it(tmp_path):
    dataset = write_dataset(tmp_path, n=6)
    e = make_evaluator()
    results = e.run_paired_benchmark(dataset_path=str(dataset), checkpoint_path=str(tmp_path / "c.json"), delay_seconds=0, seed=7)
    expected = ev.arm_orders([f"q{i}" for i in range(6)], seed=7)
    assert {r["id"]: r["arm_order"] for r in results} == expected

    timed = [c for c in e.calls if c[0] == "retrieve"][2:]  # skip the two warm-up retrievals
    for i, r in enumerate(results):
        first = timed[2 * i][2]
        assert first == (RetrievalStrategy.NAIVE_VECTOR if r["arm_order"] == "naive_first" else RetrievalStrategy.HYBRID_ADAPTIVE)


def test_warm_up_runs_once_before_timed_calls_and_only_when_pending(tmp_path):
    dataset = write_dataset(tmp_path, n=3)
    ckpt = tmp_path / "c.json"
    e = make_evaluator()
    e.run_paired_benchmark(dataset_path=str(dataset), checkpoint_path=str(ckpt), delay_seconds=0)
    retrievals = [c for c in e.calls if c[0] == "retrieve"]
    assert len(retrievals) == 2 + 3 * 2  # one warm-up per arm, then two arms per question
    assert {c[2] for c in retrievals[:2]} == {RetrievalStrategy.NAIVE_VECTOR, RetrievalStrategy.HYBRID_ADAPTIVE}
    assert all(c[1] == "question 0" for c in retrievals[:2])

    e2 = make_evaluator()
    e2.run_paired_benchmark(dataset_path=str(dataset), checkpoint_path=str(ckpt), delay_seconds=0)
    assert e2.calls == []  # nothing pending: no warm-up
