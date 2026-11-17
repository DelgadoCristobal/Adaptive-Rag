"""Offline tests for the judges: faithfulness sees the generator's exact context; CRAG-style correctness.

No real LLM, retrieval engine or model is constructed: fakes replace them.
"""
import pytest

from src.evaluation.evaluator import CORRECTNESS_SCORES, RAGEvaluator, parse_correctness_label
from src.generator.synthesizer import SYNTHESIS_SYSTEM_PROMPT, AnswerSynthesizer, build_context_block
from src.retrieval.router import RetrievalStrategy

ITEMS = [
    {"text": "alpha " * 400, "source": "doc_a"},
    {"text": "beta " * 400, "source": "doc_b"},
    {"text": "ONLY_IN_ITEM_3 " + "gamma " * 300, "source": "doc_c"},
]


class FakeLLMClient:
    """Captures the system prompt the synthesizer sends to its LLM."""

    def __init__(self, num_ctx=2048, num_predict=512):
        self.num_ctx, self.num_predict = num_ctx, num_predict
        self.system_prompts = []

    def complete(self, prompt, system_prompt, model, temperature):
        self.system_prompts.append(system_prompt)
        return "an answer [3]"


def make_synthesizer(client=None):
    s = AnswerSynthesizer.__new__(AnswerSynthesizer)
    s.model = "ollama/fake"
    s.llm_client = client or FakeLLMClient()
    return s


# --- T3: the context block is one pure function shared by generator and judge ------------------------

def test_build_context_block_is_deterministic_and_truncates_per_item():
    a = build_context_block(ITEMS, window_tokens=1186)
    assert a == build_context_block(list(ITEMS), window_tokens=1186)
    assert a.startswith("[1] Source (doc_a): alpha")
    assert "[3] Source (doc_c): ONLY_IN_ITEM_3 gamma" in a
    per_item = 1186 * 3 // 3
    assert f"[1] Source (doc_a): {ITEMS[0]['text'][:per_item]}\n\n[2]" in a


def test_synthesize_sends_exactly_the_context_block_to_its_llm():
    client = FakeLLMClient()
    s = make_synthesizer(client)
    s.synthesize("q?", ITEMS)
    assert client.system_prompts == [SYNTHESIS_SYSTEM_PROMPT.format(context_block=s.context_block(ITEMS))]


# --- T3: the evaluator judges against that same block ---------------------------------------------

class RecordingSynth:
    """Wraps a real (fake-client) synthesizer and records every list of items it received."""

    def __init__(self):
        self.inner = make_synthesizer()
        self.received = []

    def synthesize(self, query, retrieved):
        self.received.append(retrieved)
        return self.inner.synthesize(query, retrieved)

    def context_block(self, items):
        return self.inner.context_block(items)


class ArmEngine:
    def retrieve(self, query, strategy):
        items = ITEMS if strategy == RetrievalStrategy.HYBRID_ADAPTIVE else ITEMS[:1]
        return [dict(i) for i in items], None


def make_evaluator():
    e = RAGEvaluator.__new__(RAGEvaluator)
    e.hybrid_engine = ArmEngine()
    e.synthesizer = RecordingSynth()
    e.judged_contexts = []
    e.evaluate_faithfulness = lambda context, answer: (e.judged_contexts.append(context), (1.0, "ok"))[1]
    e.evaluate_relevance = lambda question, answer: (1.0, "ok")
    e.evaluate_correctness = lambda question, ground_truth, answer: (1.0, "correct")
    return e


def test_judge_receives_the_synthesizer_context_block_for_each_arm():
    e = make_evaluator()
    e._evaluate_arm("q?", RetrievalStrategy.NAIVE_VECTOR, "", 0)
    e._evaluate_arm("q?", RetrievalStrategy.HYBRID_ADAPTIVE, "", 0)
    expected = [e.synthesizer.context_block(items) for items in e.synthesizer.received]
    assert e.judged_contexts == expected
    assert e.judged_contexts[0] != e.judged_contexts[1]


def test_content_only_in_item_3_is_visible_to_the_judge():
    e = make_evaluator()
    e._evaluate_arm("q?", RetrievalStrategy.HYBRID_ADAPTIVE, "", 0)
    assert "ONLY_IN_ITEM_3" in e.judged_contexts[0]


def test_faithfulness_prompt_is_not_cut_to_2000_chars():
    e = RAGEvaluator.__new__(RAGEvaluator)
    seen = []
    e._judge = lambda prompt, system_prompt, key: (seen.append(prompt), (1.0, "ok"))[1]
    context = "x" * 3000 + "TAIL"
    e.evaluate_faithfulness(context, "some answer")
    assert "TAIL" in seen[0]


# --- T4: CRAG-style correctness judge ---------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ('{"label": "correct"}', "correct"),
    ('```json\n{"label": "Missing"}\n```', "missing"),
    ('Verdict: {"label": " incorrect ", "reason": "wrong year"}', "incorrect"),
    ("", None),
    ("correct", None),
    ('{"label": "partially correct"}', None),
    ('{"label": "hallucinated"}', None),
    ('{"verdict": "correct"}', None),
    ('{"label": "correct"', None),
    ('{"label": 1}', None),
])
def test_parse_correctness_label_is_strict(reply, expected):
    assert parse_correctness_label(reply) == expected


def test_correctness_score_mapping_follows_crag():
    assert CORRECTNESS_SCORES == {"correct": 1.0, "missing": 0.0, "incorrect": -1.0}


def judge_evaluator(reply):
    e = RAGEvaluator.__new__(RAGEvaluator)
    e.prompts = []
    e._call_judge_with_retry = lambda prompt, system_prompt: (e.prompts.append(prompt), reply)[1]
    return e


@pytest.mark.parametrize("reply, expected", [
    ('{"label": "correct"}', (1.0, "correct")),
    ('{"label": "missing"}', (0.0, "missing")),
    ('{"label": "incorrect"}', (-1.0, "incorrect")),
])
def test_evaluate_correctness_scores_the_label_and_shows_ground_truth(reply, expected):
    e = judge_evaluator(reply)
    assert e.evaluate_correctness("Who?", "Ada Lovelace", "It was Ada [1].") == expected
    assert "Ada Lovelace" in e.prompts[0] and "It was Ada [1]." in e.prompts[0] and "Who?" in e.prompts[0]


def test_malformed_correctness_reply_is_none_not_zero(caplog):
    e = judge_evaluator("I think it's fine")
    with caplog.at_level("WARNING"):
        assert e.evaluate_correctness("Who?", "Ada", "Ada") == (None, None)
    assert "correctness" in caplog.text.lower()


def test_missing_ground_truth_is_none_and_skips_the_judge(caplog):
    e = judge_evaluator('{"label": "correct"}')
    with caplog.at_level("WARNING"):
        assert e.evaluate_correctness("Who?", "", "Ada") == (None, None)
    assert e.prompts == [] and "ground_truth" in caplog.text
