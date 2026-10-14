from typing import List, Dict, Any, Optional, Tuple
import json
import os
import time
import random
import re
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.retrieval.router import RetrievalStrategy
from src.generator.synthesizer import AnswerSynthesizer
from src.utils.llm_client import CLIUnavailableError, GeminiCLIClient, LLMClient, extract_json, qualify_model

logger = logging.getLogger(__name__)
console = Console()

FAITHFULNESS_PROMPT = """You are a strict evaluation auditor. Evaluate whether the ANSWER is completely faithful to the CONTEXT.

Strict Scoring Rules:
- Score 1.0 ONLY if EVERY factual claim in the ANSWER is explicitly stated in the CONTEXT without any assumption.
- Score 0.5 if the ANSWER is mostly faithful but makes minor unverified inferences or extrapolations.
- Score 0.0 if the ANSWER contains ungrounded claims, hallucinations, external assumptions, or if CONTEXT is missing/empty.

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY in JSON format:
{{"faithfulness_score": 0.0, "reason": "concise explanation"}}
JSON:"""

RELEVANCE_PROMPT = """You are a strict evaluation auditor. Evaluate how directly and specifically the ANSWER answers the QUESTION.

Strict Scoring Rules:
- Score 1.0 ONLY if the ANSWER directly, specifically, and accurately answers the core question asked.
- Score 0.5 if the ANSWER is partially relevant, incomplete, or answers only around the question.
- Score 0.0 if the ANSWER is irrelevant, off-topic, evasive, or fails to answer what was asked.

QUESTION:
{question}

ANSWER:
{answer}

Respond ONLY in JSON format:
{{"relevance_score": 0.0, "reason": "concise explanation"}}
JSON:"""

REFUSAL_PHRASES = (
    "no information provided",
    "no mention",
    "does not provide",
    "cannot answer",
    "not enough information",
    "context does not contain",
    "information is not provided",
    "i am unable to answer",
)

_NUMBER = r"([0-9]+(?:\.[0-9]+)?|\.[0-9]+)"


def parse_judge_score(response: str, key: str) -> Tuple[float, str]:
    """Extract a [0, 1] score and reason from a judge reply, tolerating Markdown fences, chatter and truncated JSON.

    Raises ValueError when no score is present, so a malformed reply is never silently scored as 0.0.
    """
    data = extract_json(response) or {}
    raw = data.get(key, data.get("score"))
    reason = str(data.get("reason", "")).strip()
    try:
        score = float(raw)
    except (TypeError, ValueError):
        # Truncated/invalid JSON or prose: scan for `"<key>": 0.5` / `<key> = 1` / `score: 0.5`.
        match = re.search(rf"(?:{re.escape(key)}|score)\W{{0,4}}[:=]\s*\W{{0,2}}{_NUMBER}", response or "", re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse '{key}' from judge response: {response!r}")
        score = float(match.group(1))
        reason_match = re.search(r'"reason"\s*:\s*"(.*)', response)
        reason = reason_match.group(1).rstrip('"} \n') if reason_match else reason
    return min(1.0, max(0.0, score)), reason or "Evaluated by judge."


MAX_QUOTA_WAIT_SECONDS = 120.0


def _retry_delay(error_text: str) -> Optional[float]:
    """Seconds the provider asks us to wait ('retry in 11h19m35.4s', 'retry in 40.8s'), or None."""
    match = re.search(r"retry in\s*(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", error_text)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)


def _norm_id(value: Any) -> str:
    return str(value or "").lower().strip().replace(".md", "")


class RAGEvaluator:
    """Automated RAG Evaluation engine with rate limiting and row-by-row checkpointing."""

    def __init__(self, model_name: Optional[str] = None, judge_model_name: Optional[str] = None):
        self.model_name = qualify_model(model_name)
        self.judge_model_name = qualify_model(
            judge_model_name or ("gemini/gemini-3.8-flash" if os.getenv("GEMINI_API_KEY") else self.model_name),
            provider="gemini",
        )
        # Fallback judge models for resilience against 503 / 429 errors
        self.judge_fallbacks = (
            [m for m in ["gemini/gemini-3.5-flash-lite"] if m != self.judge_model_name]
            if self.judge_model_name.startswith("gemini/") else []
        )

        self.llm_client = LLMClient(default_model=self.model_name)
        # Subscription-backed CLI judge is tried first; LiteLLM (API key) is the fallback. JUDGE_BACKEND=api disables it.
        self.cli_judge = GeminiCLIClient() if os.getenv("JUDGE_BACKEND", "cli") != "api" else None
        if self.cli_judge and not self.cli_judge.available:
            logger.warning("No subscription CLI (agy/antigravity/gemini) on PATH: log in / install one. Using the API judge.")
            self.cli_judge = None
        self.hybrid_engine = HybridRetrievalEngine()
        self.synthesizer = AnswerSynthesizer(model=self.model_name)

    def _call_judge_with_retry(self, prompt: str, system_prompt: str, max_retries: int = 5) -> str:
        """Invokes the judge LLM with quota-aware / exponential-jitter backoff and model fallback.

        Synchronous by design (the evaluator never runs inside an event loop), so `time.sleep` blocks nothing else.
        """
        last_error = None

        if self.cli_judge:
            try:
                try:
                    return self.cli_judge.complete(prompt, system_prompt)
                except TimeoutError:  # agent start-up latency is spiky: one retry before paying API quota
                    logger.warning("CLI judge timed out; retrying once.")
                    return self.cli_judge.complete(prompt, system_prompt)
            except CLIUnavailableError as e:
                logger.warning("CLI judge disabled for this run: %s", e)
                self.cli_judge = None  # not logged in / unsupported: do not pay the process start-up on every call
            except Exception as e:
                logger.warning("CLI judge failed (%s); falling back to the API judge.", str(e)[:150])

        for model in [self.judge_model_name] + self.judge_fallbacks:
            for attempt in range(max_retries):
                try:
                    res = self.llm_client.complete(
                        prompt=prompt, system_prompt=system_prompt, model=model, temperature=0.0, max_tokens=128
                    )
                    if res and res.strip():
                        return res
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()

                    if any(code in err_str for code in ["404", "not_found", "not found", "not supported"]):
                        logger.warning("Judge model %s not found/supported. Skipping to next fallback...", model)
                        break

                    if any(code in err_str for code in ["429", "ratelimit", "resource_exhausted", "quota"]):
                        # Respect the quota window the provider reports; otherwise back off exponentially.
                        retry_after = _retry_delay(err_str)
                        if retry_after is not None and retry_after > MAX_QUOTA_WAIT_SECONDS:
                            logger.warning("Judge model %s quota exhausted (resets in %.0f min). Skipping to next fallback...", model, retry_after / 60)
                            break
                        sleep_seconds = retry_after + 2.0 if retry_after is not None else 22.0 * (1.5 ** attempt)
                    else:  # 503 / high demand / timeout / transient errors
                        sleep_seconds = 4.0 * (1.5 ** attempt)
                    sleep_seconds += random.uniform(0.5, 2.0)  # jitter
                    logger.warning(
                        "Judge error on %s (attempt %d/%d): %s. Waiting %.1fs...",
                        model, attempt + 1, max_retries, str(e)[:120], sleep_seconds
                    )
                    if attempt < max_retries - 1:
                        time.sleep(sleep_seconds)

            logger.warning("Exhausted retries on judge model %s. Trying next fallback model...", model)

        raise RuntimeError(f"All judge models and retries exhausted. Last error: {last_error}")

    def _judge(self, prompt: str, system_prompt: str, key: str) -> Tuple[float, str]:
        return parse_judge_score(self._call_judge_with_retry(prompt, system_prompt), key)

    def evaluate_faithfulness(self, context: str, answer: str) -> Tuple[float, str]:
        """Calculates faithfulness score in [0.0, 1.0] and justification reason."""
        if not answer or not context or any(phrase in answer.lower() for phrase in REFUSAL_PHRASES):
            return 1.0, "Properly refused to hallucinate when context is missing."

        return self._judge(
            FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer),
            "You are a strict evaluation judge. Output valid JSON.",
            "faithfulness_score",
        )

    def evaluate_relevance(self, question: str, answer: str) -> Tuple[float, str]:
        """Calculates answer relevance score in [0.0, 1.0] and justification reason."""
        if not answer:
            return 0.0, "Empty answer."

        return self._judge(
            RELEVANCE_PROMPT.format(question=question, answer=answer),
            "You are a strict relevance judge. Output valid JSON.",
            "relevance_score",
        )

    def evaluate_context_precision(self, expected_source: Any, retrieved_contexts: List[Dict[str, Any]]) -> float:
        """Reciprocal rank (top-5) of the first retrieved item matching an expected source document."""
        if not retrieved_contexts or not expected_source:
            return 0.0

        expected = expected_source if isinstance(expected_source, list) else [expected_source]
        target_ids = [_norm_id(x) for x in expected if x]

        for rank, item in enumerate(retrieved_contexts[:5], start=1):
            for key in ("source_doc_id", "source", "doc_id", "id"):
                # Drop the chunk suffix so chunk ids map back to their document.
                norm_src = _norm_id(item.get(key)).split("_chunk_")[0]
                if not norm_src:
                    continue
                for tid in target_ids:
                    clean_tid = tid.replace("crag_doc_", "")
                    if (
                        tid == norm_src
                        or clean_tid == norm_src.replace("crag_doc_", "")
                        or tid in norm_src
                        or (len(norm_src) >= 12 and norm_src in tid)
                        or (len(clean_tid) > 6 and clean_tid in norm_src)
                    ):
                        return 1.0 / rank

        return 0.0

    def _evaluate_arm(
        self, q_text: str, strategy: RetrievalStrategy, expected_src: Any, delay: float
    ) -> Dict[str, Any]:
        """Retrieve, generate and judge one experimental arm (control or treatment) for a single query."""
        t0 = time.time()
        retrieved, _ = self.hybrid_engine.retrieve(q_text, strategy=strategy)
        answer = self.synthesizer.synthesize(q_text, retrieved)["answer"]
        latency_ms = round((time.time() - t0) * 1000.0, 2)
        rerank_scores = [float(c.get("rerank_score", c.get("score", 0.0))) for c in retrieved]

        if delay > 0:
            time.sleep(delay)
        faithfulness, f_reason = self.evaluate_faithfulness("\n".join(c.get("text", "") for c in retrieved), answer)
        if delay > 0:
            time.sleep(delay)
        relevance, r_reason = self.evaluate_relevance(q_text, answer)

        return {
            "faithfulness": faithfulness,
            "faithfulness_reason": f_reason,
            "answer_relevance": relevance,
            "relevance_reason": r_reason,
            "context_precision": self.evaluate_context_precision(expected_src, retrieved),
            "latency_ms": latency_ms,
            "top_rerank_score": round(max(rerank_scores), 4) if rerank_scores else 0.0,
            "retrieved_sources": [c.get("source_doc_id") or c.get("source") for c in retrieved],
            "answer": answer,
        }

    @staticmethod
    def _save_checkpoint(results: List[Dict[str, Any]], ckpt_file: Path) -> None:
        """Persist the JSON checkpoint and a flat CSV mirror for quick analysis."""
        ckpt_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        metrics = [("f", "faithfulness"), ("r", "answer_relevance"), ("p", "context_precision")]
        rows = []
        for r in results:
            row = {"id": r["id"], "domain": r.get("domain", "general"), "type": r["type"]}
            for short, metric in metrics:
                row.update({
                    f"{short}_naive": r["naive"][metric],
                    f"{short}_hybrid": r["hybrid"][metric],
                    f"{short}_delta": r["delta"][metric],
                })
            row.update({
                "lat_naive_ms": r["naive"]["latency_ms"],
                "lat_hybrid_ms": r["hybrid"]["latency_ms"],
                "rerank_naive": r["naive"]["top_rerank_score"],
                "rerank_hybrid": r["hybrid"]["top_rerank_score"],
            })
            rows.append(row)
        pd.DataFrame(rows).to_csv(ckpt_file.with_suffix(".csv"), index=False)

    def run_paired_benchmark(
        self,
        dataset_path: str = "data/eval_dataset.json",
        checkpoint_path: str = "data/eval_checkpoint.json",
        max_samples: Optional[int] = None,
        delay_seconds: Optional[float] = None,
        no_checkpoint: bool = False
    ) -> List[Dict[str, Any]]:
        uses_gemini = self.judge_model_name.startswith("gemini/") or self.model_name.startswith("gemini/")
        effective_delay = delay_seconds if delay_seconds is not None else (4.5 if uses_gemini else 0.0)
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if max_samples:
            dataset = dataset[:max_samples]

        ckpt_file = Path(checkpoint_path)
        if no_checkpoint and ckpt_file.exists():
            ckpt_file.unlink()
            console.print("[bold yellow]Bypassing existing checkpoint (--no-checkpoint active). Starting clean evaluation.[/bold yellow]")

        results: List[Dict[str, Any]] = []
        if not no_checkpoint and ckpt_file.exists():
            try:
                results = json.loads(ckpt_file.read_text(encoding="utf-8"))
                console.print(f"[bold cyan]Resuming from checkpoint ({len(results)}/{len(dataset)} completed)[/bold cyan]")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Unreadable checkpoint %s (%s); starting clean.", ckpt_file, e)

        processed_ids = {r["id"] for r in results}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[{task.completed}/{task.total}]"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Evaluating Paired Queries...", total=len(dataset))
            progress.update(task, completed=len(results))

            for item in dataset:
                q_id = item["id"]
                if q_id in processed_ids:
                    continue

                q_text = item.get("question") or item.get("query", "")
                expected_src = item.get("gold_document_ids") or item.get("gold_doc_ids") or item.get("expected_source") or ""

                try:
                    naive = self._evaluate_arm(q_text, RetrievalStrategy.NAIVE_VECTOR, expected_src, effective_delay)
                    hybrid = self._evaluate_arm(q_text, RetrievalStrategy.HYBRID_ADAPTIVE, expected_src, effective_delay)
                except Exception as sample_err:
                    logger.error("Sample %s failed after retries and fallbacks (%s). Discarding to preserve statistical integrity.", q_id, sample_err)
                    console.print(f"[bold red]Discarding sample {q_id} due to judge failure: {sample_err}[/bold red]")
                    progress.update(task, advance=1)
                    continue

                results.append({
                    "id": q_id,
                    "type": item.get("type") or item.get("question_type", "general"),
                    "domain": item.get("domain", "general"),
                    "question": q_text,
                    "ground_truth": item.get("ground_truth") or item.get("answer", ""),
                    "expected_source": expected_src,
                    "naive": naive,
                    "hybrid": hybrid,
                    "delta": {
                        "faithfulness": round(hybrid["faithfulness"] - naive["faithfulness"], 4),
                        "answer_relevance": round(hybrid["answer_relevance"] - naive["answer_relevance"], 4),
                        "context_precision": round(hybrid["context_precision"] - naive["context_precision"], 4),
                        "latency_ms": round(hybrid["latency_ms"] - naive["latency_ms"], 2),
                        "top_rerank_score": round(hybrid["top_rerank_score"] - naive["top_rerank_score"], 4),
                    },
                })
                processed_ids.add(q_id)
                self._save_checkpoint(results, ckpt_file)
                progress.update(task, advance=1)

        return results
