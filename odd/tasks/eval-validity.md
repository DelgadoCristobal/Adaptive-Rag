# Feature: eval-validity (Phase 1 of the repo audit)

Locator: `odd/tasks/eval-validity.md` · Engram mirror: `odd/eval-validity/tasks`
Branch: `feat/eval-validity` (branched from `feat/crag-40-eval-report` @ 5fa3b2a)

## Objective

Make the naive-vs-hybrid RAG benchmark produce numbers that can be defended before any new run (40 or 200 questions).

## Problem

The audit found that the current evaluation pipeline can report spurious conclusions:

- The "Rank-Biserial r" effect size is a sign-test ratio that ignores ranks (`src/evaluation/statistics.py:86-88`).
- Five tests are run with no multiplicity correction. One-sided tests are paired with a two-sided CI. Direction is inferred from the substring "ms". The test is chosen by a Shapiro pre-test.
- "Cross-Encoder Top-1" favours hybrid by construction because the hybrid pool is a superset of the naive pool, yet it feeds `_conclusions`. A hard-coded narrative claims factual queries favour the vector baseline regardless of the data.
- The checkpoint path is fixed and is not keyed by dataset or config, so a 200-question run would merge the 40 existing rows (`src/evaluation/evaluator.py:295-321`).
- The faithfulness judge sees a different context from the one the generator saw (`evaluator.py:197,252` vs `src/generator/synthesizer.py:43-50`).
- `ground_truth` is never judged, so there is no correctness metric.
- The naive arm always runs first and absorbs model warm-up in latency.

## Scope

In scope: `src/evaluation/statistics.py`, `src/evaluation/evaluator.py`, `src/generator/synthesizer.py`, `main.py` (evaluate command only), and `tests/test_evaluation.py` (plus new test files under `tests/`).

Out of scope, deferred to later phases: graph extraction quality, Cypher parameters, UI bugs, judge temperature pinning, the refusal shortcut, dependencies, and README.

## Constraints

- Keep the existing result keys consumed by `src/ui/app.py`. Add new keys rather than renaming old ones.
- Tests must run offline, with no LLM, network, or HF model downloads; use fakes.
- Test-first: observe RED, then GREEN.
- Do not regenerate `reports/evaluation_report.md` here (Phase 3).

## Tasks

- [x] **T1: Statistics correctness.** Use Wilcoxon signed-rank for every metric, with no Shapiro selection. Use the matched-pairs rank-biserial effect size (Kerby). Make tests two-sided, with a matching two-sided bootstrap CI. Pre-declare each metric's direction in an explicit map. Apply Holm correction across the outcome family. Exclude Top-1 from the outcome family and the conclusions. Remove the hard-coded narrative. Route: delegated writer (statistics module plus tests, with design decisions given).
- [x] **T2: Checkpoint keying.** Key the checkpoint by a hash of the dataset ids and the run config. Filter resumed rows to the current dataset ids, and trim them to `--samples`.
- [x] **T3: Judge context alignment.** The synthesizer returns the exact context block it used, and faithfulness is judged against that block.
- [x] **T4: Correctness judge.** Judge the answer against `ground_truth` (correct / missing / hallucinated, CRAG-style) and add it to the outcome family.
- [x] **T5: Arm order.** Randomise the naive/hybrid order per question with a seed, and add a warm-up query before timing.

## Acceptance criteria

- Offline tests cover each task's behaviour and pass.
- The existing offline suite stays green.

## Checks

```
.venv/Scripts/python -m pytest -q --ignore=tests/test_vector.py -k "not test_answer_synthesizer and not test_cross_encoder_reranker"
```

Baseline before the work: 23 passed (audit run).

## Delivery

Strategy: `ask-on-risk`. Chain strategy: **stacked-to-main**, chosen by the user on 2026-11-17.

Each branch is stacked on the one before it:

```
master ← feat/crag-40-eval-report ← feat/eval-validity (T1) ← feat/eval-checkpoint (T2+T5) ← feat/eval-judges (T3+T4)
```

| Slice | Branch | Tasks | Commits |
|---|---|---|---|
| 1 | feat/eval-validity | T1 | b64c543 |
| 2 | feat/eval-checkpoint | T2, T5 | 65931c6, bef7f09, plus the UI checkpoint fix |
| 3 | feat/eval-judges | T3, T4 | fa1105f, 19c8935 |

T2 and T5 are grouped because both change the `run` loop in `evaluator.py`, and the arm-order seed must be part of the checkpoint fingerprint. T3 and T4 are grouped because both change the judge prompts and the synthesizer contract.

Push and PR creation remain the user's decision.

## Progress

| Task | Route | Commit | Checks | Review |
|---|---|---|---|---|
| T1 | delegated writer (statistics module plus tests) | see git log `fix(evaluation): ...statistics` | RED: ImportError, 1 failed, then the constant-Δ test failed. GREEN: offline suite 35 passed, 2 deselected | assess: risk medium; review_due false (under_budget), so the slice stays pending |

Commit: b64c543. The running count after T1 is 460 changed lines, over the budget, so the chain strategy must be decided before the T2 commit.

T1 notes:
- `is_significant` inside `generate_full_report` now uses the Holm-adjusted p. A standalone `analyze_metric` call uses the raw p.
- New keys: `p_value_adjusted`, `direction`, `diagnostic`, `better_arm`, `improved`.
- Removed keys: `shapiro_*` and `is_normal`, which no consumer read.
- The parent fixed one case: a constant non-zero Δ now goes to Wilcoxon. Only all-zero Δ forces p = 1.

Accepted limitation: Holm is computed on p rounded to 5 decimals, so the error is at most about 1e-5.

| T2 | delegated writer | 65931c6 | RED: ImportError on `load_checkpoint`. GREEN: 9 tests in `test_evaluator_run.py`; offline suite 44 passed | — |
| T5 | delegated writer | bef7f09 | RED: 4 failed. GREEN: offline suite 48 passed, 2 deselected | — |

T2 and T5 notes:
- The default checkpoint is now `data/eval_checkpoint_<fp>.json`. The fingerprint covers the sorted dataset ids (after `--samples`), `arm_order_seed`, `generator_model`, `judge_backend`, `judge_model`, `judge_fallbacks`, `vector_top_k` and `rerank_top_k`.
- A mismatched or legacy checkpoint is never merged or overwritten. On the default path the run starts clean at `<stem>_<n>.json`; on an explicit path it raises `ValueError`.
- Each row gets an `arm_order` field, and `main.py evaluate` gains `--seed`.
- Latency is not comparable with earlier runs.
- Parent fix: `src/ui/app.py` hard-coded `data/eval_checkpoint.json`. It now loads the newest `eval_checkpoint*.json` and unwraps `results`. Checked with `py_compile` and a temp-dir script.
- Declined, YAGNI: a `--checkpoint` CLI flag.

| T3 | delegated writer | fa1105f | RED: ImportError on `build_context_block`, then 2 failed. GREEN: offline suite 53 passed | — |
| T4 | delegated writer | 19c8935 | RED: ImportError on `CORRECTNESS_SCORES`, then 6 failed. GREEN: offline suite 75 passed, 2 deselected | — |

T3 and T4 notes:
- Faithfulness is judged against `synthesizer.context_block(retrieved)`, the same post-rerank list that is passed to `synthesize`. The parent spot-checked this at `evaluator.py:363,370`.
- Correctness labels are correct, missing and incorrect, scored 1, 0 and -1. A label that does not parse, or an empty `ground_truth`, records None, never 0.
- Statistics drop a pair when either arm has no value for that metric, and report `n` and `n_excluded`. The Holm family is 5 metrics when correctness is present and 4 on old checkpoints.
- `METRICS_VERSION = 2` goes into the fingerprint.
- Known gap: the UI does not show correctness yet. That belongs to Phase 4.

Feature-level verification: the offline suite passed 75, with 2 deselected because they need an LLM or HF model. `test_vector.py` was not run. No live benchmark has been run; that is Phase 3.

## Next step

Phase 1 is complete. Push and the stacked PRs are the user's decision. Then comes Phase 2 (graph quality), and Phase 3 (re-index and re-run the 40 and 200 benchmarks).
