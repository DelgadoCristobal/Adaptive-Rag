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
- [ ] **T2: Checkpoint keying.** Key the checkpoint by a hash of the dataset ids and the run config. Filter resumed rows to the current dataset ids, and trim them to `--samples`.
- [ ] **T3: Judge context alignment.** The synthesizer returns the exact context block it used, and faithfulness is judged against that block.
- [ ] **T4: Correctness judge.** Judge the answer against `ground_truth` (correct / missing / hallucinated, CRAG-style) and add it to the outcome family.
- [ ] **T5: Arm order.** Randomise the naive/hybrid order per question with a seed, and add a warm-up query before timing.

## Acceptance criteria

- Offline tests cover each task's behaviour and pass.
- The existing offline suite stays green.

## Checks

```
.venv/Scripts/python -m pytest -q --ignore=tests/test_vector.py -k "not test_answer_synthesizer and not test_cross_encoder_reranker"
```

Baseline before the work: 23 passed (audit run).

## Delivery

Strategy: `ask-on-risk`. The forecast is about 600 authored lines across T1-T5, which is over the 400 budget, so the chain strategy will be asked before the commit that crosses it.

## Progress

| Task | Route | Commit | Checks | Review |
|---|---|---|---|---|
| T1 | delegated writer (statistics module plus tests) | see git log `fix(evaluation): ...statistics` | RED: ImportError, 1 failed, then the constant-Δ test failed. GREEN: offline suite 35 passed, 2 deselected | pending assess |

T1 notes:
- `is_significant` inside `generate_full_report` now uses the Holm-adjusted p. A standalone `analyze_metric` call uses the raw p.
- New keys: `p_value_adjusted`, `direction`, `diagnostic`, `better_arm`, `improved`.
- Removed keys: `shapiro_*` and `is_normal`, which no consumer read.
- The parent fixed one case: a constant non-zero Δ now goes to Wilcoxon. Only all-zero Δ forces p = 1.

Accepted limitation: Holm is computed on p rounded to 5 decimals, so the error is at most about 1e-5.

## Next step

T2.
