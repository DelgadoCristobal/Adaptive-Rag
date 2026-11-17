# Feature: graph-quality (Phase 2 of the repo audit)

Locator: `odd/tasks/graph-quality.md` · Engram mirror: `odd/graph-quality/tasks`

## Objective

Make the Kuzu graph trustworthy before the Phase 3 re-index and benchmark re-run.

## Problem

Evidence comes from the read-only map made on 2026-11-17.

- **Extraction falls back silently.** On any failure, `src/graph/extractor.py:85-88` logs at INFO and switches to a capitalised-word heuristic. That heuristic creates `ASSOCIATED_WITH` edges between items of an unordered set, so the edges are random. 1213 of the 1611 vault concept notes come from this fallback.
- **The extraction budget is too small.** With `num_ctx` 2048 and `max_tokens` 512, the JSON output gets cut off. `main.py` and `app.py` pass whole files, and Ollama truncates them silently.
- **Relation writes interpolate values into Cypher.** At `src/graph/kuzu_store.py:137-141` the description is escaped by hand, and a failure is logged only at DEBUG. Kuzu 0.11.3 accepts `$params` in CREATE/MERGE relation property maps, as an in-memory test showed.
- **Node upsert is destructive.** It overwrites `name`, `type` and `description` with whatever the latest document says.
- **Concept-note filenames are not sanitized** (`concept_{entity.id}.md`). Note provenance is a single `source:` scalar that each new document overwrites. That provenance feeds context precision through `graph_retriever.py:19-41`, and it is also read by `dataset_filter.py:44-52`.
- **The three ingestion paths diverge.** `main.py`, `crag_loader.py` and `app.py` differ in input size and `source_doc` naming. The UI path never writes notes, and `main.py` sleeps 4.5 s per document even on Ollama.

## Decisions

- When extraction fails for a chunk, **write nothing to the graph for that chunk**. Log a WARNING and count the failure. The random-edge heuristic is removed from the write path.
- Extraction is split into chunks sized from settings so each call fits inside `num_ctx`. Results are merged across chunks before writing.
- Provenance accumulates as a list. Readers still accept the legacy `source:` field.
- The existing `db_kuzu` and vault data are not migrated. Phase 3 re-indexes from scratch.

## Scope

Graph, extraction, ingestion and note readers, plus the context-precision provenance check.

Out of scope:
- cross-document entity resolution and dedup threshold tuning;
- k-hop query quality;
- choosing the extraction model (a Phase 3 decision);
- UI display.

## Tasks

### Slice A: `feat/graph-store-safety`

- [x] **T1: Kuzu writes.** A parameterized relation MERGE replaces both the f-string and the duplicate-check query. A non-destructive node upsert keeps the existing non-empty or longer description. Edge write failures log at WARNING. `hops` is cast to `int`. Tests check real edge counts and an escaping round-trip.
- [x] **T2: Notes.** Filenames become a sanitized slug plus a short hash of the id. Provenance accumulates as a `sources:` list, merged with any existing note. Readers (`graph_retriever`, `dataset_filter`) parse `sources:` and fall back to `source:`. Candidates gain a `source_doc_ids` list, the old `source_doc_id` is kept, and context precision matches any of them.

### Slice B: `feat/graph-extraction`

- [ ] **T3: Chunked, visible extraction.** Chunk size and `max_tokens` come from settings. Each chunk is extracted on its own and the results are merged. A failure is a WARNING, counted, and written nowhere. The extraction result reports whether the fallback was used and how many chunks failed.
- [ ] **T4: One `index_document`.** All three ingestion paths call it. The UI now writes notes. `source_doc` naming is normalized. The unconditional sleep in `main.py` is removed for Ollama.

## Checks

```
.venv/Scripts/python -m pytest -q --ignore=tests/test_vector.py -k "not test_answer_synthesizer and not test_cross_encoder_reranker"
```

Baseline: 75 passed. Tests use Kuzu in-memory or under `tmp_path`, with a stubbed LLM client.

## Delivery

Stacked-to-main, following on from Phase 1:

```
master ← … ← feat/eval-judges (#5) ← feat/graph-store-safety ← feat/graph-extraction
```

The forecast is about 350 lines for slice A and about 450 for slice B.

## Progress

| Task | Route | Commit | Checks | Review |
|---|---|---|---|---|
| T1 | delegated writer | 7947623 | RED: 7 failed. GREEN: 11 passed in `test_graph_store.py`; offline suite 86 passed | — |
| T2 | delegated writer | b6877eb | RED: ImportError on `concept_note_path`. GREEN: offline suite 94 passed, 2 deselected | — |

T1 and T2 notes:
- Relations are written with a parameterized `MERGE ... ON CREATE SET`.
- The node upsert uses CASE so it never loses a real name, type or description. A placeholder (name equal to id, or type CONCEPT) can be upgraded.
- `hops` is clamped to 1..5; Kuzu rejects upper bounds above 30.
- Notes are named `concept_<slug60>_<sha1[:8]>.md` and the frontmatter carries `sources: [..]`. Readers still accept legacy `source:`.
- **Context precision now matches ids exactly** after normalizing (lowercase, `.md`, `_chunk_N` and `crag_doc_` stripped). It used to match substrings, so the numbers can change.
- Carried to T4:
  - `community_retriever` labels notes by filename stem, which is now `slug_hash`; it should read `id:` from the frontmatter.
  - The `source_doc_ids` of a graph path takes the docs of only one endpoint; it should union both.

## Next step

Slice B (T3 and T4) on `feat/graph-extraction`.
