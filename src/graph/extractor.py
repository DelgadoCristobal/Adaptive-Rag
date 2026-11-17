import logging
import re
from typing import List, Dict
from pathlib import Path
import yaml
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import settings
from .schema import (
    Entity, Relation, ExtractionResult, ConceptNote, concept_note_path, parse_note_frontmatter, parse_note_sources
)

logger = logging.getLogger(__name__)

_WIKILINK_LINE_RE = re.compile(r"^- (\[\[.*\]\])\s*$", re.MULTILINE)
_DESCRIPTION_RE = re.compile(r"^\*\*Description:\*\*\n(.*?)\n\n## Related Concepts", re.MULTILINE | re.DOTALL)

EXTRACTION_PROMPT = """You are an expert Knowledge Graph Architect. Extract all key domain entities and directed relationships connecting them from the provided text.
Ensure relation types are normalized to UPPER_SNAKE_CASE (for example: FOUNDED_BY, LOCATED_IN, PRODUCES, LEADS, PART_OF, ACQUIRED_BY, ASSOCIATED_WITH).
Do NOT output placeholders like 'Canonical_Entity_Id' or 'UPPER_SNAKE_CASE_RELATION'; use the actual names and identifiers extracted from the text.

Example format:
{{
  "entities": [
    {{
      "id": "Google",
      "name": "Google LLC",
      "type": "ORGANIZATION",
      "description": "Technology company specializing in search and cloud computing"
    }},
    {{
      "id": "Sundar_Pichai",
      "name": "Sundar Pichai",
      "type": "PERSON",
      "description": "Chief Executive Officer of Alphabet and Google"
    }}
  ],
  "relationships": [
    {{
      "source": "Sundar_Pichai",
      "relation_type": "LEADS",
      "target": "Google",
      "description": "Sundar Pichai serves as CEO of Google",
      "weight": 1.0
    }}
  ]
}}

Respond ONLY with a valid JSON object matching this structure.

Text to analyze:
{text}
"""


from src.utils.llm_client import LLMClient, extract_json, qualify_model

SYSTEM_PROMPT = "You extract structured knowledge graphs in strict JSON format."
CHARS_PER_TOKEN = 4  # rough estimate, good enough to keep a call inside num_ctx
MIN_CHUNK_CHARS = 200
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def effective_chunk_chars(chunk_chars: int, max_tokens: int, num_ctx: int, prompt_chars: int) -> int:
    """Largest chunk size (chars) such that prompt + chunk + output fit in num_ctx tokens."""
    available_tokens = num_ctx - max_tokens - prompt_chars // CHARS_PER_TOKEN
    return max(MIN_CHUNK_CHARS, min(chunk_chars, available_tokens * CHARS_PER_TOKEN))


def _pack(pieces: List[str], max_chars: int, sep: str) -> List[str]:
    """Greedily join consecutive pieces with sep while the result stays within max_chars."""
    packed: List[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of at most max_chars, preferring paragraph, then sentence boundaries."""
    units: List[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        sentences: List[str] = []
        for sentence in _SENTENCE_END_RE.split(paragraph):
            sentences.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
        units.extend(_pack(sentences, max_chars, " "))
    return _pack(units, max_chars, "\n\n")


def _merge_extractions(results: List[ExtractionResult]) -> ExtractionResult:
    """Union chunk results: entities by normalized id, relations by (source, target, type)."""
    entities: Dict[str, Entity] = {}
    relations: Dict[tuple, Relation] = {}
    for result in results:
        for ent in result.entities:
            key = ent.id.lower()
            kept = entities.get(key)
            if kept is None:
                entities[key] = ent.model_copy(deep=True)
                continue
            kept.name = kept.name or ent.name
            kept.type = kept.type or ent.type
            if len(ent.description) > len(kept.description):
                kept.description = ent.description
            kept.aliases = list(dict.fromkeys(kept.aliases + ent.aliases))
        for rel in result.relations:
            key = (rel.source.lower(), rel.target.lower(), rel.relation_type)
            kept = relations.get(key)
            if kept is None:
                relations[key] = rel.model_copy(deep=True)
                continue
            if len(rel.description) > len(kept.description):
                kept.description = rel.description
            kept.weight = max(kept.weight, rel.weight)
    return ExtractionResult(entities=list(entities.values()), relations=list(relations.values()))


class EntityRelationExtractor:
    """Extractor for entities and relationships supporting Gemini, Groq, Ollama, and OpenAI."""

    def __init__(self, model: str = None):
        self.model = qualify_model(model)
        self.llm_client = LLMClient(default_model=self.model)

    def _chunk_chars(self) -> int:
        """Configured chunk size, shrunk once (with a WARNING) when an Ollama call would overflow num_ctx."""
        cached = getattr(self, "_cached_chunk_chars", None)
        if cached is not None:
            return cached
        configured = settings.extraction_chunk_chars
        chunk_chars = configured
        if str(self.model).startswith("ollama/"):
            num_ctx = int(getattr(self.llm_client, "num_ctx", settings.ollama_num_ctx))
            chunk_chars = effective_chunk_chars(
                configured, settings.extraction_max_tokens, num_ctx,
                prompt_chars=len(EXTRACTION_PROMPT) + len(SYSTEM_PROMPT),
            )
            if chunk_chars < configured:
                logger.warning(
                    "extraction_chunk_chars=%d with extraction_max_tokens=%d does not fit num_ctx=%d; "
                    "using %d chars per chunk", configured, settings.extraction_max_tokens, num_ctx, chunk_chars,
                )
        self._cached_chunk_chars = chunk_chars
        return chunk_chars

    def _extract_chunk(self, chunk: str, source_doc: str) -> ExtractionResult:
        response = self.llm_client.complete(
            prompt=EXTRACTION_PROMPT.format(text=chunk),
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            temperature=0.0,
            max_tokens=settings.extraction_max_tokens,
        )
        data = extract_json(response)
        if data is None:
            raise ValueError("no valid JSON object in LLM response")
        return ExtractionResult.model_validate(data)

    def extract_from_text(self, text: str, source_doc: str = "unknown") -> ExtractionResult:
        """Extract entities and relations chunk by chunk; a failed chunk is logged, counted and contributes nothing."""
        if not text.strip():
            return ExtractionResult()

        chunks = split_into_chunks(text, self._chunk_chars())
        results: List[ExtractionResult] = []
        failed = 0
        for index, chunk in enumerate(chunks, start=1):
            try:
                results.append(self._extract_chunk(chunk, source_doc))
            except Exception as e:
                failed += 1
                logger.warning("Extraction failed for %s chunk %d/%d: %s", source_doc, index, len(chunks), e)

        if failed == len(chunks):
            logger.warning("Extraction produced nothing for %s: all %d chunks failed", source_doc, len(chunks))
            return ExtractionResult(chunks_total=len(chunks), chunks_failed=failed)

        # Reconcile entity name/id aliases between relations and entities, once over the merged result
        merged = self._reconcile_and_link(_merge_extractions(results), source_doc)
        merged.chunks_total = len(chunks)
        merged.chunks_failed = failed
        return merged

    def _reconcile_and_link(self, extraction: ExtractionResult, source_doc: str) -> ExtractionResult:
        """Harmonizes relation endpoints with entity IDs, matching lowercase and title variants."""
        entities_by_id = {e.id: e for e in extraction.entities}
        name_to_id = {}
        for e in extraction.entities:
            name_to_id[e.id.lower()] = e.id
            name_to_id[e.name.lower()] = e.id
            name_to_id[e.id.replace("_", " ").lower()] = e.id
            name_to_id[e.name.replace(" ", "_").lower()] = e.id

        reconciled_relations: List[Relation] = []
        for rel in extraction.relations:
            src_key = rel.source.strip()
            tgt_key = rel.target.strip()
            if not src_key or not tgt_key or src_key == tgt_key:
                continue

            resolved_src = name_to_id.get(src_key.lower()) or name_to_id.get(src_key.replace(" ", "_").lower()) or src_key
            resolved_tgt = name_to_id.get(tgt_key.lower()) or name_to_id.get(tgt_key.replace(" ", "_").lower()) or tgt_key

            # Synthesize missing endpoint entities if LLM discovered a relation without declaring node
            if resolved_src not in entities_by_id:
                new_e = Entity(id=resolved_src, name=resolved_src.replace("_", " ").title(), type="CONCEPT", description=f"Referenced in {source_doc}")
                entities_by_id[resolved_src] = new_e
                name_to_id[resolved_src.lower()] = resolved_src

            if resolved_tgt not in entities_by_id:
                new_e = Entity(id=resolved_tgt, name=resolved_tgt.replace("_", " ").title(), type="CONCEPT", description=f"Referenced in {source_doc}")
                entities_by_id[resolved_tgt] = new_e
                name_to_id[resolved_tgt.lower()] = resolved_tgt

            reconciled_relations.append(Relation(
                source=resolved_src,
                target=resolved_tgt,
                relation_type=rel.relation_type,
                description=rel.description,
                weight=rel.weight
            ))

        return ExtractionResult(
            entities=list(entities_by_id.values()),
            relations=reconciled_relations
        )

    def create_atomic_concept_notes(
        self, extraction: ExtractionResult, source_doc: str, vault_dir: Path = None
    ) -> List[ConceptNote]:
        """Convert extracted entities & relations into atomic Obsidian Markdown concept notes with wikilinks."""
        vault_dir = Path(vault_dir) if vault_dir else settings.data_vault_dir
        concept_notes = []

        # Map relations per source entity
        rel_map: Dict[str, List[Relation]] = {}
        for rel in extraction.relations:
            rel_map.setdefault(rel.source, []).append(rel)

        for entity in extraction.entities:
            wikilinks = []
            related = rel_map.get(entity.id, [])
            for r in related:
                wikilinks.append(f"[[{r.target}|{r.relation_type}]]")

            # Write concept note to vault concepts subfolder, merging with the note other documents wrote
            note_path = concept_note_path(vault_dir, entity.id)
            note_path.parent.mkdir(parents=True, exist_ok=True)
            description = entity.description or ""
            sources, aliases = [source_doc], list(entity.aliases)
            if note_path.exists():
                existing = note_path.read_text(encoding="utf-8", errors="ignore")
                meta = parse_note_frontmatter(existing)
                sources = parse_note_sources(existing) + sources
                aliases = [str(a) for a in (meta.get("aliases") or [])] + aliases
                wikilinks = _WIKILINK_LINE_RE.findall(existing) + wikilinks
                old_desc = _DESCRIPTION_RE.search(existing)
                if old_desc and len(old_desc.group(1).strip()) > len(description):
                    description = old_desc.group(1).strip()
            sources = list(dict.fromkeys(sources))
            aliases = list(dict.fromkeys(aliases))
            wikilinks = list(dict.fromkeys(wikilinks))

            concept_note = ConceptNote(
                title=entity.name,
                entity_type=entity.type,
                description=description,
                wikilinks=wikilinks,
                source_documents=sources
            )
            concept_notes.append(concept_note)

            frontmatter = {
                "id": entity.id,
                "type": entity.type,
                "tags": ["concept", entity.type.lower()],
                "sources": sources,
                "aliases": aliases
            }

            wikilink_lines = "\n".join([f"- {link}" for link in wikilinks]) if wikilinks else "*No outgoing relations*"
            content = (
                f"---\n{yaml.dump(frontmatter, sort_keys=False, default_flow_style=None, allow_unicode=True, width=10**6)}---\n\n"
                f"# {entity.name}\n\n"
                f"**Type:** {entity.type}\n\n"
                f"**Description:**\n{description}\n\n"
                f"## Related Concepts (Wikilinks)\n{wikilink_lines}\n"
            )
            note_path.write_text(content, encoding="utf-8")

        return concept_notes
