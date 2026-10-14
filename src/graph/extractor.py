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
from .schema import Entity, Relation, ExtractionResult, ConceptNote

logger = logging.getLogger(__name__)

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


class EntityRelationExtractor:
    """Extractor for entities and relationships supporting Gemini, Groq, Ollama, and OpenAI."""

    def __init__(self, model: str = None):
        self.model = qualify_model(model)
        self.llm_client = LLMClient(default_model=self.model)


    def extract_from_text(self, text: str, source_doc: str = "unknown") -> ExtractionResult:
        """Extract structured entities and relations from a text snippet."""
        if not text.strip():
            return ExtractionResult()

        try:
            response = self.llm_client.complete(
                prompt=EXTRACTION_PROMPT.format(text=text),
                system_prompt="You extract structured knowledge graphs in strict JSON format.",
                model=self.model,
                temperature=0.0,
                max_tokens=512,
            )
            data = extract_json(response)
            if data is None:
                raise ValueError("no valid JSON object in LLM response")
            extraction = ExtractionResult.model_validate(data)

            # Reconcile entity name/id aliases between relations and entities
            return self._reconcile_and_link(extraction, source_doc)
        except Exception as e:
            logger.info(f"LLM extraction notice ({e}). Falling back to zero-cost heuristic extraction.")

        return self._heuristic_fallback_extraction(text, source_doc)

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

    def _heuristic_fallback_extraction(self, text: str, source_doc: str) -> ExtractionResult:
        """Rule-based extractor fallback when LLM is unavailable."""
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []

        # Find capitalized terms/phrases as potential entities
        candidates = set(re.findall(r'\b[A-Z][a-zA-Z0-9_\-]{2,}\b', text))
        ignored = {"The", "This", "That", "With", "From", "Have", "Were", "Been", "Which", "Their", "Where", "When", "Slide", "Notes", "Error"}
        candidates = candidates - ignored

        for cand in candidates:
            ent_id = cand.replace(" ", "_")
            entities[ent_id] = Entity(
                id=ent_id,
                name=cand,
                type="CONCEPT",
                description=f"Extracted concept from {source_doc}",
                aliases=[cand]
            )

        # Simple co-occurrence relations between consecutive candidates
        cand_list = list(candidates)
        for i in range(len(cand_list) - 1):
            src = cand_list[i].replace(" ", "_")
            tgt = cand_list[i+1].replace(" ", "_")
            if src != tgt:
                relations.append(Relation(
                    source=src,
                    target=tgt,
                    relation_type="ASSOCIATED_WITH",
                    description=f"Co-occurs in {source_doc}",
                    weight=0.8
                ))

        return ExtractionResult(entities=list(entities.values()), relations=relations)

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

            concept_note = ConceptNote(
                title=entity.name,
                entity_type=entity.type,
                description=entity.description,
                wikilinks=wikilinks,
                source_documents=[source_doc]
            )
            concept_notes.append(concept_note)

            # Write concept note to vault concepts subfolder
            concepts_dir = vault_dir / "concepts"
            concepts_dir.mkdir(parents=True, exist_ok=True)
            note_filename = f"concept_{entity.id}.md"
            note_path = concepts_dir / note_filename
            
            frontmatter = {
                "id": entity.id,
                "type": entity.type,
                "tags": ["concept", entity.type.lower()],
                "source": source_doc,
                "aliases": entity.aliases
            }
            
            wikilink_lines = "\n".join([f"- {link}" for link in wikilinks]) if wikilinks else "*No outgoing relations*"
            content = (
                f"---\n{yaml.dump(frontmatter, sort_keys=False)}---\n\n"
                f"# {entity.name}\n\n"
                f"**Type:** {entity.type}\n\n"
                f"**Description:**\n{entity.description}\n\n"
                f"## Related Concepts (Wikilinks)\n{wikilink_lines}\n"
            )
            note_path.write_text(content, encoding="utf-8")

        return concept_notes
