import json
import logging
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import yaml
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import settings
from .schema import Entity, Relation, ExtractionResult, ConceptNote

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an expert Knowledge Graph Architect. Extract all domain entities, types, properties, and directed relations from the text.

Respond ONLY with valid JSON following this exact JSON Schema:
{
  "entities": [
    {
      "id": "Canonical_Entity_Id",
      "name": "Entity Name",
      "type": "ORGANIZATION | PERSON | TECHNOLOGY | PROJECT | CONCEPT | LOCATION | EVENT",
      "description": "Short explanation of entity in this document context",
      "properties": {},
      "aliases": ["Alias1"]
    }
  ],
  "relations": [
    {
      "source": "Canonical_Entity_Id_1",
      "target": "Canonical_Entity_Id_2",
      "relation_type": "RELATION_NAME_UPPERCASE",
      "description": "Short explanation of the relation",
      "weight": 1.0
    }
  ]
}

Text to analyze:
{text}
"""


class EntityRelationExtractor:
    """Extractor for entities and relationships using LLM structured output with fallback."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        
        if self.api_key and self.api_key != "mock-key":
            self.client = OpenAI(api_key=self.api_key, base_url=settings.openai_base_url)
        else:
            self.client = None

    def extract_from_text(self, text: str, source_doc: str = "unknown") -> ExtractionResult:
        """Extract structured entities and relations from a text snippet."""
        if not text.strip():
            return ExtractionResult()

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You extract structured knowledge graphs in strict JSON format."},
                        {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}
                    ],
                    temperature=0.0,
                )
                raw_json = response.choices[0].message.content
                data = json.loads(raw_json)
                return ExtractionResult.model_validate(data)
            except Exception as e:
                logger.warning(f"LLM Extraction failed: {e}. Falling back to rule-based extraction.")

        return self._heuristic_fallback_extraction(text, source_doc)

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

            # Write concept note to vault file
            note_filename = f"concept_{entity.id}.md"
            note_path = vault_dir / note_filename
            
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
