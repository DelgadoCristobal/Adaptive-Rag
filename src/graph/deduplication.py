from typing import List, Dict
import logging
from src.config import settings
from .schema import Entity, Relation, ExtractionResult

logger = logging.getLogger(__name__)


try:
    import Levenshtein
    _ratio_fn = Levenshtein.ratio
except ImportError:
    import difflib
    _ratio_fn = lambda s1, s2: difflib.SequenceMatcher(None, s1, s2).ratio()


class EntityDeduplicator:
    """Levenshtein & Semantic clustering for entity resolution and canonicalization."""

    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or settings.fuzzy_dedup_threshold

    def calculate_levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity score (0.0 to 1.0)."""
        return _ratio_fn(s1.lower(), s2.lower())

    def deduplicate(self, extraction: ExtractionResult) -> ExtractionResult:
        """Merge duplicate entities and update relation source/target pointers."""
        if not extraction.entities:
            return extraction

        canonical_map: Dict[str, str] = {}  # original_id -> canonical_id
        unique_entities: Dict[str, Entity] = {}

        for entity in extraction.entities:
            matched_canonical_id = None

            for existing_id, existing_entity in unique_entities.items():
                ratio = self.calculate_levenshtein_ratio(entity.id, existing_id)
                if ratio >= self.similarity_threshold:
                    matched_canonical_id = existing_id
                    break

            if matched_canonical_id:
                # Merge aliases & descriptions into existing entity
                existing_ent = unique_entities[matched_canonical_id]
                new_aliases = set(existing_ent.aliases + entity.aliases + [entity.name, entity.id])
                existing_ent.aliases = list(new_aliases)
                if not existing_ent.description and entity.description:
                    existing_ent.description = entity.description
                canonical_map[entity.id] = matched_canonical_id
            else:
                canonical_map[entity.id] = entity.id
                if entity.id not in entity.aliases:
                    entity.aliases.append(entity.id)
                unique_entities[entity.id] = entity

        # Remap relations using canonical IDs
        updated_relations: List[Relation] = []
        for rel in extraction.relations:
            new_source = canonical_map.get(rel.source, rel.source)
            new_target = canonical_map.get(rel.target, rel.target)
            if new_source != new_target:  # avoid self-loops
                updated_relations.append(Relation(
                    source=new_source,
                    target=new_target,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight
                ))

        logger.info(f"Deduplicated {len(extraction.entities)} entities -> {len(unique_entities)} canonical entities.")
        return ExtractionResult(
            entities=list(unique_entities.values()),
            relations=updated_relations
        )
