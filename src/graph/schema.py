from typing import List, Dict, Any
import re
from pydantic import BaseModel, Field, model_validator


def normalize_node_name(name: str) -> str:
    """Normalizes an entity name/id by trimming whitespace and formatting consistently."""
    if not name:
        return ""
    clean = name.strip()
    # Replace multiple whitespaces/hyphens with a clean single underscore if needed
    clean = re.sub(r'[\s\-]+', '_', clean)
    clean = re.sub(r'^[^\w]+|[^\w]+$', '', clean)
    return clean.strip()


class Entity(BaseModel):
    """Extracted entity node schema."""
    id: str = Field(description="Normalized unique identifier or canonical name")
    name: str = Field(description="Original entity name as found in text")
    type: str = Field(default="CONCEPT", description="Entity type/category (e.g. Person, Organization, Technology, Project, Concept)")
    description: str = Field(default="", description="Summary or description of the entity context")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Key-value dynamic attributes")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or acronyms")

    @model_validator(mode="before")
    @classmethod
    def prepare_entity_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            name = str(data.get("name") or data.get("id") or data.get("label") or "Unknown").strip()
            ent_id = str(data.get("id") or name).strip()
            ent_type = str(data.get("type") or data.get("category") or data.get("entity_type") or "CONCEPT").strip().upper()
            desc = str(data.get("description") or data.get("desc") or "").strip()
            
            data["name"] = name
            data["id"] = normalize_node_name(ent_id) or name
            data["type"] = ent_type
            data["description"] = desc
        return data


class Relation(BaseModel):
    """Extracted relationship edge schema."""
    source: str = Field(description="ID/Name of the source entity")
    target: str = Field(description="ID/Name of the target entity")
    relation_type: str = Field(default="RELATED_TO", description="Relationship type (e.g., DEPENDS_ON, USES, PART_OF, MANAGED_BY, CREATED_BY)")
    description: str = Field(default="", description="Contextual explanation of the relationship")
    weight: float = Field(default=1.0, description="Confidence score or edge weight (0.0 to 1.0)")

    @model_validator(mode="before")
    @classmethod
    def prepare_relation_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Resolve source aliases
            src = data.get("source") or data.get("from") or data.get("src") or data.get("head") or data.get("entity1") or ""
            # Resolve target aliases
            tgt = data.get("target") or data.get("to") or data.get("dst") or data.get("tail") or data.get("entity2") or ""
            # Resolve relation_type aliases
            rel_type = (
                data.get("relation_type")
                or data.get("relationship")
                or data.get("relation")
                or data.get("type")
                or data.get("predicate")
                or data.get("label")
                or "RELATED_TO"
            )
            desc = data.get("description") or data.get("desc") or ""
            weight = data.get("weight") or 1.0

            # Normalize relation_type to UPPER_SNAKE_CASE
            clean_rel = re.sub(r'[\s\-]+', '_', str(rel_type).strip()).upper()
            clean_rel = re.sub(r'[^A-Z0-9_]', '', clean_rel) or "RELATED_TO"

            data["source"] = normalize_node_name(str(src))
            data["target"] = normalize_node_name(str(tgt))
            data["relation_type"] = clean_rel
            data["description"] = str(desc).strip()
            try:
                data["weight"] = float(weight)
            except (ValueError, TypeError):
                data["weight"] = 1.0
        return data


class ExtractionResult(BaseModel):
    """Structured extraction output from LLM for a text chunk or document."""
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="List of extracted relations")

    @model_validator(mode="before")
    @classmethod
    def normalize_container(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check for relations under various aliases
            rels = (
                data.get("relations")
                or data.get("relationships")
                or data.get("edges")
                or data.get("links")
                or []
            )
            # Check for entities under various aliases
            ents = (
                data.get("entities")
                or data.get("nodes")
                or []
            )
            data["relations"] = rels
            data["entities"] = ents
        return data

    @property
    def relationships(self) -> List[Relation]:
        """Convenience alias for relations."""
        return self.relations


class ConceptNote(BaseModel):
    """Atomic Obsidian concept note representation with typed wikilinks."""
    title: str
    entity_type: str
    description: str
    wikilinks: List[str] = Field(default_factory=list, description="List of Obsidian wikilinks e.g. [[TargetEntity|RELATION_TYPE]]")
    source_documents: List[str] = Field(default_factory=list)
