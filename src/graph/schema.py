from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Extracted entity node schema."""
    id: str = Field(description="Normalized unique identifier or canonical name")
    name: str = Field(description="Original entity name as found in text")
    type: str = Field(description="Entity type/category (e.g. Person, Organization, Technology, Project, Concept)")
    description: str = Field(default="", description="Summary or description of the entity context")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Key-value dynamic attributes")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or acronyms")


class Relation(BaseModel):
    """Extracted relationship edge schema."""
    source: str = Field(description="ID/Name of the source entity")
    target: str = Field(description="ID/Name of the target entity")
    relation_type: str = Field(description="Relationship type (e.g., DEPENDS_ON, USES, PART_OF, MANAGED_BY, CREATED_BY)")
    description: str = Field(default="", description="Contextual explanation of the relationship")
    weight: float = Field(default=1.0, description="Confidence score or edge weight (0.0 to 1.0)")


class ExtractionResult(BaseModel):
    """Structured extraction output from LLM for a text chunk or document."""
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="List of extracted relations")


class ConceptNote(BaseModel):
    """Atomic Obsidian concept note representation with typed wikilinks."""
    title: str
    entity_type: str
    description: str
    wikilinks: List[str] = Field(default_factory=list, description="List of Obsidian wikilinks e.g. [[TargetEntity|RELATION_TYPE]]")
    source_documents: List[str] = Field(default_factory=list)
