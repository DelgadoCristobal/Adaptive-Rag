from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from src.config import settings
from .schema import Entity, Relation, ExtractionResult

logger = logging.getLogger(__name__)


class KuzuGraphStore:
    """Manager for KùzuDB embedded C++ property graph storage and Cypher query execution."""

    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else settings.kuzu_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize KùzuDB connection and create property graph schema if not existing."""
        try:
            import kuzu
            self.db = kuzu.Database(str(self.db_path))
            self.conn = kuzu.Connection(self.db)
            self._create_schema()
        except Exception as e:
            logger.error(f"Failed to initialize KùzuDB at {self.db_path}: {e}")

    def _create_schema(self):
        """Execute DDL statements to create Node and Rel tables."""
        if not self.conn:
            return

        # Entity Node table DDL
        try:
            self.conn.execute(
                "CREATE NODE TABLE EntityNode(id STRING, name STRING, type STRING, description STRING, PRIMARY KEY(id))"
            )
            logger.info("Created KùzuDB table EntityNode")
        except Exception:
            pass  # Table likely exists

        # General Rel table DDL
        try:
            self.conn.execute(
                "CREATE REL TABLE RELATION(FROM EntityNode TO EntityNode, relation_type STRING, description STRING, weight DOUBLE)"
            )
            logger.info("Created KùzuDB table RELATION")
        except Exception:
            pass  # Table likely exists

    def add_extraction_result(self, extraction: ExtractionResult):
        """Bulk insert entities and relations into KùzuDB."""
        if not self.conn:
            return

        # Insert Node Entities
        for entity in extraction.entities:
            query = """
            MERGE (e:EntityNode {id: $id})
            ON CREATE SET e.name = $name, e.type = $type, e.description = $description
            ON MATCH SET e.name = $name, e.type = $type, e.description = $description
            """
            try:
                self.conn.execute(
                    query,
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "description": entity.description or ""
                    }
                )
            except Exception as e:
                # Fallback to simple insert if MERGE syntax varies by kuzu version
                try:
                    self.conn.execute(
                        "CREATE (e:EntityNode {id: $id, name: $name, type: $type, description: $description})",
                        {"id": entity.id, "name": entity.name, "type": entity.type, "description": entity.description or ""}
                    )
                except Exception:
                    pass

        # Insert Relations
        for rel in extraction.relations:
            query = """
            MATCH (a:EntityNode {id: $source}), (b:EntityNode {id: $target})
            CREATE (a)-[r:RELATION {relation_type: $rel_type, description: $desc, weight: $weight}]->(b)
            """
            try:
                self.conn.execute(
                    query,
                    {
                        "source": rel.source,
                        "target": rel.target,
                        "rel_type": rel.relation_type,
                        "desc": rel.description or "",
                        "weight": float(rel.weight)
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to insert relation {rel.source}->{rel.target}: {e}")

    def query_k_hop_neighborhood(self, entity_id: str, hops: int = 2) -> List[Dict[str, Any]]:
        """Traverse graph up to K-hops from target entity_id and return connected paths."""
        if not self.conn:
            return []

        paths = []
        cypher = f"""
        MATCH (a:EntityNode {{id: $id}})-[r:RELATION*1..{hops}]->(b:EntityNode)
        RETURN a.id AS source, b.id AS target, b.name AS target_name, b.type AS target_type, b.description AS target_desc
        """
        try:
            response = self.conn.execute(cypher, {"id": entity_id})
            while response.has_next():
                row = response.get_next()
                paths.append({
                    "source": row[0],
                    "target": row[1],
                    "target_name": row[2],
                    "target_type": row[3],
                    "target_desc": row[4]
                })
        except Exception as e:
            logger.warning(f"K-hop query failed for {entity_id}: {e}")

        return paths

    def search_entities_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search entities by keyword matching name or description."""
        if not self.conn:
            return []

        results = []
        cypher = "MATCH (e:EntityNode) WHERE e.name CONTAINS $kw OR e.id CONTAINS $kw OR e.description CONTAINS $kw RETURN e.id, e.name, e.type, e.description"
        try:
            response = self.conn.execute(cypher, {"kw": keyword})
            while response.has_next():
                row = response.get_next()
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3]
                })
        except Exception as e:
            logger.warning(f"Keyword search failed for '{keyword}': {e}")

        return results

    def get_stats(self) -> Dict[str, int]:
        """Return total counts of nodes and edges in KùzuDB."""
        if not self.conn:
            return {"node_count": 0, "rel_count": 0}

        node_count, rel_count = 0, 0
        try:
            r1 = self.conn.execute("MATCH (n:EntityNode) RETURN count(n) AS cnt")
            if r1.has_next():
                node_count = r1.get_next()[0]
            r2 = self.conn.execute("MATCH ()-[r:RELATION]->() RETURN count(r) AS cnt")
            if r2.has_next():
                rel_count = r2.get_next()[0]
        except Exception as e:
            logger.warning(f"Failed to fetch stats: {e}")

        return {"node_count": node_count, "rel_count": rel_count}
