from pathlib import Path
from typing import List, Dict, Any
import logging
import math
import re
from src.config import settings
from .schema import ExtractionResult

logger = logging.getLogger(__name__)

MIN_HOPS, MAX_HOPS = 1, 5

# On match, a node keeps its first real name/type (endpoint placeholders use name = id and type CONCEPT)
# and only takes a new description when it is longer than the stored one.
NODE_UPSERT_CYPHER = """
MERGE (e:EntityNode {id: $id})
ON CREATE SET e.name = $name, e.type = $type, e.description = $description
ON MATCH SET
    e.name = CASE WHEN e.name IS NULL OR e.name = '' OR (e.name = e.id AND $name <> '') THEN $name ELSE e.name END,
    e.type = CASE WHEN e.type IS NULL OR e.type = '' OR (e.type = 'CONCEPT' AND $type <> '') THEN $type ELSE e.type END,
    e.description = CASE WHEN size($description) > size(coalesce(e.description, '')) THEN $description ELSE e.description END
"""

# One parameterized MERGE per (source, target, relation_type): duplicates are deduplicated by Kuzu and the
# first description/weight wins. Values are never interpolated into the query text.
RELATION_MERGE_CYPHER = """
MATCH (a:EntityNode), (b:EntityNode)
WHERE a.id = $source AND b.id = $target
MERGE (a)-[r:RELATION {relation_type: $rel_type}]->(b)
ON CREATE SET r.description = $description, r.weight = $weight
"""


def _clamp_hops(hops: Any) -> int:
    try:
        value = int(hops)
    except (TypeError, ValueError):
        value = MIN_HOPS
    return max(MIN_HOPS, min(MAX_HOPS, value))


def _edge_weight(weight: Any) -> float:
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) else 1.0


class KuzuGraphStore:
    """Manager for KùzuDB embedded C++ property graph storage and Cypher query execution."""
    _instances: Dict[str, Any] = {}

    def __new__(cls, db_path: Path = None):
        target_path = str(Path(db_path) if db_path else settings.kuzu_db_path)
        if target_path not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[target_path] = instance
        return cls._instances[target_path]

    def __init__(self, db_path: Path = None):
        if getattr(self, "_initialized", False):
            return
        self.db_path = Path(db_path) if db_path else settings.kuzu_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.conn = None
        self._init_db()
        self._initialized = True

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
        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Failed to create KùzuDB table EntityNode: {e}")

        # General Rel table DDL
        try:
            self.conn.execute(
                "CREATE REL TABLE RELATION(FROM EntityNode TO EntityNode, relation_type STRING, description STRING, weight DOUBLE)"
            )
            logger.info("Created KùzuDB table RELATION")
        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Failed to create KùzuDB table RELATION: {e}")

    def add_extraction_result(self, extraction: ExtractionResult):
        """Bulk insert entities and relations into KùzuDB."""
        if not self.conn:
            return

        # 1. Insert/Merge Node Entities (non-destructive on match)
        for entity in extraction.entities:
            ent_id = str(entity.id).strip()
            if not ent_id:
                continue
            params = {
                "id": ent_id,
                "name": str(entity.name).strip() or ent_id,
                "type": str(entity.type).strip().upper() or "CONCEPT",
                "description": str(entity.description or "").strip(),
            }
            try:
                self.conn.execute(NODE_UPSERT_CYPHER, params)
            except Exception as e:
                logger.warning(f"Failed to upsert entity {ent_id}: {e}")

        # 2. Insert Relations
        for rel in extraction.relations:
            src_id = str(rel.source).strip()
            tgt_id = str(rel.target).strip()
            if not src_id or not tgt_id or src_id == tgt_id:
                continue

            # Ensure endpoint nodes exist without touching existing ones
            for node_id in [src_id, tgt_id]:
                try:
                    self.conn.execute(
                        "MERGE (e:EntityNode {id: $id}) ON CREATE SET e.name = $id, e.type = 'CONCEPT', e.description = ''",
                        {"id": node_id}
                    )
                except Exception as e:
                    logger.warning(f"Failed to ensure endpoint node {node_id}: {e}")

            clean_rel_type = (rel.relation_type or "RELATED_TO").strip().upper()
            clean_rel_type = re.sub(r'[^A-Z0-9_]', '_', clean_rel_type) or "RELATED_TO"
            params = {
                "source": src_id,
                "target": tgt_id,
                "rel_type": clean_rel_type,
                "description": str(rel.description or ""),
                "weight": _edge_weight(rel.weight),
            }
            try:
                self.conn.execute(RELATION_MERGE_CYPHER, params)
            except Exception as e:
                logger.warning(f"Failed to insert relation {src_id}-[{clean_rel_type}]->{tgt_id}: {e}")

    def query_k_hop_neighborhood(self, entity_id: str, hops: int = 2) -> List[Dict[str, Any]]:
        """Traverse graph up to K-hops from target entity_id and return connected paths."""
        if not self.conn:
            return []

        paths = []
        hops = _clamp_hops(hops)  # int-cast and clamped: the only value interpolated into the query
        cypher = f"""
        MATCH (a:EntityNode)-[r:RELATION*1..{hops}]->(b:EntityNode)
        WHERE a.id = $id
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

    def search_entities_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search entities matching any of the provided keywords in a single Cypher query."""
        if not self.conn or not keywords:
            return []

        cleaned_kws = [kw.strip() for kw in keywords if kw.strip()]
        if not cleaned_kws:
            return []

        conditions = []
        params = {}
        for i, kw in enumerate(cleaned_kws):
            param_name = f"kw{i}"
            conditions.append(f"(e.name CONTAINS ${param_name} OR e.id CONTAINS ${param_name} OR e.description CONTAINS ${param_name})")
            params[param_name] = kw

        results = []
        cypher = f"MATCH (e:EntityNode) WHERE {' OR '.join(conditions)} RETURN e.id, e.name, e.type, e.description"
        try:
            response = self.conn.execute(cypher, params)
            while response.has_next():
                row = response.get_next()
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3]
                })
        except Exception as e:
            logger.warning(f"Batch keyword search failed: {e}")

        return results

    def search_entities_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search entities by keyword matching name or description."""
        return self.search_entities_by_keywords([keyword]) if keyword else []

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
