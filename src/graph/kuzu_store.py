from pathlib import Path
from typing import List, Dict, Any
import logging
from src.config import settings
from .schema import ExtractionResult

logger = logging.getLogger(__name__)


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

        import re

        # 1. Insert/Merge Node Entities
        for entity in extraction.entities:
            ent_id = str(entity.id).strip()
            ent_name = str(entity.name).strip() or ent_id
            ent_type = str(entity.type).strip().upper() or "CONCEPT"
            ent_desc = str(entity.description or "").strip()

            query = """
            MERGE (e:EntityNode {id: $id})
            ON CREATE SET e.name = $name, e.type = $type, e.description = $description
            ON MATCH SET e.name = $name, e.type = $type, e.description = $description
            """
            try:
                self.conn.execute(
                    query,
                    {
                        "id": ent_id,
                        "name": ent_name,
                        "type": ent_type,
                        "description": ent_desc
                    }
                )
            except Exception:
                try:
                    self.conn.execute(
                        "CREATE (e:EntityNode {id: $id, name: $name, type: $type, description: $description})",
                        {"id": ent_id, "name": ent_name, "type": ent_type, "description": ent_desc}
                    )
                except Exception:
                    pass

        # 2. Insert Relations
        for rel in extraction.relations:
            src_id = str(rel.source).strip()
            tgt_id = str(rel.target).strip()
            if not src_id or not tgt_id or src_id == tgt_id:
                continue

            # Ensure endpoint nodes exist
            for node_id in [src_id, tgt_id]:
                try:
                    self.conn.execute(
                        "MERGE (e:EntityNode {id: $id}) ON CREATE SET e.name = $id, e.type = 'CONCEPT', e.description = ''",
                        {"id": node_id}
                    )
                except Exception:
                    pass

            clean_rel_type = (rel.relation_type or "RELATED_TO").strip().upper()
            clean_rel_type = re.sub(r'[^A-Z0-9_]', '_', clean_rel_type) or "RELATED_TO"
            clean_desc = (rel.description or "").replace("'", "\\'").replace('"', '\\"')
            weight = float(rel.weight) if rel.weight else 1.0

            # Prevent duplicate identical edges
            try:
                chk = self.conn.execute(
                    "MATCH (a:EntityNode)-[r:RELATION]->(b:EntityNode) WHERE a.id = $source AND b.id = $target AND r.relation_type = $rel_type RETURN count(r)",
                    {"source": src_id, "target": tgt_id, "rel_type": clean_rel_type}
                )
                if chk.has_next() and chk.get_next()[0] > 0:
                    continue
            except Exception:
                pass

            # In KuzuDB, relation properties are specified as inline literals in CREATE pattern
            insert_cypher = f"""
            MATCH (a:EntityNode), (b:EntityNode)
            WHERE a.id = $source AND b.id = $target
            CREATE (a)-[r:RELATION {{relation_type: '{clean_rel_type}', description: '{clean_desc}', weight: {weight}}}]->(b)
            """
            try:
                self.conn.execute(insert_cypher, {"source": src_id, "target": tgt_id})
            except Exception as e:
                logger.debug(f"Failed to insert relation {src_id}->{tgt_id}: {e}")

    def query_k_hop_neighborhood(self, entity_id: str, hops: int = 2) -> List[Dict[str, Any]]:
        """Traverse graph up to K-hops from target entity_id and return connected paths."""
        if not self.conn:
            return []

        paths = []
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
