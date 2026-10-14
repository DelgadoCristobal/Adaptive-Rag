# Adaptive GraphRAG con Enrutamiento Dinámico, Semantic Cache y Auditoría Estadística

Repositorio de Trabajo de Fin de Máster (TFM) enfocado en arquitecturas RAG avanzadas mediante la integración de Grafos de Conocimiento (Neo4j + GDS), Recuperación Vectorial Densa (ChromaDB), Enrutamiento Adaptativo de Consultas, Caché Semántico y Auditoría Estadística Rigurosa.

---

## Arquitectura del Sistema

- **Ingestión:** Extracción layout-aware de PDFs a Markdown (`pymupdf4llm`) y construcción de grafo de entidades/relaciones en Neo4j.
- **Recuperación Híbrida & Enrutamiento:** Enrutador dinámico que clasifica consultas hacia:
 - Búsqueda Vectorial Densa (`VectorRetriever`)
 - Travesías Locales K-Hop en Grafo (`GraphRetriever`)
 - Resúmenes Jerárquicos de Comunidades GDS (`CommunityRetriever`)
 - Estrategia Híbrida Adaptativa
- **Caché Semántico:** Reducción de latencia y costes de inferencia mediante matching de similitud vectorial (`SemanticCache`).
- **Evaluación & Auditoría Estadística:** Cálculo de métricas RAGAS y validación formal de hipótesis (Wilcoxon, Paired t-test, Cohen's d) (`StatisticalAuditor`).
- **Inferencia Agnóstica:** Soporte multimodelo (Groq, Gemini, Ollama, OpenAI) mediante `LiteLLM` e `Instructor`.

---

## Estructura del Proyecto

```text
.
├── data/
│  ├── raw/        # Documentos de entrada (PDFs)
│  ├── markdown/      # Markdown extraído y procesado
│  └── processed/     # Datasets y resultados de evaluación
├── notebooks/       # Análisis exploratorio y experimentación estadística
├── src/
│  ├── ingestion/     # Parsers y construcción del grafo en Neo4j
│  ├── retrieval/     # Retrievers (Vector, Graph, Community) y Router
│  ├── cache/       # Semantic Cache
│  ├── evaluation/     # RAGAS metrics y auditoría estadística
│  └── utils/       # Cliente LLM unificado (LiteLLM)
├── tests/         # Tests unitarios e integración
├── docker-compose.yml   # Instancia Neo4j + APOC + GDS
├── requirements.txt    # Dependencias de Python fijadas
├── .env.example      # Plantilla de variables de entorno
└── README.md
```

---

## Arranque Rápido

### 1. Clonar e Instalar Dependencias

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_DIRECTORIO>
python -m venv .venv
# Activar entorno virtual:
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Copia el archivo `.env.example` a `.env` y rellena tus claves de API y credenciales de base de datos:

```bash
cp .env.example .env
```

> ⚠ **Importante:** Asegúrate de rellenar `GROQ_API_KEY`, `GEMINI_API_KEY` y verificar las credenciales de `NEO4J_*` en tu archivo `.env`.

### 3. Levantar Base de Datos Neo4j (con APOC & GDS)

Inicia el contenedor Docker con la base de datos de grafos:

```bash
docker compose up -d
```

- **Neo4j Browser:** [http://localhost:7474](http://localhost:7474)
- **Bolt Protocol:** `bolt://localhost:7687` (Usuario: `neo4j`, Contraseña: `password123`)

---

## Tests

Ejecutar la suite de pruebas:

```bash
pytest tests/
```
