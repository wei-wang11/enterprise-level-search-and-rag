# enterprise-level-search-and-rag

Enterprise search and RAG workspace with:

- FastAPI backend
- React + TypeScript frontend
- Qdrant for vector search
- SQLite FTS5 for BM25 keyword search
- hybrid retrieval, answer generation, and answer evaluation

## What It Does

The app lets you:

- upload local `.txt`, `.md`, and `.pdf` files
- index those files into SQLite + Qdrant
- retrieve relevant chunks with BM25 + vector search
- generate an answer from the retrieved evidence
- run an LLM-as-judge evaluation against:
  - the question
  - the answer
  - the retrieved evidence

## Architecture

- Backend: FastAPI in `src/`
- Frontend: Vite + React in `frontend/`
- Vector store: Qdrant
- Keyword search: SQLite FTS5 BM25
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- PDF extraction: Docling

## Project Layout

```text
.
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ Dockerfile
├─ docker-compose.yml
├─ frontend/
│  ├─ package.json
│  ├─ vite.config.ts
│  └─ src/
│     ├─ App.tsx
│     ├─ api.ts
│     ├─ main.tsx
│     └─ styles.css
├─ configs/
│  ├─ base.yaml
│  └─ usecases/
│     └─ contracts.yaml
├─ src/
│  ├─ app/
│  ├─ core/
│  ├─ indexing/
│  ├─ rag/
│  └─ storage/
└─ tests/
```

## Prerequisites

Local run:

- Python 3.11+
- Node.js 22+ recommended
- Docker Desktop if you want local Qdrant via Docker

Windows note:

- In PowerShell, `npm` may be blocked by execution policy. If that happens, use `npm.cmd` instead.

## Environment

Copy the example file first:

```powershell
Copy-Item .env.example .env
```

Current `.env` variables:

```env
APP_NAME=enterprise-level-search-and-rag
APP_ENV=local
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
API_PREFIX=
FRONTEND_ORIGIN=http://localhost:5173
OPENAI_API_KEY=
VECTOR_STORE_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_chunks
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
DOC_STORE_URL=sqlite:///./data/rag.db
```

Minimum setup to run locally:

- `VECTOR_STORE_URL=http://localhost:6333`
- `DOC_STORE_URL=sqlite:///./data/rag.db`
- `FRONTEND_ORIGIN=http://localhost:5173`

For real answer generation and evaluation:

- set `OPENAI_API_KEY`

For hosted Qdrant:

- set `VECTOR_STORE_URL`
- set `QDRANT_API_KEY`

## Run Locally

### 1. Create and use a Python environment

If you already have `.venv`, keep using it.

Install backend dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

This installs:

- FastAPI backend dependencies
- `fastembed` for embeddings
- `docling` for PDF extraction
- `openai` for answer generation and evaluation

### 2. Start Qdrant

If you do not already have a hosted Qdrant cluster, start the local one:

```powershell
docker compose up qdrant
```

Qdrant will be available at:

- `http://127.0.0.1:6333`

### 3. Start the backend

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend endpoints:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 4. Start the frontend

From the `frontend` directory:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Frontend URL:

- `http://127.0.0.1:5173`

The Vite dev server proxies API requests to the backend automatically.

## Run With Docker

This starts:

- backend on `8000`
- frontend on `5173`
- Qdrant on `6333`

```powershell
Copy-Item .env.example .env
docker compose up --build
```

URLs:

- Frontend: `http://127.0.0.1:5173`
- Backend docs: `http://127.0.0.1:8000/docs`
- Qdrant: `http://127.0.0.1:6333`

## First Run Flow

Once both frontend and backend are running:

1. Open `http://127.0.0.1:5173`
2. Upload one or more `.txt`, `.md`, or `.pdf` files
3. Confirm the files appear in the left indexed-files panel
4. Enter a question in the third column
5. Review retrieved evidence
6. Read the final answer
7. Optionally click `Evaluate` to run the judge model

## Current API Routes

Chat and retrieval:

- `POST /v1/retrieve`
- `POST /v1/generate`
- `POST /v1/chat`
- `POST /v1/evaluate`

Documents:

- `GET /v1/documents`
- `POST /v1/documents/upload`
- `DELETE /v1/documents/{doc_id}`
- `DELETE /v1/documents`

## Retrieval and Generation Behavior

Retrieval:

- SQLite FTS5 BM25 keyword search
- Qdrant vector search
- weighted hybrid merge
- reranking

Generation:

- the backend retrieves first
- top 6 retrieved chunks are passed into the final answer generation

Evaluation:

- uses the question
- uses the final answer
- uses the top retrieved evidence
- checks relevance, grounding, and hallucination risk

## PDF Support

PDF uploads are parsed with Docling before chunking and indexing.

If PDF upload fails, the most common cause is that backend dependencies were not installed after `docling` was added. Re-run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Common Problems

### `npm` is blocked in PowerShell

Use:

```powershell
npm.cmd install
npm.cmd run dev
```

### `{"detail":"Not Found"}` on backend root

Use:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

The frontend runs separately at `http://127.0.0.1:5173`.

### Upload works but answers do not

Most likely:

- `OPENAI_API_KEY` is missing
- your OpenAI project has no quota

Retrieval can still work without an LLM, but final answer generation and evaluation require a working key.

### PDF upload fails

Check:

- backend dependencies installed with `pip install -e .`
- the backend was restarted after installation

### Retrieval quality looks wrong after changing embedding model

Clear and rebuild the index. Old vectors are not compatible with a new embedding model.

### Qdrant connection errors

Check:

- `VECTOR_STORE_URL`
- `QDRANT_API_KEY` if using hosted Qdrant
- local Qdrant container is running if using Docker

## Notes

- SQLite is used locally for metadata and BM25 retrieval
- Qdrant is used for vector search
- the frontend is a separate app and talks to the backend over HTTP
- the left indexed-files panel supports deleting one file or clearing the full index

## Recommended Local Start Commands

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm.cmd run dev
```
