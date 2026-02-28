import sqlite3
import re
from pathlib import Path

from src.rag.retrieval.hybrid import RetrievalHit


class DocumentStore:
    def __init__(self, db_url: str = "sqlite:///./data/rag.db") -> None:
        self.db_path = self._db_path_from_url(db_url)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _db_path_from_url(self, db_url: str) -> Path:
        if not db_url.startswith("sqlite:///"):
            raise ValueError("Only sqlite URLs are supported in this draft.")
        return Path(db_url.removeprefix("sqlite:///"))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    created_by TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page INTEGER,
                    url TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    title,
                    content,
                    tokenize = 'porter unicode61'
                );
                """
            )

    def save_document(self, document: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (doc_id, title, source_path, created_by)
                VALUES (?, ?, ?, ?)
                """,
                (
                    document["doc_id"],
                    document["title"],
                    document.get("source_path"),
                    document.get("created_by"),
                ),
            )

    def save_chunks(self, chunks: list[dict]) -> int:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, title, content, chunk_index, page, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["doc_id"],
                        chunk["title"],
                        chunk["content"],
                        chunk["chunk_index"],
                        chunk.get("page"),
                        chunk.get("url"),
                    )
                    for chunk in chunks
                ],
            )
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks_fts (rowid, chunk_id, doc_id, title, content)
                VALUES (
                    (SELECT rowid FROM chunks WHERE chunk_id = ?),
                    ?, ?, ?, ?
                )
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["chunk_id"],
                        chunk["doc_id"],
                        chunk["title"],
                        chunk["content"],
                    )
                    for chunk in chunks
                ],
            )
        return len(chunks)

    def get_chunk(self, chunk_id: str) -> RetrievalHit | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, doc_id, title, content, page, url
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            return None

        return RetrievalHit(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            title=row["title"],
            snippet=row["content"][:400],
            score=0.0,
            page=row["page"],
            url=row["url"],
        )

    def search_chunks(self, query: str, top_k: int) -> list[RetrievalHit]:
        normalized_query = self._normalize_fts_query(query)
        if not normalized_query:
            return []

        with self._connect() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT
                        chunks.chunk_id,
                        chunks.doc_id,
                        chunks.title,
                        chunks.content,
                        chunks.page,
                        chunks.url,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks ON chunks.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (normalized_query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                fallback_query = self._normalize_fts_query(query, strict=True)
                if not fallback_query:
                    return []
                rows = connection.execute(
                    """
                    SELECT
                        chunks.chunk_id,
                        chunks.doc_id,
                        chunks.title,
                        chunks.content,
                        chunks.page,
                        chunks.url,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks ON chunks.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (fallback_query, top_k),
                ).fetchall()

        return [
            RetrievalHit(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                snippet=row["content"][:400],
                score=1.0 / (1.0 + max(float(row["score"]), 0.0)),
                page=row["page"],
                url=row["url"],
            )
            for row in rows
        ]

    def _normalize_fts_query(self, query: str, strict: bool = False) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
        if strict:
            return " ".join(tokens)
        if not tokens:
            return ""
        return " OR ".join(tokens)

    def count_documents(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    documents.doc_id,
                    documents.title,
                    documents.source_path,
                    COUNT(chunks.chunk_id) AS chunk_count
                FROM documents
                LEFT JOIN chunks ON chunks.doc_id = documents.doc_id
                GROUP BY documents.doc_id, documents.title, documents.source_path
                ORDER BY documents.rowid DESC
                """
            ).fetchall()

        return [
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "source_path": row["source_path"],
                "chunk_count": int(row["chunk_count"]),
            }
            for row in rows
        ]

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_index
                """,
                (doc_id,),
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def delete_document(self, doc_id: str) -> dict[str, int]:
        chunk_ids = self.get_chunk_ids_for_document(doc_id)
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            deleted_chunks = connection.execute(
                "DELETE FROM chunks WHERE doc_id = ?",
                (doc_id,),
            ).rowcount
            deleted_documents = connection.execute(
                "DELETE FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).rowcount
        return {
            "deleted_documents": deleted_documents,
            "deleted_chunks": deleted_chunks,
            "chunk_ids": chunk_ids,
        }

    def delete_all_documents(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_id FROM chunks").fetchall()
            chunk_ids = [row["chunk_id"] for row in rows]
            deleted_chunks = connection.execute("DELETE FROM chunks").rowcount
            connection.execute("DELETE FROM chunks_fts")
            deleted_documents = connection.execute("DELETE FROM documents").rowcount
        return {
            "deleted_documents": deleted_documents,
            "deleted_chunks": deleted_chunks,
            "chunk_ids": chunk_ids,
        }


class BM25Store:
    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        return self.document_store.search_chunks(query, top_k)


class ACLStore:
    def is_allowed(self, user, doc_id: str) -> bool:
        return bool(user) and not doc_id.endswith("blocked")
