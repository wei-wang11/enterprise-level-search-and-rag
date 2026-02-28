import json
from dataclasses import dataclass
from urllib import error, request

from src.rag.embeddings.client import EmbeddingClient
from src.rag.retrieval.hybrid import RetrievalHit


class VectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "rag_chunks",
        api_key: str = "",
        dimension: int = 384,
        embedder: EmbeddingClient | None = None,
        document_store=None,
    ) -> None:
        self.url = url.rstrip("/")
        self.collection_name = collection_name
        self.api_key = api_key
        self.dimension = dimension
        self.embedder = embedder or EmbeddingClient()
        self.document_store = document_store
        self._ensure_collection()

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(f"{self.url}{path}", data=data, headers=headers, method=method)
        with request.urlopen(req, timeout=5) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _ensure_collection(self) -> None:
        try:
            self._request(
                "PUT",
                f"/collections/{self.collection_name}",
                {
                    "vectors": {
                        "size": self.dimension,
                        "distance": "Cosine",
                    }
                },
            )
        except error.URLError:
            pass

    def upsert(self, records: list[dict]) -> int:
        vectors = self.embedder.embed_documents([record["content"] for record in records])
        points = [
            {
                "id": record["chunk_id"],
                "vector": vector,
                "payload": {
                    "chunk_id": record["chunk_id"],
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "snippet": record["content"][:400],
                    "page": record.get("page"),
                    "url": record.get("url"),
                },
            }
            for record, vector in zip(records, vectors, strict=False)
        ]

        try:
            self._request(
                "PUT",
                f"/collections/{self.collection_name}/points?wait=true",
                {"points": points},
            )
            return len(records)
        except error.URLError:
            return 0

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        try:
            response = self._request(
                "POST",
                f"/collections/{self.collection_name}/points/search",
                {
                    "vector": self.embedder.embed_query(query),
                    "limit": top_k,
                    "with_payload": True,
                },
            )
        except error.URLError:
            return []

        hits = []
        for point in response.get("result", []):
            payload = point.get("payload", {})
            hit = None
            if self.document_store:
                hit = self.document_store.get_chunk(point["id"])
            if hit is None:
                hit = RetrievalHit(
                    chunk_id=payload.get("chunk_id", str(point["id"])),
                    doc_id=payload.get("doc_id", str(point["id"])),
                    title=payload.get("title", "Untitled"),
                    snippet=payload.get("snippet", ""),
                    score=float(point.get("score", 0.0)),
                    page=payload.get("page"),
                    url=payload.get("url"),
                )
            else:
                hit.score = float(point.get("score", 0.0))
            hits.append(hit)
        return hits

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        try:
            self._request(
                "POST",
                f"/collections/{self.collection_name}/points/delete?wait=true",
                {"points": chunk_ids},
            )
            return len(chunk_ids)
        except error.URLError:
            return 0


@dataclass
class StoreRegistry:
    vector: VectorStore
    bm25: object
    acl: object
    doc: object
