from dataclasses import dataclass

from src.rag.retrieval.query_preprocessor import build_retrieval_query


@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    score: float
    page: int | None = None
    url: str | None = None

    def key(self) -> str:
        return self.chunk_id

    def to_context_block(self) -> str:
        return f"[{self.doc_id}:{self.chunk_id}] {self.title}\n{self.snippet}"

    def debug(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "score": self.score,
            "page": self.page,
            "url": self.url,
        }


def _weighted_merge(
    dense: list[RetrievalHit],
    sparse: list[RetrievalHit],
    dense_weight: float,
    bm25_weight: float,
) -> list[RetrievalHit]:
    merged: dict[str, RetrievalHit] = {}
    score_map: dict[str, float] = {}

    for hit in dense:
        key = hit.key()
        merged[key] = hit
        score_map[key] = score_map.get(key, 0.0) + hit.score * dense_weight

    for hit in sparse:
        key = hit.key()
        if key not in merged:
            merged[key] = hit
        score_map[key] = score_map.get(key, 0.0) + hit.score * bm25_weight

    ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    return [merged[key] for key, _ in ordered]


def _rerank(query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    query_terms = set(query.lower().split())
    reranked = sorted(
        hits,
        key=lambda hit: (
            len(query_terms.intersection(hit.snippet.lower().split())),
            hit.score,
        ),
        reverse=True,
    )
    return reranked[:top_k]


def hybrid_retrieve(query, user, cfg, vector_store, bm25_store, acl_store):
    retrieval_query = build_retrieval_query(query, cfg=cfg)
    return hybrid_retrieve_with_query(
        retrieval_query=retrieval_query,
        user=user,
        cfg=cfg,
        vector_store=vector_store,
        bm25_store=bm25_store,
        acl_store=acl_store,
    )


def hybrid_retrieve_with_query(
    retrieval_query: str,
    user,
    cfg,
    vector_store,
    bm25_store,
    acl_store,
):
    if not retrieval_query:
        return []

    dense = vector_store.search(retrieval_query, top_k=cfg.retrieval.top_k)
    sparse = bm25_store.search(retrieval_query, top_k=cfg.retrieval.top_k)

    merged = _weighted_merge(
        dense,
        sparse,
        cfg.retrieval.dense_weight,
        cfg.retrieval.bm25_weight,
    )

    allowed = [hit for hit in merged if acl_store.is_allowed(user, hit.doc_id)]

    if cfg.retrieval.rerank.enabled:
        return _rerank(retrieval_query, allowed, top_k=cfg.retrieval.rerank.top_k)

    return allowed
