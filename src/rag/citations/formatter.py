from src.app.schemas.chat import Citation


def format_with_citations(answer: str, hits: list, cfg) -> tuple[str, list[Citation]]:
    if not cfg.citations.enabled:
        return answer, []

    citations = [
        Citation(
            doc_id=hit.doc_id,
            title=hit.title,
            snippet=hit.snippet,
            page=hit.page,
            url=hit.url,
        )
        for hit in hits[: cfg.citations.max_citations]
    ]
    return answer, citations
