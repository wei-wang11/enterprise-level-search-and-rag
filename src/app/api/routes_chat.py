from fastapi import APIRouter, Depends

from src.app.api.deps import get_rag_service, get_user
from src.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResponse,
    GenerateRequest,
    RetrieveResponse,
    RetrievalHitResponse,
)
from src.app.services.rag_service import RagService
from src.rag.retrieval.hybrid import RetrievalHit


router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user=Depends(get_user),
    rag: RagService = Depends(get_rag_service),
) -> ChatResponse:
    result = rag.answer(user=user, query=req.query, usecase=req.usecase)
    return ChatResponse(answer=result.answer, citations=result.citations)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: ChatRequest,
    user=Depends(get_user),
    rag: RagService = Depends(get_rag_service),
) -> RetrieveResponse:
    result = rag.retrieve(user=user, query=req.query, usecase=req.usecase)
    return RetrieveResponse(
        retrieval_query=result.retrieval_query,
        hits=[
            RetrievalHitResponse(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                title=hit.title,
                snippet=hit.snippet,
                score=hit.score,
                page=hit.page,
                url=hit.url,
            )
            for hit in result.hits
        ],
    )


@router.post("/generate", response_model=ChatResponse)
def generate(
    req: GenerateRequest,
    rag: RagService = Depends(get_rag_service),
) -> ChatResponse:
    hits = [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            title=hit.title,
            snippet=hit.snippet,
            score=hit.score,
            page=hit.page,
            url=hit.url,
        )
        for hit in req.hits
    ]
    result = rag.generate(
        query=req.query,
        usecase=req.usecase,
        hits=hits,
        retrieval_query=req.retrieval_query,
    )
    return ChatResponse(answer=result.answer, citations=result.citations)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    req: EvaluationRequest,
    rag: RagService = Depends(get_rag_service),
) -> EvaluationResponse:
    hits = [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            title=hit.title,
            snippet=hit.snippet,
            score=hit.score,
            page=hit.page,
            url=hit.url,
        )
        for hit in req.hits
    ]
    result = rag.evaluate(
        query=req.query,
        answer=req.answer,
        usecase=req.usecase,
        retrieval_query=req.retrieval_query,
        hits=hits,
    )
    return EvaluationResponse(**result)
