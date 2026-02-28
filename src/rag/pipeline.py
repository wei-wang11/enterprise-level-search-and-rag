import logging
import time
from dataclasses import dataclass

from src.rag.citations.formatter import format_with_citations
from src.rag.llm.client import LLMClient
from src.rag.llm.guardrails import apply_guardrails
from src.rag.retrieval.hybrid import RetrievalHit, hybrid_retrieve_with_query
from src.rag.retrieval.query_preprocessor import build_retrieval_query


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    retrieval_query: str
    hits: list[RetrievalHit]
    debug: dict


@dataclass
class RagResult:
    answer: str
    citations: list
    debug: dict


class RagPipeline:
    TOP_CONTEXT_BLOCKS = 6

    def __init__(self, cfg, stores, llm: LLMClient, prompt_loader):
        self.cfg = cfg
        self.stores = stores
        self.llm = llm
        self.prompt_loader = prompt_loader

    def retrieve(self, user, query: str, usecase: str) -> RetrievalResult:
        uc_cfg = self.cfg.for_usecase(usecase)
        logger.info(
            "rag.request.start usecase=%s user_id=%s query=%r",
            usecase,
            user.get("user_id", "unknown"),
            query,
        )
        retrieval_query = build_retrieval_query(query, cfg=uc_cfg)
        logger.info("rag.request.query_preprocessed retrieval_query=%r", retrieval_query)

        retrieval_started_at = time.perf_counter()
        hits = hybrid_retrieve_with_query(
            retrieval_query=retrieval_query,
            user=user,
            cfg=uc_cfg,
            vector_store=self.stores.vector,
            bm25_store=self.stores.bm25,
            acl_store=self.stores.acl,
        )
        logger.info(
            "rag.request.retrieved usecase=%s hits=%d elapsed_ms=%d",
            usecase,
            len(hits),
            int((time.perf_counter() - retrieval_started_at) * 1000),
        )

        return RetrievalResult(
            retrieval_query=retrieval_query,
            hits=hits,
            debug={
                "retrieval_query": retrieval_query,
                "retrieved": [hit.debug() for hit in hits],
            },
        )

    def generate(
        self,
        query: str,
        usecase: str,
        hits: list[RetrievalHit],
        retrieval_query: str | None = None,
    ) -> RagResult:
        started_at = time.perf_counter()
        uc_cfg = self.cfg.for_usecase(usecase)
        system_prompt = self.prompt_loader.load(uc_cfg.usecase.system_prompt)

        llm_hits = hits[: self.TOP_CONTEXT_BLOCKS]
        context_blocks = [hit.to_context_block() for hit in llm_hits]
        logger.info(
            "rag.request.context_selected total_hits=%d llm_context_blocks=%d",
            len(hits),
            len(context_blocks),
        )

        generation_started_at = time.perf_counter()
        raw = self.llm.generate(
            system=system_prompt,
            user=query,
            context=context_blocks,
            temperature=self.cfg.llm.temperature,
            model=self.cfg.llm.model,
            max_output_tokens=self.cfg.llm.max_output_tokens,
        )
        logger.info(
            "rag.request.generated usecase=%s elapsed_ms=%d",
            usecase,
            int((time.perf_counter() - generation_started_at) * 1000),
        )

        guarded = apply_guardrails(raw, hits, uc_cfg)
        final_text, citations = format_with_citations(guarded, hits, uc_cfg)
        logger.info(
            "rag.request.complete usecase=%s citations=%d total_elapsed_ms=%d",
            usecase,
            len(citations),
            int((time.perf_counter() - started_at) * 1000),
        )

        return RagResult(
            answer=final_text,
            citations=citations,
            debug={
                "retrieval_query": retrieval_query,
                "retrieved": [hit.debug() for hit in hits],
                "llm_context": [hit.debug() for hit in llm_hits],
            },
        )

    def run(self, user, query: str, usecase: str) -> RagResult:
        retrieval = self.retrieve(user=user, query=query, usecase=usecase)
        return self.generate(
            query=query,
            usecase=usecase,
            hits=retrieval.hits,
            retrieval_query=retrieval.retrieval_query,
        )

    def evaluate(
        self,
        query: str,
        answer: str,
        usecase: str,
        retrieval_query: str | None = None,
        hits: list[RetrievalHit] | None = None,
    ) -> dict[str, str]:
        judge_hits = hits
        judge_query = retrieval_query
        if judge_hits is None:
            retrieval = self.retrieve(user={"user_id": "judge"}, query=query, usecase=usecase)
            judge_hits = retrieval.hits
            judge_query = retrieval.retrieval_query

        return self.llm.evaluate_answer(
            query=query,
            answer=answer,
            retrieval_query=judge_query,
            hits=[
                hit.debug() | {"snippet": hit.snippet}
                for hit in (judge_hits or [])[: self.TOP_CONTEXT_BLOCKS]
            ],
            model=self.cfg.llm.model,
            max_output_tokens=min(self.cfg.llm.max_output_tokens, 200),
        )
