from src.rag.pipeline import RagPipeline, RagResult, RetrievalResult
from src.rag.retrieval.hybrid import RetrievalHit


class RagService:
    def __init__(self, cfg, stores, llm, prompt_loader) -> None:
        self.pipeline = RagPipeline(
            cfg=cfg,
            stores=stores,
            llm=llm,
            prompt_loader=prompt_loader,
        )

    def answer(self, user, query: str, usecase: str) -> RagResult:
        return self.pipeline.run(user=user, query=query, usecase=usecase)

    def retrieve(self, user, query: str, usecase: str) -> RetrievalResult:
        return self.pipeline.retrieve(user=user, query=query, usecase=usecase)

    def generate(
        self,
        query: str,
        usecase: str,
        hits: list[RetrievalHit],
        retrieval_query: str | None = None,
    ) -> RagResult:
        return self.pipeline.generate(
            query=query,
            usecase=usecase,
            hits=hits,
            retrieval_query=retrieval_query,
        )

    def evaluate(
        self,
        query: str,
        answer: str,
        usecase: str,
        retrieval_query: str | None = None,
        hits: list[RetrievalHit] | None = None,
    ) -> dict[str, str]:
        return self.pipeline.evaluate(
            query=query,
            answer=answer,
            usecase=usecase,
            retrieval_query=retrieval_query,
            hits=hits,
        )
