from functools import lru_cache

from src.app.services.rag_service import RagService
from src.core.config import AppConfig, load_app_config
from src.core.settings import get_settings
from src.rag.embeddings.client import EmbeddingClient
from src.rag.llm.client import LLMClient
from src.rag.prompt_loader import PromptLoader
from src.storage.doc_store import ACLStore, BM25Store, DocumentStore
from src.storage.vector_store import StoreRegistry, VectorStore


def get_user() -> dict[str, str]:
    return {"user_id": "demo-user", "role": "analyst"}


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_app_config()


@lru_cache(maxsize=1)
def get_prompt_loader() -> PromptLoader:
    return PromptLoader()


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return LLMClient(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    settings = get_settings()
    return EmbeddingClient(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_store_registry() -> StoreRegistry:
    settings = get_settings()
    doc_store = DocumentStore(db_url=settings.doc_store_url)
    return StoreRegistry(
        vector=VectorStore(
            url=settings.vector_store_url,
            collection_name=settings.qdrant_collection,
            api_key=settings.qdrant_api_key,
            dimension=settings.embedding_dimension,
            embedder=get_embedding_client(),
            document_store=doc_store,
        ),
        bm25=BM25Store(doc_store),
        acl=ACLStore(),
        doc=doc_store,
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(
        cfg=get_config(),
        stores=get_store_registry(),
        llm=get_llm_client(),
        prompt_loader=get_prompt_loader(),
    )
