import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.app.api.deps import get_store_registry, get_user
from src.app.schemas.documents import (
    DocumentDeleteAllResponse,
    DocumentDeleteResponse,
    IndexedDocumentListResponse,
    DocumentUploadBatchResponse,
    DocumentUploadError,
    DocumentUploadResponse,
)
from src.indexing.ingest import ingest_file


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.get("", response_model=IndexedDocumentListResponse)
def list_documents(stores=Depends(get_store_registry)) -> IndexedDocumentListResponse:
    return IndexedDocumentListResponse(documents=stores.doc.list_documents())


@router.delete("", response_model=DocumentDeleteAllResponse)
def delete_all_documents(stores=Depends(get_store_registry)) -> DocumentDeleteAllResponse:
    result = stores.doc.delete_all_documents()
    deleted_vectors = stores.vector.delete_chunks(result["chunk_ids"])
    logger.info(
        "documents.delete_all.complete deleted_documents=%d deleted_chunks=%d deleted_vectors=%d",
        result["deleted_documents"],
        result["deleted_chunks"],
        deleted_vectors,
    )
    return DocumentDeleteAllResponse(
        deleted_documents=result["deleted_documents"],
        deleted_chunks=result["deleted_chunks"],
        deleted_vectors=deleted_vectors,
    )


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
def delete_document(doc_id: str, stores=Depends(get_store_registry)) -> DocumentDeleteResponse:
    result = stores.doc.delete_document(doc_id)
    deleted_vectors = stores.vector.delete_chunks(result["chunk_ids"])
    logger.info(
        "documents.delete.complete doc_id=%s deleted_documents=%d deleted_chunks=%d deleted_vectors=%d",
        doc_id,
        result["deleted_documents"],
        result["deleted_chunks"],
        deleted_vectors,
    )
    return DocumentDeleteResponse(
        doc_id=doc_id,
        deleted_documents=result["deleted_documents"],
        deleted_chunks=result["deleted_chunks"],
        deleted_vectors=deleted_vectors,
    )


@router.post("/upload", response_model=DocumentUploadBatchResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    user=Depends(get_user),
    stores=Depends(get_store_registry),
) -> DocumentUploadBatchResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    uploaded: list[DocumentUploadResponse] = []
    failed: list[DocumentUploadError] = []
    logger.info(
        "documents.upload.start user_id=%s file_count=%d",
        user.get("user_id", "unknown"),
        len(files),
    )

    for file in files:
        filename = file.filename or "upload.txt"
        try:
            payload = await file.read()
            result = ingest_file(
                filename=filename,
                raw_bytes=payload,
                created_by=user["user_id"],
                doc_store=stores.doc,
                vector_store=stores.vector,
            )
            uploaded.append(DocumentUploadResponse(filename=filename, **result))
            logger.info(
                "documents.upload.file.success filename=%s chunks=%s embedded=%s",
                filename,
                result["saved_chunks"],
                result["embedded_chunks"],
            )
        except ValueError as exc:
            failed.append(DocumentUploadError(filename=filename, error=str(exc)))
            logger.warning("documents.upload.file.failed filename=%s error=%s", filename, exc)
        finally:
            await file.close()

    logger.info(
        "documents.upload.complete uploaded=%d failed=%d",
        len(uploaded),
        len(failed),
    )
    return DocumentUploadBatchResponse(
        uploaded=uploaded,
        failed=failed,
        total_files=len(files),
        uploaded_count=len(uploaded),
        failed_count=len(failed),
    )
