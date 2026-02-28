from pydantic import BaseModel, Field


class IndexedDocument(BaseModel):
    doc_id: str
    title: str
    source_path: str | None = None
    chunk_count: int


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    saved_documents: int
    saved_chunks: int
    embedded_chunks: int


class DocumentUploadError(BaseModel):
    filename: str
    error: str


class DocumentUploadBatchResponse(BaseModel):
    uploaded: list[DocumentUploadResponse] = Field(default_factory=list)
    failed: list[DocumentUploadError] = Field(default_factory=list)
    total_files: int
    uploaded_count: int
    failed_count: int


class IndexedDocumentListResponse(BaseModel):
    documents: list[IndexedDocument] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    deleted_documents: int
    deleted_chunks: int
    deleted_vectors: int


class DocumentDeleteAllResponse(BaseModel):
    deleted_documents: int
    deleted_chunks: int
    deleted_vectors: int
