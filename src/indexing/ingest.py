import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from src.storage.doc_store import DocumentStore
from src.storage.vector_store import VectorStore


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _extract_pdf_with_docling(raw_bytes: bytes) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ValueError(
            "PDF upload requires the optional 'docling' dependency to be installed."
        ) from exc

    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(raw_bytes)
            temp_path = temp_file.name

        result = DocumentConverter().convert(temp_path)
        content = result.document.export_to_markdown().strip()
        if not content:
            raise ValueError("Docling extracted no text from this PDF.")
        return content
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to extract PDF with Docling: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def ingest_document(
    *,
    title: str,
    content: str,
    source_path: str,
    created_by: str,
    doc_store: DocumentStore,
    vector_store: VectorStore,
) -> dict[str, int | str]:
    doc_id = str(uuid4())
    doc_store.save_document(
        {
            "doc_id": doc_id,
            "title": title,
            "source_path": source_path,
            "created_by": created_by,
        }
    )

    chunks = _chunk_text(content)
    chunk_records = [
        {
            "chunk_id": str(uuid4()),
            "doc_id": doc_id,
            "title": title,
            "content": chunk,
            "chunk_index": index,
            "page": None,
            "url": None,
        }
        for index, chunk in enumerate(chunks)
    ]

    saved_chunks = doc_store.save_chunks(chunk_records)
    embedded_chunks = vector_store.upsert(chunk_records)

    return {
        "doc_id": doc_id,
        "saved_documents": 1,
        "saved_chunks": saved_chunks,
        "embedded_chunks": embedded_chunks,
    }


def ingest_file(
    *,
    filename: str,
    raw_bytes: bytes,
    created_by: str,
    doc_store: DocumentStore,
    vector_store: VectorStore,
) -> dict[str, int | str]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        content = _extract_pdf_with_docling(raw_bytes)
    elif suffix in {".txt", ".md"}:
        content = raw_bytes.decode("utf-8")
    else:
        raise ValueError("Only .txt, .md, and .pdf files are supported.")

    return ingest_document(
        title=Path(filename).stem,
        content=content,
        source_path=filename,
        created_by=created_by,
        doc_store=doc_store,
        vector_store=vector_store,
    )
