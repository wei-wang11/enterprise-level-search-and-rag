export type IndexedDocument = {
  doc_id: string;
  title: string;
  source_path?: string | null;
  chunk_count: number;
};

export type Citation = {
  doc_id: string;
  title: string;
  snippet: string;
  page?: number | null;
  url?: string | null;
};

export type RetrievalHit = {
  chunk_id: string;
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
  page?: number | null;
  url?: string | null;
};

export type RetrieveResponse = {
  retrieval_query: string;
  hits: RetrievalHit[];
};

export type GenerateResponse = {
  answer: string;
  citations: Citation[];
};

export type EvaluationResponse = {
  verdict: string;
  reasoning: string;
};

export type UploadBatchResponse = {
  uploaded: Array<{
    doc_id: string;
    filename: string;
    saved_documents: number;
    saved_chunks: number;
    embedded_chunks: number;
  }>;
  failed: Array<{ filename: string; error: string }>;
  total_files: number;
  uploaded_count: number;
  failed_count: number;
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail ?? payload.error ?? "Request failed");
  }
  return payload as T;
}

export async function fetchDocuments(): Promise<IndexedDocument[]> {
  const response = await fetch(`${API_BASE_URL}/v1/documents`);
  const payload = await parseJson<{ documents: IndexedDocument[] }>(response);
  return payload.documents;
}

export async function uploadDocuments(files: File[]): Promise<UploadBatchResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const response = await fetch(`${API_BASE_URL}/v1/documents/upload`, {
    method: "POST",
    body: formData,
  });
  return parseJson<UploadBatchResponse>(response);
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/documents/${docId}`, {
    method: "DELETE",
  });
  await parseJson(response);
}

export async function deleteAllDocuments(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/documents`, {
    method: "DELETE",
  });
  await parseJson(response);
}

export async function retrieveDocuments(query: string, usecase: string): Promise<RetrieveResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, usecase }),
  });
  return parseJson<RetrieveResponse>(response);
}

export async function generateAnswer(
  query: string,
  usecase: string,
  retrievalQuery: string,
  hits: RetrievalHit[],
): Promise<GenerateResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, usecase, retrieval_query: retrievalQuery, hits }),
  });
  return parseJson<GenerateResponse>(response);
}

export async function evaluateAnswer(
  query: string,
  answer: string,
  usecase: string,
  retrievalQuery: string,
  hits: RetrievalHit[],
): Promise<EvaluationResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      answer,
      usecase,
      retrieval_query: retrievalQuery,
      hits,
    }),
  });
  return parseJson<EvaluationResponse>(response);
}
