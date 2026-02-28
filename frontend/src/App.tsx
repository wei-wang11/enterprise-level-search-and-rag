import { FormEvent, useEffect, useState } from "react";
import {
  deleteAllDocuments,
  deleteDocument,
  evaluateAnswer,
  fetchDocuments,
  generateAnswer,
  type IndexedDocument,
  type RetrievalHit,
  retrieveDocuments,
  uploadDocuments,
} from "./api";

const usecases = ["default", "contracts"];
const DEFAULT_ANSWER = "Ask a question to generate an answer.";
const DEFAULT_UPLOAD_SUMMARY = "No uploads yet.";

export default function App() {
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState("");
  const [usecase, setUsecase] = useState("default");
  const [answer, setAnswer] = useState(DEFAULT_ANSWER);
  const [retrievals, setRetrievals] = useState<RetrievalHit[]>([]);
  const [indexedStatus, setIndexedStatus] = useState("Loading indexed files...");
  const [uploadStatus, setUploadStatus] = useState("No upload in progress.");
  const [chatStatus, setChatStatus] = useState("Ready.");
  const [uploadSummary, setUploadSummary] = useState(DEFAULT_UPLOAD_SUMMARY);
  const [retrievalQuery, setRetrievalQuery] = useState("");
  const [evaluation, setEvaluation] = useState("");
  const [evaluationStatus, setEvaluationStatus] = useState("No evaluation yet.");

  function resetQueryState(nextAnswer: string = DEFAULT_ANSWER) {
    setRetrievals([]);
    setRetrievalQuery("");
    setAnswer(nextAnswer);
    setEvaluation("");
    setEvaluationStatus("No evaluation yet.");
  }

  async function refreshDocuments() {
    setIndexedStatus("Loading indexed files...");
    try {
      const next = await fetchDocuments();
      setDocuments(next);
      setIndexedStatus(next.length ? `${next.length} indexed file(s)` : "No indexed files yet.");
    } catch (error) {
      setIndexedStatus(error instanceof Error ? error.message : "Failed to load indexed files.");
    }
  }

  useEffect(() => {
    void refreshDocuments();
  }, []);

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!files.length) {
      setUploadStatus("Choose at least one local file.");
      return;
    }

    setUploadStatus("Uploading files...");
    try {
      const result = await uploadDocuments(files);
      setUploadStatus("Upload completed.");
      setUploadSummary(
        `${result.uploaded_count} uploaded, ${result.failed_count} failed. ${
          result.uploaded.map((item) => item.filename).join(", ") || "No files processed."
        }`,
      );
      setFiles([]);
      await refreshDocuments();
    } catch (error) {
      setUploadStatus(error instanceof Error ? error.message : "Upload failed.");
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) {
      setChatStatus("Enter a question first.");
      return;
    }

    resetQueryState("Waiting for retrieval.");
    setChatStatus("Retrieving evidence...");
    try {
      const retrieval = await retrieveDocuments(query, usecase);
      setRetrievals(retrieval.hits);
      setRetrievalQuery(retrieval.retrieval_query);
      setChatStatus(
        `Retrieved ${retrieval.hits.length} document block(s). Generating answer...`,
      );

      const response = await generateAnswer(
        query,
        usecase,
        retrieval.retrieval_query,
        retrieval.hits,
      );
      setAnswer(response.answer || "No answer returned.");
      setChatStatus("Query completed.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Query failed.";
      resetQueryState("Request failed.");
      setChatStatus(message);
    }
  }

  async function handleDelete(docId: string) {
    setIndexedStatus("Deleting indexed file...");
    try {
      await deleteDocument(docId);
      await refreshDocuments();
      setIndexedStatus("Indexed file deleted.");
    } catch (error) {
      setIndexedStatus(error instanceof Error ? error.message : "Delete failed.");
    }
  }

  async function handleDeleteAll() {
    setIndexedStatus("Deleting all indexed files...");
    try {
      await deleteAllDocuments();
      resetQueryState();
      await refreshDocuments();
      setIndexedStatus("All indexed files deleted.");
    } catch (error) {
      setIndexedStatus(error instanceof Error ? error.message : "Delete all failed.");
    }
  }

  async function handleEvaluate() {
    if (!query.trim() || !answer.trim() || answer === DEFAULT_ANSWER || answer === "Request failed.") {
      setEvaluationStatus("Generate an answer first.");
      return;
    }

    setEvaluationStatus("Evaluating answer...");
    try {
      const result = await evaluateAnswer(
        query,
        answer,
        usecase,
        retrievalQuery,
        retrievals.slice(0, 6),
      );
      setEvaluation(`${result.verdict.toUpperCase()}: ${result.reasoning}`);
      setEvaluationStatus("Evaluation completed.");
    } catch (error) {
      setEvaluation("");
      setEvaluationStatus(error instanceof Error ? error.message : "Evaluation failed.");
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="hero__copy">
          <p className="hero__kicker">Enterprise Search and RAG</p>
          <h1>Evidence First Workspace</h1>
          <p className="hero__lead">
            Review indexed files, ingest local documents, inspect retrieved context, and read the final answer in a single workspace.
          </p>
        </div>
        <div className="hero__meta">
          <div className="hero__badge hero__badge--soft">Hybrid BM25 + Vector</div>
        </div>
      </header>

      <section className="stats-strip">
        <article className="stat-card">
          <span className="stat-card__label">Indexed Files</span>
          <strong className="stat-card__value">{documents.length}</strong>
          <span className="stat-card__meta">Live in SQLite and Qdrant</span>
        </article>
        <article className="stat-card">
          <span className="stat-card__label">Retrieval Hits</span>
          <strong className="stat-card__value">{retrievals.length}</strong>
          <span className="stat-card__meta">Evidence cards in the third column</span>
        </article>
        <article className="stat-card">
          <span className="stat-card__label">Active Use Case</span>
          <strong className="stat-card__value stat-card__value--small">{usecase}</strong>
          <span className="stat-card__meta">Switch retrieval/generation behavior</span>
        </article>
      </section>

      <section className="workspace">
        <section className="panel panel--documents">
          <div className="panel__top">
            <div className="panel__header">
              <div>
                <h2>Indexed Files</h2>
              </div>
              <button className="button button--danger" onClick={handleDeleteAll} type="button">
                Delete All
              </button>
            </div>
            <p className="panel__status">{indexedStatus}</p>
          </div>
          <div className="panel__scroll">
            <div className="doc-list">
              {documents.length ? (
                documents.map((doc) => (
                  <article className="doc-card" key={doc.doc_id}>
                    <div className="doc-card__content">
                      <strong>{doc.title}</strong>
                      <p>{doc.source_path ?? "local upload"}</p>
                      <p>{doc.chunk_count} chunk(s)</p>
                    </div>
                    <button
                      className="button button--danger button--small"
                      onClick={() => void handleDelete(doc.doc_id)}
                      type="button"
                    >
                      Delete
                    </button>
                  </article>
                ))
              ) : (
                <div className="empty-state">No indexed files yet.</div>
              )}
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel__top">
            <h2>Upload Documents</h2>
            <div className="upload-callout">
              Drag in notes, policies, or contracts as <code>.txt</code>, <code>.md</code>, and <code>.pdf</code> files.
            </div>
            <p className="panel__status">{uploadStatus}</p>
          </div>
          <div className="panel__scroll">
            <form className="stack" onSubmit={handleUpload}>
              <label className="field field--upload">
                <span>Choose local files</span>
                <input
                  multiple
                  accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
                  onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
                  type="file"
                />
              </label>
              <div className="file-chip-list">
                {files.length ? (
                  files.map((file) => (
                    <span className="file-chip" key={file.name}>
                      {file.name}
                    </span>
                  ))
                ) : (
                  <span className="muted">No files selected.</span>
                )}
              </div>
              <button className="button" type="submit">
                Upload Selected Files
              </button>
            </form>
            <div className="surface">{uploadSummary}</div>
          </div>
        </section>

        <section className="panel">
          <div className="panel__top">
            <h2>Retrieve Relevant Documents</h2>
            <p className="panel__status">{chatStatus}</p>
          </div>
          <div className="panel__scroll">
            <form className="stack" onSubmit={handleAsk}>
              <label className="field">
                <span>Question</span>
                <textarea
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Ask about uploaded material..."
                  value={query}
                />
              </label>
              <label className="field">
                <span>Use case</span>
                <select onChange={(event) => setUsecase(event.target.value)} value={usecase}>
                  {usecases.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>
              <button className="button" type="submit">
                Retrieve Then Answer
              </button>
            </form>
            {retrievalQuery ? (
              <div className="surface">
                Retrieval query: <strong>{retrievalQuery}</strong>
              </div>
            ) : null}
            <div className="retrieval-list">
              {retrievals.length ? (
                retrievals.map((hit, index) => (
                  <article className="surface retrieval-card" key={`${hit.chunk_id}-${index}`}>
                    <div className="retrieval-card__topline">
                      <strong>{hit.title}</strong>
                      <span className="retrieval-card__badge">Evidence {index + 1}</span>
                    </div>
                    <p className="muted">{hit.doc_id} | score {hit.score.toFixed(3)}</p>
                    <p>{hit.snippet}</p>
                  </article>
                ))
              ) : (
                <div className="surface muted">No retrieval yet.</div>
              )}
            </div>
          </div>
        </section>

        <section className="panel panel--answer">
          <div className="panel__top">
            <div className="panel__header">
              <h2>Answer</h2>
              <button className="button button--small" onClick={() => void handleEvaluate()} type="button">
                Evaluate
              </button>
            </div>
            <p className="panel__status">{evaluationStatus}</p>
          </div>
          <div className="panel__scroll">
            <div className="answer-card">{answer}</div>
            <div className="surface" style={{ marginTop: "12px" }}>
              {evaluation || "Judge output will appear here."}
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}
