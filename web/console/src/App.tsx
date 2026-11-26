import { useRef, useState } from "react";
import type { FormEvent } from "react";
import type { CitationPayload, ServerEvent } from "./types";

type Tenant = "tenant_alpha" | "tenant_beta";
type Role = "all" | "engineer";

function parseFrame(frame: string): ServerEvent | null {
  let eventType = "";
  const data: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!eventType || data.length === 0) return null;
  const parsed: unknown = JSON.parse(data.join("\n"));
  if (typeof parsed !== "object" || parsed === null) return null;
  const payload = parsed as Record<string, unknown>;
  if (payload.type !== eventType || typeof payload.id !== "number") return null;
  switch (eventType) {
    case "node":
      return typeof payload.label === "string"
        ? (payload as ServerEvent)
        : null;
    case "token":
      return typeof payload.text === "string" ? (payload as ServerEvent) : null;
    case "citations":
      if (!Array.isArray(payload.citations)) return null;
      return (payload.citations as CitationPayload[]).every(
        (citation) =>
          typeof citation.doc_name === "string" &&
          typeof citation.page === "number",
      )
        ? (payload as ServerEvent)
        : null;
    case "final":
      return typeof payload.answer === "string" &&
        typeof payload.abstained === "boolean" &&
        typeof payload.cache_hit === "boolean"
        ? (payload as ServerEvent)
        : null;
    case "error":
      return typeof payload.message === "string"
        ? (payload as ServerEvent)
        : null;
    default:
      return null;
  }
}

export default function App() {
  const [tenant, setTenant] = useState<Tenant>("tenant_alpha");
  const [role, setRole] = useState<Role>("all");
  const [question, setQuestion] = useState("");
  const [nodes, setNodes] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<CitationPayload[]>([]);
  const [abstained, setAbstained] = useState<boolean | null>(null);
  const [cacheHit, setCacheHit] = useState<boolean | null>(null);
  const [chatStatus, setChatStatus] = useState("Ready");
  const [uploadStatus, setUploadStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setNodes([]);
    setAnswer("");
    setCitations([]);
    setAbstained(null);
    setCacheHit(null);
    setChatStatus("Connecting…");
    let lastEventId = 0;
    try {
      const response = await fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": tenant,
          "X-Roles": role,
          Accept: "text/event-stream",
          "Last-Event-ID": String(lastEventId),
        },
        body: JSON.stringify({ question: question.trim() }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        setChatStatus(`Request failed (${response.status}).`);
        return;
      }
      setChatStatus("Working…");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const parsed = parseFrame(frame);
          if (!parsed || parsed.id <= lastEventId) continue;
          lastEventId = parsed.id;
          switch (parsed.type) {
            case "node":
              setNodes((current) => [...current, parsed.label]);
              break;
            case "token":
              setAnswer((current) => current + parsed.text);
              break;
            case "citations":
              setCitations(parsed.citations);
              break;
            case "final":
              setAnswer(parsed.answer);
              setAbstained(parsed.abstained);
              setCacheHit(parsed.cache_hit);
              setChatStatus("Complete");
              break;
            case "error":
              setChatStatus(parsed.message);
              break;
          }
        }
      }
      if (buffer.trim()) {
        const parsed = parseFrame(buffer);
        if (parsed?.type === "final") {
          setAnswer(parsed.answer);
          setAbstained(parsed.abstained);
          setCacheHit(parsed.cache_hit);
          setChatStatus("Complete");
        } else if (parsed?.type === "error") {
          setChatStatus(parsed.message);
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setChatStatus("Cancelled");
      } else {
        setChatStatus("Could not connect to the service.");
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setUploadStatus("Choose a PDF first.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    setUploadStatus("Uploading…");
    try {
      const response = await fetch("/api/v1/ingest", {
        method: "POST",
        headers: { "X-Tenant-ID": tenant, "X-Roles": role },
        body: form,
      });
      if (!response.ok) {
        setUploadStatus(`Upload failed (${response.status}).`);
        return;
      }
      const accepted: unknown = await response.json();
      if (
        typeof accepted !== "object" ||
        accepted === null ||
        typeof (accepted as { job_id?: unknown }).job_id !== "string"
      ) {
        setUploadStatus("The server returned an invalid job response.");
        return;
      }
      const jobId = (accepted as { job_id: string }).job_id;
      setUploadStatus(`Queued job ${jobId}.`);
      for (let attempt = 0; attempt < 90; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        const statusResponse = await fetch(
          `/api/v1/ingest/${encodeURIComponent(jobId)}`,
          { headers: { "X-Tenant-ID": tenant, "X-Roles": role } },
        );
        if (!statusResponse.ok) {
          setUploadStatus(`Could not read job status (${statusResponse.status}).`);
          return;
        }
        const result: unknown = await statusResponse.json();
        const status =
          typeof result === "object" && result !== null
            ? (result as { status?: unknown }).status
            : null;
        if (status === "done" || status === "failed") {
          setUploadStatus(`Ingestion ${status}.`);
          return;
        }
      }
      setUploadStatus("Ingestion is still running; check the job again later.");
    } catch {
      setUploadStatus("Could not connect to the ingestion service.");
    }
  }

  return (
    <main className="shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">FACILITIES KNOWLEDGE</p>
          <h1>Service assistant</h1>
          <p className="subtitle">
            Ask a maintenance question and inspect the sources used to answer.
          </p>
        </div>
        <span className="environment-badge">Local development</span>
      </header>

      <section className="card controls" aria-labelledby="identity-title">
        <h2 id="identity-title">Request access</h2>
        <div className="control-grid">
          <div className="field">
            <label htmlFor="tenant">Tenant</label>
            <select
              id="tenant"
              value={tenant}
              onChange={(event) => setTenant(event.target.value as Tenant)}
            >
              <option value="tenant_alpha">Alpha</option>
              <option value="tenant_beta">Beta</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="role">Role</label>
            <select
              id="role"
              value={role}
              onChange={(event) => setRole(event.target.value as Role)}
            >
              <option value="all">All</option>
              <option value="engineer">Engineer</option>
            </select>
          </div>
        </div>
        <p className="help-text">
          These headers are accepted only while the API is running with ENV=dev.
        </p>
      </section>

      <div className="content-grid">
        <section className="card question-card" aria-labelledby="question-title">
          <h2 id="question-title">Ask a question</h2>
          <form onSubmit={ask}>
            <div className="field">
              <label htmlFor="question">Maintenance question</label>
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="For example: What is the emergency repair SLA?"
                maxLength={8000}
                required
                rows={5}
              />
            </div>
            <div className="button-row">
              <button type="submit" disabled={busy || !question.trim()}>
                {busy ? "Asking…" : "Ask assistant"}
              </button>
              {busy && (
                <button className="secondary-button" type="button" onClick={cancel}>
                  Cancel
                </button>
              )}
            </div>
          </form>
          <p className="status-line" aria-live="polite">{chatStatus}</p>
        </section>

        <section className="card result-card" aria-labelledby="result-title">
          <div className="section-heading">
            <h2 id="result-title">Answer</h2>
            <div className="badges" aria-live="polite">
              {abstained !== null && (
                <span className={abstained ? "badge warning" : "badge success"}>
                  {abstained ? "Abstained" : "Answered"}
                </span>
              )}
              {cacheHit === true && <span className="badge">Cache hit</span>}
            </div>
          </div>
          <div className="progress" aria-live="polite" aria-label="Request progress">
            {nodes.length === 0 ? (
              <span className="muted">Progress labels will appear here.</span>
            ) : (
              nodes.map((label, index) => (
                <span className="progress-item" key={`${index}-${label}`}>
                  <span className="progress-dot" aria-hidden="true" />
                  {label}
                </span>
              ))
            )}
          </div>
          <p className="answer-text" aria-live="polite">
            {answer || "Your answer will appear here."}
          </p>
          <div className="citation-section">
            <h3>Sources</h3>
            {citations.length === 0 ? (
              <p className="muted">No citations received.</p>
            ) : (
              <ul>
                {citations.map((citation, index) => (
                  <li key={`${citation.doc_name}-${citation.page}-${index}`}>
                    <span>{citation.doc_name}</span>
                    <span className="page-label">Page {citation.page}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>

      <section className="card upload-card" aria-labelledby="upload-title">
        <h2 id="upload-title">Add a document</h2>
        <p className="subtitle">Upload a PDF to the selected tenant’s library.</p>
        <form className="upload-form" onSubmit={upload}>
          <div className="field file-field">
            <label htmlFor="pdf-file">PDF document</label>
            <input
              id="pdf-file"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </div>
          <button type="submit" disabled={!file}>
            Upload and ingest
          </button>
        </form>
        <p className="status-line" aria-live="polite">{uploadStatus}</p>
      </section>
      <footer>EFM Corrective RAG · local console</footer>
    </main>
  );
}
