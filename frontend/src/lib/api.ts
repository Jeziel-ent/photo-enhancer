/**
 * Typed client for the Adinn 4K Image Enhancer backend (backend/server.py).
 *
 * Local-only HTTP API, same origin (Vite dev proxy / desktop shell both map
 * /api and /health to the real backend) — see backend/server.py's module
 * docstring for the exact contract this mirrors.
 */

export type JobStatus = "queued" | "processing" | "completed" | "failed";

export interface JobFileError {
  filename: string;
  message: string;
}

export interface CreateJobResponse {
  ok: true;
  job_id: string;
  status: JobStatus;
  total_count: number;
}

export interface JobStatusResponse {
  ok: true;
  job_id: string;
  status: JobStatus;
  progress: number;
  current_file: string | null;
  completed_count: number;
  total_count: number;
  errors: JobFileError[];
}

interface ApiErrorBody {
  ok: false;
  error: string;
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface JobResult {
  blob: Blob;
  filename: string;
  /** "application/zip" for a multi-file job, "image/png" for a single file. */
  contentType: string;
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { ok?: unknown }).ok === false &&
    typeof (value as { error?: unknown }).error === "string"
  );
}

async function readJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

async function parseJsonResponse<T>(res: Response): Promise<T> {
  const body = await readJson(res);
  if (!res.ok || isApiErrorBody(body)) {
    const message = isApiErrorBody(body) ? body.error : `request failed (${res.status})`;
    throw new ApiError(message, res.status);
  }
  return body as T;
}

/**
 * Creates a job from one or more image files. Mirrors POST /api/jobs
 * exactly: multipart/form-data, one file part per image. The backend
 * ignores the multipart field name, so any consistent name is fine.
 */
export async function createJob(files: File[]): Promise<CreateJobResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file, file.name);
  }
  const res = await fetch("/api/jobs", { method: "POST", body: formData });
  return parseJsonResponse<CreateJobResponse>(res);
}

/** GET /api/jobs/:id — a point-in-time snapshot of job progress. */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  return parseJsonResponse<JobStatusResponse>(res);
}

function filenameFromContentDisposition(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const match = /filename="?([^"]+)"?/i.exec(header);
  return match ? match[1] : fallback;
}

/**
 * GET /api/jobs/:id/result — the enhanced PNG (single-file job) or a ZIP
 * (multi-file job). Throws ApiError (409/404/500 all come back as JSON)
 * if the job isn't in a downloadable state yet.
 */
export async function downloadResult(jobId: string): Promise<JobResult> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/result`);
  if (!res.ok) {
    const body = await readJson(res);
    const message = isApiErrorBody(body) ? body.error : `request failed (${res.status})`;
    throw new ApiError(message, res.status);
  }
  const blob = await res.blob();
  const contentType = res.headers.get("Content-Type") || blob.type || "application/octet-stream";
  const filename = filenameFromContentDisposition(
    res.headers.get("Content-Disposition"),
    `${jobId}.bin`,
  );
  return { blob, filename, contentType };
}
