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

/** One successfully enhanced file within a job — `id` is stable (its
 * zero-padded upload index) and is what the gallery uses to fetch that
 * file's own preview/export, independent of any other file in the batch. */
export interface JobResultEntry {
  id: string;
  filename: string;
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
  current_stage: string | null;
  current_stage_label: string | null;
  completed_count: number;
  total_count: number;
  errors: JobFileError[];
  results: JobResultEntry[];
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

/** Billboard rectangle coordinates (3840x2160 image space) to composite onto
 * the downloaded result. Mirrors desktop.ts's BillboardOverlayRect; the
 * backend (backend/server.py) renders them via OpenCV before serving. */
export interface BillboardRectSpec {
  x: number;
  y: number;
  width: number;
  height: number;
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

export type ProcessingDevicePreference = "auto" | "gpu" | "cpu";

export interface SettingsResponse {
  ok: true;
  processing_device: ProcessingDevicePreference;
  /** null briefly right after app launch, before the one-time hardware
   * resolution in the backend's worker thread has run. */
  effective_device: "gpu" | "cpu" | null;
  detected_gpu: string | null;
  warning: string | null;
  /** true when the on-disk preference no longer matches what this running
   * process actually resolved at startup -- CUDA visibility can't be
   * changed mid-process, so the new choice needs an app restart. */
  restart_required: boolean;
}

/** GET /api/settings — current processing-device preference plus what's
 * actually detected/active in the running backend process. */
export async function getSettings(): Promise<SettingsResponse> {
  const res = await fetch("/api/settings");
  return parseJsonResponse<SettingsResponse>(res);
}

/** PUT /api/settings — persists a new processing-device preference. */
export async function updateSettings(
  processingDevice: ProcessingDevicePreference,
): Promise<SettingsResponse> {
  const res = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ processing_device: processingDevice }),
  });
  return parseJsonResponse<SettingsResponse>(res);
}

/** One image's per-image export request within a batch export — its own
 * board rects only, never another image's. */
export interface BatchExportImage {
  result_id: string;
  rects: BillboardRectSpec[];
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
 *
 * Pass `billboardRects` to have the backend composite the confirmed editor
 * rectangles onto a single-image result before serving it.
 */
export async function downloadResult(
  jobId: string,
  billboardRects?: BillboardRectSpec[],
): Promise<JobResult> {
  let url = `/api/jobs/${encodeURIComponent(jobId)}/result`;
  if (billboardRects && billboardRects.length > 0) {
    url += `?billboard=${encodeURIComponent(JSON.stringify(billboardRects))}`;
  }
  const res = await fetch(url);
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

/**
 * GET /api/jobs/:id/results/:resultId — one gallery image's own clean
 * enhanced PNG (never the ZIP, never another image's file). Used to
 * populate the gallery's main preview and thumbnails without waiting on a
 * ZIP re-fetch, and without re-running enhancement.
 */
export async function downloadIndividualResult(
  jobId: string,
  resultId: string,
): Promise<JobResult> {
  const res = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}/results/${encodeURIComponent(resultId)}`,
  );
  if (!res.ok) {
    const body = await readJson(res);
    const message = isApiErrorBody(body) ? body.error : `request failed (${res.status})`;
    throw new ApiError(message, res.status);
  }
  const blob = await res.blob();
  return { blob, filename: `${resultId}.png`, contentType: "image/png" };
}

/**
 * POST /api/jobs/:id/export — the batch ("Save ZIP") export path: one
 * COMMON format for every image, each image getting only its own board
 * rects burned in when `includeOutlines` is on. Mirrors
 * backend/output_manager.build_batch_export exactly.
 */
export async function exportBatch(
  jobId: string,
  format: "png" | "jpg" | "jpeg",
  includeOutlines: boolean,
  images: BatchExportImage[],
): Promise<JobResult> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, include_outlines: includeOutlines, images }),
  });
  if (!res.ok) {
    const body = await readJson(res);
    const message = isApiErrorBody(body) ? body.error : `request failed (${res.status})`;
    throw new ApiError(message, res.status);
  }
  const blob = await res.blob();
  const filename = filenameFromContentDisposition(
    res.headers.get("Content-Disposition"),
    `enhanced_images_${jobId.slice(0, 8)}.zip`,
  );
  return { blob, filename, contentType: "application/zip" };
}
