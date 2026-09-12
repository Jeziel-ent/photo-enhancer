/**
 * Bridge to the pywebview desktop shell (backend/shell.py's DesktopBridge,
 * exposed as window.pywebview.api). Absent in the Vite dev server / any
 * plain browser tab, so every call here is optional — callers fall back to
 * the browser download flow when this isn't available.
 */

export interface SaveResultOk {
  ok: true;
  path: string;
}

export interface SaveResultCancelled {
  ok: false;
  cancelled: true;
}

export interface SaveResultError {
  ok: false;
  error: string;
}

export type SaveResultOutcome = SaveResultOk | SaveResultCancelled | SaveResultError;

/** A lightweight record of one image/batch the user actually saved. */
export interface RecentEntry {
  id: string;
  filename: string;
  /** Full path to the saved file, exactly as the user chose it. */
  path: string;
  /** The saved file's parent directory. */
  directory: string;
  /** ISO 8601 timestamp. */
  saved_at: string;
  kind: "image" | "zip";
  file_count: number | null;
  thumbnail_data_url: string | null;
}

export interface RecordSavedResultInput {
  path: string;
  kind: "image" | "zip";
  file_count?: number;
  thumbnail_data_url?: string;
}

export interface OkResult {
  ok: true;
}

export interface ErrorResult {
  ok: false;
  error: string;
}

export type OkOrError = OkResult | ErrorResult;

export interface RecentEntriesOk {
  ok: true;
  entries: RecentEntry[];
}

export type RecentEntriesResult = RecentEntriesOk | ErrorResult;

interface PywebviewApi {
  save_result(jobId: string): Promise<SaveResultOutcome>;
  record_saved_result(entry: RecordSavedResultInput): Promise<RecentEntriesResult>;
  get_recent_history(): Promise<RecentEntriesResult>;
  open_in_explorer(path: string): Promise<OkOrError>;
  open_saved_file(path: string): Promise<OkOrError>;
}

declare global {
  interface Window {
    pywebview?: { api: PywebviewApi };
  }
}

/** True when running inside the pywebview desktop shell. */
export function isDesktopShell(): boolean {
  return typeof window !== "undefined" && window.pywebview?.api != null;
}

function requireApi(): PywebviewApi {
  if (!window.pywebview?.api) {
    throw new Error("desktop bridge is not available in this environment");
  }
  return window.pywebview.api;
}

/**
 * Opens the native "Save As" dialog for a completed job's result and copies
 * it to the chosen destination. Only call this when isDesktopShell() is
 * true — throws otherwise.
 */
export async function saveResultNative(jobId: string): Promise<SaveResultOutcome> {
  return requireApi().save_result(jobId);
}

/** Persists a lightweight record of a result the user just actually saved. */
export async function recordSavedResult(
  entry: RecordSavedResultInput,
): Promise<RecentEntriesResult> {
  return requireApi().record_saved_result(entry);
}

/** Reads back every saved-result record, newest first. */
export async function getRecentHistory(): Promise<RecentEntriesResult> {
  return requireApi().get_recent_history();
}

/** Opens Windows File Explorer at a saved result's directory. */
export async function openInExplorer(path: string): Promise<OkOrError> {
  return requireApi().open_in_explorer(path);
}

/** Opens a saved result with its default viewer. */
export async function openSavedFile(path: string): Promise<OkOrError> {
  return requireApi().open_saved_file(path);
}
