/**
 * Client-side mirror of the validation rules enforced by backend/server.py
 * (_validate_upload / MAX_FILE_BYTES / MAX_FILES_PER_JOB / MAX_BODY_BYTES).
 * Catching these before upload gives instant feedback instead of a round
 * trip to get the same 400. The backend remains the source of truth — these
 * limits must stay in sync with it by hand, there is no shared config file.
 */

export const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"] as const;
export const ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"] as const;

export const MAX_FILE_BYTES = 40 * 1024 * 1024;
export const MAX_FILES_PER_JOB = 50;
export const MAX_BODY_BYTES = 200 * 1024 * 1024;

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot === -1 ? "" : filename.slice(dot).toLowerCase();
}

/** Returns an error message for one file, or null if it passes validation on its own. */
export function validateFile(file: File): string | null {
  if (file.size === 0) {
    return "empty file";
  }
  if (file.size > MAX_FILE_BYTES) {
    return `exceeds the ${formatBytes(MAX_FILE_BYTES)} per-file limit`;
  }
  const ext = extensionOf(file.name);
  if (!ALLOWED_EXTENSIONS.includes(ext as (typeof ALLOWED_EXTENSIONS)[number])) {
    return "unsupported file type (allowed: .jpg, .jpeg, .png)";
  }
  if (file.type && !ALLOWED_CONTENT_TYPES.includes(file.type as (typeof ALLOWED_CONTENT_TYPES)[number])) {
    return `unsupported content type (${file.type})`;
  }
  return null;
}

/** Batch-level errors that depend on the whole selection, not one file. */
export function validateBatch(files: File[]): string[] {
  const errors: string[] = [];
  if (files.length > MAX_FILES_PER_JOB) {
    errors.push(`too many files selected (${files.length}) — max ${MAX_FILES_PER_JOB} per job`);
  }
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
  if (totalBytes > MAX_BODY_BYTES) {
    errors.push(`selected files total ${formatBytes(totalBytes)} — max ${formatBytes(MAX_BODY_BYTES)} per job`);
  }
  return errors;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}
