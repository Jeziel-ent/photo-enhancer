/**
 * Bridge to the pywebview desktop shell (backend/shell.py's DesktopBridge,
 * exposed as window.pywebview.api). Absent in the Vite dev server / any
 * plain browser tab, so every call here is optional — callers fall back to
 * the browser download flow when this isn't available.
 */

import { isDefaultAdjustments, type AdjustmentParams } from "./adjustments";

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

export type ImageSaveFormat = "png" | "jpg" | "jpeg";

/** A confirmed billboard rectangle in 3840x2160 image-space coordinates,
 * passed to the backend so OpenCV can composite it at save time (never
 * rasterized frontend-side). Matches the shape of BillboardCanvas's
 * BillboardRect minus its display-only `id`. */
export interface BillboardOverlayRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One gallery image's own rects for a native batch export — mirrors the
 * browser path's api.ts BatchExportImage. */
export interface BatchExportImage {
  result_id: string;
  rects: BillboardOverlayRect[];
}

interface PywebviewApi {
  save_result(jobId: string): Promise<SaveResultOutcome>;
  save_result_as(
    jobId: string,
    imageFormat: ImageSaveFormat,
    billboardRects?: BillboardOverlayRect[],
    adjust?: AdjustmentParams,
  ): Promise<SaveResultOutcome>;
  save_batch_export(
    jobId: string,
    imageFormat: ImageSaveFormat,
    includeOutlines: boolean,
    images: BatchExportImage[],
    adjust?: AdjustmentParams,
  ): Promise<SaveResultOutcome>;
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

/**
 * Same as saveResultNative, but for a single-image job lets the caller pick
 * PNG/JPG/JPEG (see backend/shell.py's DesktopBridge.save_result_as — a
 * batch/ZIP job ignores the format and always saves the zip as-is). Only
 * call this when isDesktopShell() is true — throws otherwise. `adjust`, when
 * given and non-default, is applied before any billboard rects (see
 * lib/adjustments.ts and adjustment_overlay.py — the same semantics as the
 * browser download path).
 */
export async function saveResultAsNative(
  jobId: string,
  format: ImageSaveFormat,
  billboardRects: BillboardOverlayRect[] = [],
  adjust?: AdjustmentParams,
): Promise<SaveResultOutcome> {
  return requireApi().save_result_as(
    jobId, format, billboardRects, adjust && !isDefaultAdjustments(adjust) ? adjust : undefined);
}

/**
 * Native "Save As" for a batch export ZIP: one common format + include-
 * outlines flag for the whole batch, each image carrying only its own
 * rects (see backend/shell.py's DesktopBridge.save_batch_export). Only call
 * this when isDesktopShell() is true — throws otherwise. `adjust`, when
 * given and non-default, is applied to every image in the batch.
 */
export async function saveBatchExportNative(
  jobId: string,
  format: ImageSaveFormat,
  includeOutlines: boolean,
  images: BatchExportImage[],
  adjust?: AdjustmentParams,
): Promise<SaveResultOutcome> {
  return requireApi().save_batch_export(
    jobId, format, includeOutlines, images,
    adjust && !isDefaultAdjustments(adjust) ? adjust : undefined);
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
