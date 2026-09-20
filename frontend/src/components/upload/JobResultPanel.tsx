import { useEffect, useRef, useState } from "react";
import type { JobStatusResponse } from "../../lib/api";
import { DEFAULT_ADJUSTMENTS, isDefaultAdjustments, type AdjustmentParams } from "../../lib/adjustments";
import { cn } from "../../lib/cn";
import { isDesktopShell, type ImageSaveFormat } from "../../lib/desktop";
import { loadImageDimensions, type ImageDimensions } from "../../lib/imageDimensions";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { Select } from "../ui/Field";
import {
  IconAlertTriangle,
  IconCheck,
  IconChevronRight,
  IconDownload,
  IconFocus,
  IconPlus,
  IconRefresh,
  IconTrash,
} from "../ui/Icon";
import { AdjustedPreviewCanvas } from "./AdjustedPreviewCanvas";
import { AdjustmentControls } from "./AdjustmentControls";
import {
  BillboardCanvas,
  formatRectPosition,
  formatRectSize,
  type BillboardRect,
} from "./BillboardCanvas";
import { CompareSlider } from "./CompareSlider";
import { FileErrorList } from "./FileErrorList";

/** One gallery image, already fetched: its own before (original upload) and
 * after (enhanced result) object URLs. Populated once per job by the
 * caller (App.tsx) — this component never fetches or re-enhances anything
 * itself, only ever switches which already-loaded image is shown. */
export interface GalleryResult {
  id: string;
  filename: string;
  beforeUrl: string;
  afterUrl: string;
}

/** Per-image board state, keyed by GalleryResult.id (or SINGLE_ID for a
 * one-photo job) — never one shared rects[] for the whole batch, so
 * switching the gallery selection can never leak one photo's boards onto
 * another's. */
interface ImageSelection {
  rects: BillboardRect[];
  selectedId: string | null;
}

const SINGLE_ID = "single";
const EMPTY_SELECTION: ImageSelection = { rects: [], selectedId: null };

function shortFilename(name: string, max = 16): string {
  if (name.length <= max) return name;
  const dot = name.lastIndexOf(".");
  const ext = dot > 0 ? name.slice(dot) : "";
  const stem = dot > 0 ? name.slice(0, dot) : name;
  const keep = Math.max(3, max - ext.length - 1);
  return `${stem.slice(0, keep)}…${ext}`;
}

function Toggle({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors",
        checked ? "bg-brand" : "bg-line-strong",
        disabled && "cursor-not-allowed opacity-40",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 transform rounded-full bg-white shadow-card transition-transform",
          checked ? "translate-x-[19px]" : "translate-x-0.5",
        )}
      />
    </button>
  );
}

function BoardThumb({
  src,
  rect,
  imageDimensions,
}: {
  src?: string;
  rect: BillboardRect;
  /** The result image's real pixel dimensions (rect coordinates are in this
   * same space) — falls back to the classic 3840x2160 while not yet known. */
  imageDimensions: ImageDimensions;
}) {
  if (!src) {
    return (
      <span className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-md border border-line bg-canvas text-faint">
        <IconFocus className="h-4.5 w-4.5" />
      </span>
    );
  }
  const rectW = rect.width / imageDimensions.width;
  const rectH = rect.height / imageDimensions.height;
  return (
    <span className="relative inline-block h-12 w-12 shrink-0 overflow-hidden rounded-md border border-line bg-canvas">
      <img
        src={src}
        alt=""
        draggable={false}
        className="absolute max-w-none"
        style={{
          width: `${(1 / rectW) * 100}%`,
          height: `${(1 / rectH) * 100}%`,
          left: `${-(rect.x / rect.width) * 100}%`,
          top: `${-(rect.y / rect.height) * 100}%`,
        }}
      />
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 rounded-md border border-brand/70"
      />
    </span>
  );
}

/** Compact horizontal thumbnail strip below the main preview — only shown
 * for a multi-photo (>1) job. Clicking a thumbnail only ever swaps which
 * already-fetched result is shown; it never re-fetches or re-enhances. */
function GalleryStrip({
  results,
  activeId,
  boardCounts,
  disabled,
  onSelect,
}: {
  results: GalleryResult[];
  activeId: string | null;
  boardCounts: Record<string, number>;
  disabled: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Enhanced photos">
      {results.map((result) => {
        const isActive = result.id === activeId;
        const boardCount = boardCounts[result.id] ?? 0;
        return (
          <button
            key={result.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            disabled={disabled}
            onClick={() => onSelect(result.id)}
            title={result.filename}
            className={cn(
              "group relative w-[100px] shrink-0 overflow-hidden rounded-lg border-2 bg-canvas text-left transition-colors",
              isActive ? "border-brand shadow-pop" : "border-line hover:border-line-strong",
              disabled && !isActive && "opacity-60",
            )}
          >
            <span className="block aspect-[4/3] w-full overflow-hidden bg-canvas">
              <img
                src={result.afterUrl}
                alt=""
                draggable={false}
                className="h-full w-full object-cover"
              />
            </span>
            {boardCount > 0 ? (
              <span className="absolute right-1 top-1 rounded-full bg-brand px-1.5 py-0.5 text-[9.5px] font-semibold leading-none text-white shadow-card">
                {boardCount} board{boardCount === 1 ? "" : "s"}
              </span>
            ) : null}
            <span className="block truncate px-1.5 py-1 text-[10.5px] font-medium text-ink-2">
              {shortFilename(result.filename)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function JobResultPanel({
  status,
  comparison,
  results,
  onDownload,
  onExportBatch,
  downloading,
  downloadError,
  onReset,
}: {
  status: JobStatusResponse;
  /** Single-photo job only (total_count === 1). */
  comparison: { beforeUrl: string; afterUrl: string } | null;
  /** Multi-photo job only (total_count > 1) — one entry per successfully
   * enhanced image, in upload order. Empty until the gallery has finished
   * fetching each image's own preview. */
  results: GalleryResult[];
  /** `format`/`billboardRects` only apply to a single-image result — the
   * confirmed editor rectangles (3840x2160 image space) the backend should
   * composite onto the saved image. `adjust`, when not at its defaults, is
   * the current manual-adjustment slider state (see lib/adjustments.ts). */
  onDownload: (
    format: ImageSaveFormat,
    billboardRects: BillboardRect[],
    adjust?: AdjustmentParams,
  ) => void;
  /** Batch ("Save ZIP") export: one common format + include-outlines flag
   * for the whole job; each image carries its own confirmed rects and its
   * own `adjust` (undefined = defaults → exported untouched). */
  onExportBatch: (
    format: ImageSaveFormat,
    includeOutlines: boolean,
    images: { resultId: string; rects: BillboardRect[]; adjust?: AdjustmentParams }[],
  ) => void;
  downloading: boolean;
  downloadError: string | null;
  onReset: () => void;
}) {
  const isBatch = status.total_count > 1;
  const succeededCount = status.total_count - status.errors.length;
  const nativeSave = isDesktopShell();

  const [activeResultId, setActiveResultId] = useState<string | null>(null);
  const [selections, setSelections] = useState<Record<string, ImageSelection>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [includeOutlines, setIncludeOutlines] = useState(true);
  const [saveFormat, setSaveFormat] = useState<ImageSaveFormat>("png");
  // MVP 2: manual post-processing adjustments, applied AFTER the automatic
  // enhancement (see lib/adjustments.ts + AdjustmentControls). Keyed by
  // result id (or SINGLE_ID for a one-photo job) so every enhanced image
  // keeps its OWN slider state — switching the gallery selection can never
  // leak one photo's adjustments onto another's, and Export sends each
  // image's own values. A key that has never been edited reads as the
  // shared DEFAULT_ADJUSTMENTS constant (never mutated — every write
  // below stores a fresh object for exactly that id).
  const [adjustmentsByResult, setAdjustmentsByResult] = useState<Record<string, AdjustmentParams>>(
    {},
  );
  const areasPanelRef = useRef<HTMLDivElement>(null);
  const editSnapshotRef = useRef<{ id: string; selection: ImageSelection } | null>(null);
  // MVP 2: each result's REAL pixel dimensions (no longer always
  // 3840x2160 — aspect ratio is preserved, not stretched). Cached per
  // result id so re-selecting an already-loaded gallery photo is instant.
  // Falls back to the classic 3840x2160 (matching pre-MVP2 behavior
  // exactly) for the brief moment before a freshly-selected image's real
  // size has loaded.
  const [dimensionsByResult, setDimensionsByResult] = useState<Record<string, ImageDimensions>>(
    {},
  );

  // Default the gallery selection to the first available result as soon as
  // it shows up — never leaves the main preview stuck on "no selection".
  useEffect(() => {
    if (!isBatch) return;
    if (activeResultId && results.some((r) => r.id === activeResultId)) return;
    setActiveResultId(results[0]?.id ?? null);
  }, [isBatch, results, activeResultId]);

  const activeId = isBatch ? activeResultId : SINGLE_ID;
  const activeResult = isBatch ? results.find((r) => r.id === activeResultId) ?? null : null;
  const activeComparison = isBatch
    ? activeResult
      ? { beforeUrl: activeResult.beforeUrl, afterUrl: activeResult.afterUrl }
      : null
    : comparison;

  const getSelection = (id: string | null): ImageSelection =>
    (id && selections[id]) || EMPTY_SELECTION;

  // The active image's own adjustment values: an unedited image reads the
  // (never-mutated) DEFAULT_ADJUSTMENTS constant; every edit stores a fresh
  // object under exactly that id, so switching away and back restores it.
  const activeAdjustments: AdjustmentParams =
    (activeId && adjustmentsByResult[activeId]) || DEFAULT_ADJUSTMENTS;

  const setActiveAdjustments = (next: AdjustmentParams) => {
    if (!activeId) return;
    setAdjustmentsByResult((prev) => ({ ...prev, [activeId]: next }));
  };

  const activeSelection = getSelection(activeId);
  const rects = activeSelection.rects;
  const selectedId = activeSelection.selectedId;

  const setActiveSelection = (updater: (prev: ImageSelection) => ImageSelection) => {
    if (!activeId) return;
    setSelections((prev) => ({ ...prev, [activeId]: updater(prev[activeId] || EMPTY_SELECTION) }));
  };

  const setRects = (next: BillboardRect[]) =>
    setActiveSelection((prev) => ({ ...prev, rects: next }));
  const setSelectedId = (next: string | null) =>
    setActiveSelection((prev) => ({ ...prev, selectedId: next }));

  const isEditing = editingId !== null && editingId === activeId;
  const canDrawBoards = activeComparison !== null && activeId !== null;
  const activeDimensions: ImageDimensions = (activeId && dimensionsByResult[activeId]) || {
    width: 3840,
    height: 2160,
  };

  useEffect(() => {
    if (!activeId || !activeComparison || dimensionsByResult[activeId]) return;
    let cancelled = false;
    loadImageDimensions(activeComparison.afterUrl)
      .then((dims) => {
        if (!cancelled) setDimensionsByResult((prev) => ({ ...prev, [activeId]: dims }));
      })
      .catch(() => {
        // Keep the 3840x2160 fallback — board marking still works, just
        // against the pre-MVP2 default until/unless it happens to load.
      });
    return () => {
      cancelled = true;
    };
  }, [activeId, activeComparison, dimensionsByResult]);

  // MVP 2: the adjustment sliders' live preview is now rendered entirely
  // client-side (see AdjustedPreviewCanvas / lib/previewAdjustments.ts) --
  // no backend fetch per slider change. `adjustments` itself is passed
  // straight down to the canvas below; only the Export button's click
  // handler ever sends it to the backend (once, for the real, full-
  // resolution, pixel-exact result).

  const enterEditing = () => {
    if (!activeId) return;
    editSnapshotRef.current = {
      id: activeId,
      selection: {
        rects: activeSelection.rects.map((r) => ({ ...r })),
        selectedId: activeSelection.selectedId,
      },
    };
    setEditingId(activeId);
    setAdding(false);
  };

  const cancelEditing = () => {
    const snap = editSnapshotRef.current;
    if (snap) {
      setSelections((prev) => ({ ...prev, [snap.id]: snap.selection }));
    }
    setEditingId(null);
    setAdding(false);
  };

  const confirmBoards = () => {
    setEditingId(null);
    setAdding(false);
  };

  const deleteSelected = () => {
    if (!selectedId) return;
    setActiveSelection((prev) => ({
      rects: prev.rects.filter((r) => r.id !== selectedId),
      selectedId: null,
    }));
  };

  const clearAllBoards = () => {
    setActiveSelection(() => ({ rects: [], selectedId: null }));
    setAdding(false);
  };

  const selectGalleryResult = (id: string) => {
    if (editingId !== null) return; // finish Cancel/Done on the current photo first
    setActiveResultId(id);
  };

  const scrollToAreas = () => {
    areasPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const boardCounts: Record<string, number> = {};
  for (const r of results) boardCounts[r.id] = getSelection(r.id).rects.length;

  const exportRects = includeOutlines ? rects : [];

  const galleryReady = !isBatch || results.length > 0;

  const contextual = isBatch
    ? galleryReady
      ? "Choose a photo below to preview it or mark board areas, then export the batch."
      : "Preparing your gallery…"
    : isEditing
      ? "Draw the areas you want marked on the photo, then tap Done."
      : "Ready to save.";

  return (
    <Card variant="glass" padding="lg" className="animate-rise-in space-y-6">
      {/* Status header */}
      <header className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3 pb-1">
        <div className="flex items-center gap-3.5">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-ok text-white shadow-card">
            <IconCheck className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-brand">
                {isBatch ? "Batch Complete" : "4K Ready"}
              </p>
              <Badge tone="ok">Completed</Badge>
            </div>
            <h2 className="mt-1 text-[17px] font-semibold tracking-tight text-ink">
              {succeededCount} photo{succeededCount === 1 ? "" : "s"} enhanced to 4K
            </h2>
            <p className="mt-0.5 text-[12.5px] text-muted">{contextual}</p>
          </div>
        </div>
      </header>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_332px]">
        {/* Main column: image + gallery + actions */}
        <div className="min-w-0 space-y-3">
          <div className="relative">
            <div className="overflow-hidden rounded-xl border border-line bg-panel shadow-card">
              {activeComparison ? (
                isEditing || rects.length > 0 ? (
                  <BillboardCanvas
                    imageSrc={activeComparison.afterUrl}
                    imageWidth={activeDimensions.width}
                    imageHeight={activeDimensions.height}
                    rectangles={rects}
                    selectedId={selectedId}
                    onChange={setRects}
                    onSelect={setSelectedId}
                    readOnly={!isEditing}
                    addingMode={adding}
                    onDrawStart={() => setAdding(true)}
                    onDrawCommit={() => setAdding(false)}
                    background={
                      <AdjustedPreviewCanvas
                        srcUrl={activeComparison.afterUrl}
                        adjustments={activeAdjustments}
                        aspect={{
                          width: activeDimensions.width,
                          height: activeDimensions.height,
                        }}
                        alt="Enhanced 4K result"
                        className="h-full w-full"
                      />
                    }
                  />
                ) : (
                  <CompareSlider
                    beforeSrc={activeComparison.beforeUrl}
                    after={
                      <AdjustedPreviewCanvas
                        srcUrl={activeComparison.afterUrl}
                        adjustments={activeAdjustments}
                        alt="Enhanced 4K result"
                        className="h-full w-full object-cover"
                      />
                    }
                  />
                )
              ) : (
                <div className="flex aspect-[16/9] items-center justify-center bg-canvas">
                  <div className="flex flex-col items-center gap-2 text-faint">
                    <IconFocus className="h-6 w-6 animate-soft-pulse" />
                    <p className="text-[12.5px]">
                      {isBatch ? "Preparing gallery…" : "Loading preview…"}
                    </p>
                  </div>
                </div>
              )}
            </div>

            {activeComparison ? (
              <>
                {!isEditing && rects.length === 0 ? (
                  <span className="pointer-events-none absolute left-3 top-3 rounded-full bg-ink/60 px-2.5 py-1 text-[10.5px] font-medium text-white backdrop-blur-sm">
                    Before
                  </span>
                ) : null}
                <span className="pointer-events-none absolute right-3 top-3 rounded-full bg-brand px-2.5 py-1 text-[10.5px] font-semibold text-white">
                  After · 4K
                </span>
              </>
            ) : null}
          </div>

          {isBatch && results.length > 0 ? (
            <GalleryStrip
              results={results}
              activeId={activeResultId}
              boardCounts={boardCounts}
              disabled={editingId !== null}
              onSelect={selectGalleryResult}
            />
          ) : null}

          {isEditing ? (
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line/70 bg-white/70 px-3 py-2.5">
              <Button
                variant="secondary"
                size="sm"
                icon={<IconPlus className="h-4 w-4" />}
                onClick={() => setAdding((value) => !value)}
              >
                Add Board
              </Button>
              <div className="flex items-center gap-1.5">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!selectedId}
                  icon={<IconTrash className="h-4 w-4" />}
                  onClick={deleteSelected}
                >
                  Delete
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={rects.length === 0}
                  onClick={clearAllBoards}
                >
                  Clear All
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={cancelEditing}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  disabled={rects.length === 0}
                  icon={<IconCheck className="h-4 w-4" />}
                  onClick={confirmBoards}
                >
                  Done
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Button variant="ghost" onClick={onReset} icon={<IconRefresh className="h-4 w-4" />}>
                Enhance more photos
              </Button>
              {canDrawBoards ? (
                <Button
                  variant="secondary"
                  onClick={enterEditing}
                  icon={<IconFocus className="h-4 w-4" />}
                >
                  {rects.length > 0 ? "Edit Boards" : "Mark Boards"}
                </Button>
              ) : null}
            </div>
          )}

          {!isEditing && rects.length > 0 ? (
            <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-line/70 bg-ink/[0.03] px-3.5 py-2.5">
              <span className="flex items-center gap-2 text-[12.5px] font-medium text-ink">
                <IconFocus className="h-4 w-4 shrink-0 text-brand" />
                {rects.length} Board{rects.length === 1 ? "" : "s"} selected
              </span>
              <button
                type="button"
                onClick={scrollToAreas}
                className="flex items-center gap-0.5 text-[12px] font-semibold text-brand transition-colors hover:text-brand-dark"
              >
                View details
                <IconChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : null}
        </div>

        {/* Right column: adjustments + board areas + export */}
        <div ref={areasPanelRef} className="flex min-w-0 scroll-mt-6 flex-col gap-5">
          <AdjustmentControls
            value={activeAdjustments}
            onChange={setActiveAdjustments}
            disabled={!activeComparison}
          />

          <Card padding="md" className="animate-fade-in">
            <div className="mb-3 flex items-center justify-between gap-2">
              <div className="min-w-0">
                <h3 className="text-[13px] font-semibold tracking-tight text-ink">
                  Board Areas
                </h3>
                {isBatch && activeResult ? (
                  <p className="mt-0.5 truncate text-[11px] text-faint">
                    {shortFilename(activeResult.filename, 28)}
                  </p>
                ) : null}
              </div>
              {canDrawBoards && isEditing ? (
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<IconPlus className="h-4 w-4" />}
                  onClick={() => setAdding(true)}
                >
                  Add Board
                </Button>
              ) : null}
            </div>

            {rects.length === 0 ? (
              <div className="rounded-lg border border-dashed border-line-strong px-3 py-5 text-center">
                {canDrawBoards ? (
                  <>
                    <p className="text-[12px] text-faint">
                      No boards yet. Mark the areas you want outlined on the photo.
                    </p>
                    <Button
                      variant="secondary"
                      size="sm"
                      className="mt-2.5"
                      icon={<IconFocus className="h-4 w-4" />}
                      onClick={enterEditing}
                    >
                      Mark Boards
                    </Button>
                  </>
                ) : (
                  <p className="text-[12px] text-faint">
                    {isBatch
                      ? "Select a photo from the gallery to mark board areas."
                      : "Board marking is available once the preview has loaded."}
                  </p>
                )}
              </div>
            ) : (
              <>
                <p className="mb-2 text-[12px] text-muted">
                  {rects.length} Board{rects.length === 1 ? "" : "s"} selected
                </p>
                <ul className="space-y-1.5">
                  {rects.map((rect) => (
                    <li key={rect.id}>
                      <button
                        type="button"
                        onClick={() =>
                          setSelectedId(selectedId === rect.id ? null : rect.id)
                        }
                        className={cn(
                          "w-full rounded-lg border px-2.5 py-2 text-left transition-colors",
                          selectedId === rect.id
                            ? "border-brand/45 bg-brand-soft/60 ring-1 ring-brand/20"
                            : "border-line hover:bg-ink/[0.02]",
                        )}
                      >
                        <span className="flex items-center gap-3">
                          <BoardThumb
                            src={activeComparison?.afterUrl}
                            rect={rect}
                            imageDimensions={activeDimensions}
                          />
                          <span className="min-w-0 flex-1">
                            <span className="block text-[13px] font-medium text-ink">
                              {formatRectSize(rect)}
                            </span>
                            <span className="mt-0.5 block truncate text-[11.5px] text-faint">
                              {formatRectPosition(rect)}
                            </span>
                          </span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </Card>

          <Card padding="md" className="animate-fade-in">
            <h3 className="mb-3 text-[13px] font-semibold tracking-tight text-ink">Export</h3>

            <div className="space-y-3">
              {!isBatch && !nativeSave ? null : (
                <label className="block">
                  <span className="mb-1 block text-[11.5px] font-medium text-ink-2">
                    Format
                  </span>
                  <Select
                    aria-label="Save format"
                    value={saveFormat}
                    onChange={(event) =>
                      setSaveFormat(event.target.value as ImageSaveFormat)
                    }
                    className="h-9 w-full"
                  >
                    <option value="png">PNG</option>
                    <option value="jpg">JPG</option>
                    <option value="jpeg">JPEG</option>
                  </Select>
                </label>
              )}

              <div
                className={cn(
                  "flex items-center justify-between gap-3",
                  !isBatch && rects.length === 0 && "opacity-50",
                )}
              >
                <div className="text-left">
                  <p className="text-[13px] font-medium text-ink">
                    Include board outlines
                  </p>
                  <p className="mt-0.5 text-[11px] leading-snug text-faint">
                    {isBatch
                      ? "Burns each photo's own board outlines into its export"
                      : "Burns the red board outlines into the photo"}
                  </p>
                </div>
                <Toggle
                  label="Include board outlines"
                  checked={includeOutlines}
                  onChange={setIncludeOutlines}
                  disabled={!isBatch && rects.length === 0}
                />
              </div>
            </div>

            <Button
              variant="primary"
              size="lg"
              className="mt-4 w-full"
              disabled={downloading || (isBatch && results.length === 0)}
              icon={<IconDownload className="h-4 w-4" />}
              onClick={() =>
                isBatch
                  ? onExportBatch(
                      saveFormat,
                      includeOutlines,
                      results.map((r) => ({
                        resultId: r.id,
                        rects: getSelection(r.id).rects,
                        adjust: isDefaultAdjustments(adjustmentsByResult[r.id] ?? DEFAULT_ADJUSTMENTS)
                          ? undefined
                          : { ...adjustmentsByResult[r.id] },
                      })),
                    )
                  : onDownload(saveFormat, exportRects, activeAdjustments)
              }
            >
              {downloading
                ? "Saving…"
                : isBatch
                  ? nativeSave
                    ? "Save ZIP"
                    : "Download ZIP"
                  : nativeSave
                    ? "Save Image"
                    : "Download PNG"}
            </Button>

            {downloadError ? (
              <div className="mt-3 flex items-start gap-1.5 rounded-md border border-brand/20 bg-brand-soft px-3 py-2.5 text-[12.5px] text-brand-dark">
                <IconAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{downloadError}</span>
              </div>
            ) : null}
          </Card>
        </div>
      </div>

      <FileErrorList
        errors={status.errors}
        heading={`${status.errors.length} file${status.errors.length === 1 ? "" : "s"} could not be enhanced and were left out`}
      />
    </Card>
  );
}
