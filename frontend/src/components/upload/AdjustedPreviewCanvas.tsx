import { useEffect, useRef, useState } from "react";
import type { AdjustmentParams } from "../../lib/adjustments";
import { renderPreviewFrame } from "../../lib/previewAdjustments";

/** Small raster used while a slider is actively being dragged (rAF-throttled,
 * redrawn continuously) -- cheap enough that a full pixel pass costs well
 * under a millisecond. */
const DRAFT_SIZE = { width: 480, height: 270 };
/** Sharper raster drawn once, a short beat after the slider stops moving
 * ("release") -- see SETTLE_DELAY_MS. Still far below the base image's real
 * resolution: this is a PREVIEW only, capped for speed; Export always
 * re-derives the exact, full-resolution result from the real base image via
 * the backend (see JobResultPanel/App.tsx's onDownload/onExportBatch). */
const SETTLE_SIZE = { width: 1440, height: 810 };
const SETTLE_DELAY_MS = 120;

function coverSourceRect(srcW: number, srcH: number, dstW: number, dstH: number) {
  const srcRatio = srcW / srcH;
  const dstRatio = dstW / dstH;
  let sx = 0, sy = 0, sw = srcW, sh = srcH;
  if (srcRatio > dstRatio) {
    sw = srcH * dstRatio;
    sx = (srcW - sw) / 2;
  } else {
    sh = srcW / dstRatio;
    sy = (srcH - sh) / 2;
  }
  return { sx, sy, sw, sh };
}

/** Crops `img` to `dstW`x`dstH` the same way CSS `object-fit: cover` would,
 * then reads it back as a plain, immutable ImageData buffer. */
function drawCoverImageData(img: HTMLImageElement, dstW: number, dstH: number): ImageData {
  const { sx, sy, sw, sh } = coverSourceRect(img.naturalWidth, img.naturalHeight, dstW, dstH);
  const canvas = document.createElement("canvas");
  canvas.width = dstW;
  canvas.height = dstH;
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, dstW, dstH);
  return ctx.getImageData(0, 0, dstW, dstH);
}

/**
 * MVP 2 manual-adjustment live preview: renders `srcUrl` (the immutable
 * base enhanced image) through the client-side pixel math in
 * lib/previewAdjustments.ts, entirely in the browser -- no network request,
 * no backend call, no AI pipeline re-invocation, on every `adjustments`
 * change. See that module's own docstring for why this replaced the
 * earlier debounced-backend-fetch preview (it was the actual source of
 * drag lag: every slider tick paid a full HTTP + cv2 round trip).
 *
 * Renders a plain `<img>` (identical to the old behavior) until the base
 * image has finished loading/decoding, then swaps to a live `<canvas>`.
 * While a slider is actively moving, redraws a small, cheap "draft" raster
 * at most once per animation frame (rAF-throttled, always the LATEST
 * adjustment values -- never a queued backlog of stale ones); a short beat
 * after it stops, redraws once more at a sharper "settle" resolution.
 */
export function AdjustedPreviewCanvas({
  srcUrl,
  adjustments,
  alt,
  className,
}: {
  srcUrl: string;
  adjustments: AdjustmentParams;
  alt?: string;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draftDataRef = useRef<ImageData | null>(null);
  const settleDataRef = useRef<ImageData | null>(null);
  const [ready, setReady] = useState(false);

  // Always the latest slider values, readable from inside the rAF callback
  // below without that callback needing to be re-created every render.
  const latestParamsRef = useRef(adjustments);
  latestParamsRef.current = adjustments;
  const rafPendingRef = useRef(false);
  const settleTimerRef = useRef<number | undefined>(undefined);

  const drawFrame = (source: ImageData, size: { width: number; height: number }, params: AdjustmentParams) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (canvas.width !== size.width) canvas.width = size.width;
    if (canvas.height !== size.height) canvas.height = size.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(renderPreviewFrame(source, params), 0, 0);
  };

  // Load + pre-crop the base image once per srcUrl into two immutable
  // pixel buffers (object-fit: cover, matching what a plain <img> would
  // have shown). The base image itself is never mutated -- every
  // adjustment render below starts fresh from these cached buffers, which
  // is what makes Reset (and every other slider change) exact and
  // reversible.
  useEffect(() => {
    let cancelled = false;
    setReady(false);
    draftDataRef.current = null;
    settleDataRef.current = null;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      draftDataRef.current = drawCoverImageData(img, DRAFT_SIZE.width, DRAFT_SIZE.height);
      settleDataRef.current = drawCoverImageData(img, SETTLE_SIZE.width, SETTLE_SIZE.height);
      setReady(true);
    };
    img.src = srcUrl;
    return () => {
      cancelled = true;
    };
  }, [srcUrl]);

  // Instant first paint once the base image is ready (sharp "settle"
  // resolution -- there's no drag in progress yet).
  useEffect(() => {
    if (!ready || !settleDataRef.current) return;
    drawFrame(settleDataRef.current, SETTLE_SIZE, latestParamsRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // The actual fix: rAF-throttled draft redraw on every adjustment change
  // (at most one draw per animation frame, always the latest values, so
  // rapid pointer movement can never queue a backlog of stale updates),
  // plus one sharper "settle" redraw a short beat after changes stop.
  useEffect(() => {
    if (!ready || !draftDataRef.current) return;
    if (!rafPendingRef.current) {
      rafPendingRef.current = true;
      requestAnimationFrame(() => {
        rafPendingRef.current = false;
        if (draftDataRef.current) drawFrame(draftDataRef.current, DRAFT_SIZE, latestParamsRef.current);
      });
    }
    if (settleTimerRef.current) window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = window.setTimeout(() => {
      if (settleDataRef.current) drawFrame(settleDataRef.current, SETTLE_SIZE, latestParamsRef.current);
    }, SETTLE_DELAY_MS);
    return () => {
      if (settleTimerRef.current) window.clearTimeout(settleTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adjustments, ready]);

  if (!ready) {
    return (
      <img
        src={srcUrl}
        alt={alt ?? "Enhanced 4K result"}
        className={className ?? "h-full w-full object-cover"}
        draggable={false}
      />
    );
  }

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={alt ?? "Enhanced 4K result"}
      className={className ?? "h-full w-full"}
    />
  );
}
