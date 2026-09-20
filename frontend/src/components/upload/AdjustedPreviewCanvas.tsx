import { useEffect, useMemo, useRef, useState } from "react";
import type { AdjustmentParams } from "../../lib/adjustments";
import {
  createPreviewCache,
  renderPreviewFrame,
  type PreviewRenderCache,
} from "../../lib/previewAdjustments";

/** Longest side of the small raster used while a slider is actively being
 * dragged (rAF-throttled, redrawn continuously) -- cheap enough that a full
 * pixel pass costs well under a millisecond. The other edge follows the
 * display box's aspect ratio (see previewSizeForLongestSide), so 16:9 yields
 * exactly the old 480x270 buffer. */
const DRAFT_LONGEST_SIDE = 480;
/** Longest side of the sharper raster drawn once, a short beat after the
 * slider stops moving ("release") -- see SETTLE_DELAY_MS. Still far below
 * the base image's real resolution: this is a PREVIEW only, capped for
 * speed; Export always re-derives the exact, full-resolution result from the
 * real base image via the backend (see JobResultPanel/App.tsx's
 * onDownload/onExportBatch). 16:9 yields the old 1440x810 buffer. */
const SETTLE_LONGEST_SIDE = 1440;
/** Wait this long of genuine quiet before drawing the sharper settle frame.
 * Deliberately longer than the old 120ms so a slow or hesitant drag (micro-
 * pauses between pointer moves are common) never interrupts the live draft
 * redraw with a settle-size pass mid-gesture. */
const SETTLE_DELAY_MS = 220;
/** The compare view's fixed 16:9 preview box (also matches BillboardCanvas's
 * pre-dimensions fallback), used when the caller doesn't pass an `aspect`. */
const DEFAULT_ASPECT = { width: 16, height: 9 };

/** Buffer size sharing the display box's aspect ratio with `longestSide` as
 * its longest edge -- the preview <canvas> fills its box exactly (a canvas
 * can't be trusted to honor CSS object-fit, so the raster itself must match
 * the box's shape). The compare view passes the fixed 16:9 box; the board
 * editor passes the image's true pixel ratio so its underlay lines up 1:1
 * with the aspect-ratio box.
 *
 * Exported (alongside coverSourceRect below) purely for direct unit
 * testing without a canvas/DOM -- see AdjustedPreviewCanvas.test.ts. Not
 * used as a public API by any other module. */
export function previewSizeForLongestSide(
  longestSide: number,
  aspect: { width: number; height: number },
): { width: number; height: number } {
  if (aspect.width >= aspect.height) {
    return {
      width: longestSide,
      height: Math.round((longestSide * aspect.height) / aspect.width),
    };
  }
  return {
    width: Math.round((longestSide * aspect.width) / aspect.height),
    height: longestSide,
  };
}

export function coverSourceRect(srcW: number, srcH: number, dstW: number, dstH: number) {
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
 * Render scheduling (driven by one rAF-throttled redraw plus one
 * version-guarded settle):
 *  - While a slider is moving, redraw the small "draft" raster at most once
 *    per animation frame using ALWAYS the latest adjustment values, so rapid
 *    pointer movement can never queue a backlog of stale frames. All pixel
 *    work goes into preallocated caches (zero per-frame allocation, no GC
 *    stalls mid-drag).
 *  - A short beat after the values stop changing, redraw once more at the
 *    sharp "settle" resolution. The settle is version-guarded (a newer
 *    slider change invalidates any in-flight settle timer/rAF) so it can
 *    never paint stale values or interrupt an active drag.
 *
 * Always renders the live `<canvas>` (never a backing `<img>`), so switching
 * srcUrl can never blank the preview: while a new base image decodes, the
 * canvas keeps the previous frame; the moment the new buffers are ready, the
 * same rendering engine paints base immediately. A slider moved during that
 * decode window is applied on the first paint, so no interaction is ever
 * dropped onto a white surface.
 *
 * The raster always shares the display box's aspect ratio (see `aspect`) --
 * the board editor renders this component as BillboardCanvas's backdrop, so
 * MVP 2 board markers sit on the CURRENT adjusted image instead of the stale
 * base enhanced one.
 */
export function AdjustedPreviewCanvas({
  srcUrl,
  adjustments,
  aspect = DEFAULT_ASPECT,
  alt,
  className,
}: {
  srcUrl: string;
  adjustments: AdjustmentParams;
  /** Pixel aspect ratio of the display box the raster must fill -- 16:9 by
   * default (the compare view); the board editor passes the image's true
   * dimensions (see BillboardCanvas's aspectRatio style). */
  aspect?: { width: number; height: number };
  alt?: string;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const draftDataRef = useRef<ImageData | null>(null);
  const settleDataRef = useRef<ImageData | null>(null);
  const draftCacheRef = useRef<PreviewRenderCache | null>(null);
  const settleCacheRef = useRef<PreviewRenderCache | null>(null);
  const loadedImgRef = useRef<HTMLImageElement | null>(null);
  const [ready, setReady] = useState(false);

  const draftSize = useMemo(
    () => previewSizeForLongestSide(DRAFT_LONGEST_SIDE, aspect),
    [aspect.width, aspect.height],
  );
  const settleSize = useMemo(
    () => previewSizeForLongestSide(SETTLE_LONGEST_SIDE, aspect),
    [aspect.width, aspect.height],
  );
  // Live buffer sizes, read by the async image onload handler so a load that
  // finishes after an aspect change crops to the CURRENT ratio, not the one
  // the effect captured.
  const sizeRef = useRef({ draft: draftSize, settle: settleSize });
  sizeRef.current = { draft: draftSize, settle: settleSize };

  // Always the latest slider values, readable from inside the rAF/timer
  // callbacks below without those callbacks being re-created every render.
  const latestParamsRef = useRef(adjustments);
  latestParamsRef.current = adjustments;
  const rafPendingRef = useRef(false);
  // True when a settle (the sharper 1440x810 pass) has been requested while
  // a draft frame was still in flight -- piggybacked right after it.
  const settleQueuedRef = useRef(false);
  const settleTimerRef = useRef<number | undefined>(undefined);
  // Monotonic guard: every slider change bumps it, so any settle timer or
  // rAF captured before that change sees a stale version and bails.
  const settleVersionRef = useRef(0);
  // Monotonic per-source token: every srcUrl change bumps it, so a decode
  // started for an earlier source can never paint (or re-slice) after the
  // source has been switched away -- "switching invalidates pending preview
  // work of the previous image".
  const srcVersionRef = useRef(0);

  const getCtx = (): CanvasRenderingContext2D | null => {
    if (ctxRef.current) return ctxRef.current;
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const ctx = canvas.getContext("2d");
    if (ctx) ctxRef.current = ctx;
    return ctx;
  };

  /** Creates the two reusable render caches (draft + settle resolution) the
   * first time a base image is ready. Each cache wraps the same buffer that
   * its ImageData uses (no copy per frame). `sizes` defaults to the render's
   * draft/settle sizes; the decode path passes the live sizeRef so a decode
   * that finishes after an aspect change builds caches matching the buffers
   * it just sliced (dims can never disagree between source and cache). */
  const ensureCaches = (
    sizes?: { draft: { width: number; height: number }; settle: { width: number; height: number } },
  ): boolean => {
    if (!draftCacheRef.current || !settleCacheRef.current) {
      const ctx = getCtx();
      if (!ctx || !draftDataRef.current || !settleDataRef.current) return false;
      const s = sizes ?? { draft: draftSize, settle: settleSize };
      draftCacheRef.current = createPreviewCache(ctx, s.draft.width, s.draft.height);
      settleCacheRef.current = createPreviewCache(ctx, s.settle.width, s.settle.height);
    }
    return true;
  };

  const drawFrame = (source: ImageData, cache: PreviewRenderCache, params: AdjustmentParams) => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;
    if (canvas.width !== cache.width) canvas.width = cache.width;
    if (canvas.height !== cache.height) canvas.height = cache.height;
    ctx.putImageData(renderPreviewFrame(source, params, cache), 0, 0);
  };

  const drawSettleIfQueued = () => {
    if (!settleQueuedRef.current) return;
    settleQueuedRef.current = false;
    if (settleDataRef.current && settleCacheRef.current) {
      drawFrame(settleDataRef.current, settleCacheRef.current, latestParamsRef.current);
    }
  };

  // Load + pre-crop the base image once per srcUrl into two immutable
  // pixel buffers (object-fit: cover, matching what a plain <img> would
  // have shown). The base image itself is never mutated -- every
  // adjustment render below starts fresh from these cached buffers, which
  // is what makes Reset (and every other slider change) exact and
  // reversible. Any existing caches from a previous srcUrl are dropped so
  // the new buffers get fresh scratch.
  useEffect(() => {
    const srcVersion = ++srcVersionRef.current;
    setReady(false);
    draftDataRef.current = null;
    settleDataRef.current = null;
    draftCacheRef.current = null;
    settleCacheRef.current = null;
    ctxRef.current = null;
    loadedImgRef.current = null;
    const img = new Image();
    img.onload = () => {
      if (srcVersionRef.current !== srcVersion) return;
      const sizes = sizeRef.current;
      loadedImgRef.current = img;
      draftDataRef.current = drawCoverImageData(img, sizes.draft.width, sizes.draft.height);
      settleDataRef.current = drawCoverImageData(img, sizes.settle.width, sizes.settle.height);
      // First paint happens HERE, straight from the decode, through the very
      // same rendering engine the sliders use -- the "base renders
      // immediately" moment. This is deliberately NOT delegated to the
      // [ready] effect: `ready` only gates the slider-driven redraws below,
      // and the first paint must not depend on the state flip succeeding.
      // It also applies `latestParamsRef.current`, so any slider value moved
      // during the decode window lands on this very first frame (never on a
      // white/blank surface).
      if (ensureCaches(sizes) && settleDataRef.current && settleCacheRef.current) {
        drawFrame(settleDataRef.current, settleCacheRef.current, latestParamsRef.current);
      }
      setReady(true);
    };
    img.src = srcUrl;
    return () => {
      srcVersionRef.current += 1;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [srcUrl]);

  // Unlike the load effect above, this handles an aspect-only change after
  // the base image has already decoded (board editing waits on the image's
  // real pixel dimensions, which arrive async while the box sits at the 16:9
  // fallback). Re-slice the buffers to the new ratio immediately instead of
  // flashing back through the plain, unadjusted <img> fallback -- the same
  // srcUrl is still loaded, so it's just a re-crop plus one repaint.
  useEffect(() => {
    const img = loadedImgRef.current;
    if (!img) return;
    draftDataRef.current = drawCoverImageData(img, draftSize.width, draftSize.height);
    settleDataRef.current = drawCoverImageData(img, settleSize.width, settleSize.height);
    draftCacheRef.current = null;
    settleCacheRef.current = null;
    if (ensureCaches() && settleDataRef.current && settleCacheRef.current) {
      drawFrame(settleDataRef.current, settleCacheRef.current, latestParamsRef.current);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aspect.width, aspect.height]);

  // Instant first paint once the base image is ready (sharp "settle"
  // resolution -- there's no drag in progress yet). The decode path above
  // already paints straight from onload; this is an idempotent safety net so
  // a source never goes unpainted even if the state flip above is swallowed.
  useEffect(() => {
    if (!ready) return;
    if (!ensureCaches()) return;
    if (settleDataRef.current && settleCacheRef.current) {
      drawFrame(settleDataRef.current, settleCacheRef.current, latestParamsRef.current);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // The actual fix: rAF-throttled draft redraw on every adjustment change
  // (at most one draw per animation frame, always the latest values, so
  // rapid pointer movement can never queue a backlog of stale updates),
  // plus one sharper "settle" redraw after a real quiet period, guarded so
  // a superseded timer/rAF can never paint stale values mid-drag.
  useEffect(() => {
    if (!ready) return;
    if (!ensureCaches()) return;

    if (!rafPendingRef.current) {
      rafPendingRef.current = true;
      requestAnimationFrame(() => {
        rafPendingRef.current = false;
        if (draftDataRef.current && draftCacheRef.current) {
          drawFrame(draftDataRef.current, draftCacheRef.current, latestParamsRef.current);
        }
        // A settle requested while this frame was in flight draws right
        // after it (still the latest values).
        drawSettleIfQueued();
      });
    }

    const version = ++settleVersionRef.current;
    if (settleTimerRef.current) window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = window.setTimeout(() => {
      settleTimerRef.current = undefined;
      if (settleVersionRef.current !== version) return; // superseded -- newer slider change
      settleQueuedRef.current = true;
      if (!rafPendingRef.current) {
        rafPendingRef.current = true;
        requestAnimationFrame(() => {
          rafPendingRef.current = false;
          drawSettleIfQueued();
        });
      }
    }, SETTLE_DELAY_MS);

    return () => {
      if (settleTimerRef.current) window.clearTimeout(settleTimerRef.current);
      settleVersionRef.current += 1;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adjustments, ready]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={alt ?? "Enhanced 4K result"}
      className={className ?? "h-full w-full"}
    />
  );
}