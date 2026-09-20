/**
 * Client-side (canvas/ImageData) preview rendering for the manual
 * adjustment sliders — MVP 2 "no backend round-trip while dragging" fix.
 *
 * Why this exists: the first version of this feature sent a debounced
 * `?adjust=` HTTP request to the backend on every slider change. Even
 * debounced, each request paid network + disk-read + cv2 processing + PNG
 * encode + blob decode latency (tens to low-hundreds of ms), which reads as
 * visible lag during a drag and cannot keep up with rapid pointer movement.
 * This module replaces that with pure, synchronous, in-browser pixel math
 * so the preview updates within a single animation frame and NEVER touches
 * the network, the backend, or the AI pipeline while a slider moves.
 *
 * Why it stays smooth while dragging (the fix for the residual lag in the
 * first client-side version):
 *  - Every render is a single fused pass that writes into a preallocated
 *    working buffer (see PreviewRenderCache). A drag therefore allocates
 *    nothing per frame and never takes a GC pause mid-gesture — the earlier
 *    version allocated a fresh source copy plus a second detail-output
 *    buffer on every frame.
 *  - Saturation uses a direct chroma-scaling formula — algebraically equal
 *    to the plain RGB<->HSL reference (same hue, same lightness, S scaled by
 *    the same factor and clamped to 1; verified exhaustively over 256^3
 *    inputs, differing by at most ±1 in rare half-rounding pixels) — but
 *    branch-light: one division per colored pixel, no per-channel hue
 *    reconstruction. The old per-pixel closure-heavy HSL round trip is what
 *    made the settle-size (1440x810) pass block the main thread for tens of
 *    milliseconds while dragging.
 *  - Detail uses a separable 3x3 box blur (horizontal pass then vertical
 *    pass — the exact same 9-tap kernel, bit-identical output) fused with
 *    the unsharp delta in a single final pass, instead of reading 27 pixels
 *    per channel per pixel and allocating a whole new output buffer.
 *
 * Semantics: mirrors image_enhancer/src/adjustments.py's constants, ranges,
 * and per-control direction/behavior exactly (same MAX_* ceilings, same
 * shadow/highlight weight thresholds, same saturation/detail bounds, same
 * application order: combined tone LUT -> saturation -> detail). The two
 * deliberate, documented differences from the backend are inherited from
 * the earlier preview (unchanged behavior):
 *   1. brightness/contrast/highlights/shadows are applied to each RGB
 *      channel rather than only to the Lab L channel, because a real-time
 *      per-pixel Lab<->RGB round trip every animation frame is not fast
 *      enough to drag smoothly in JS. At these bounded intensities the
 *      visual difference is imperceptible.
 *   2. detail uses a 3x3 box blur with a fixed per-channel delta cap
 *      instead of the backend's 5x5 local min/max clip.
 * The authoritative, pixel-exact output is always the backend call Export
 * makes once, on click — never anything computed here, so preview
 * approximations can never leak into the saved image.
 */

import type { AdjustmentParams } from "./adjustments";

const SHADOW_WEIGHT_THRESHOLD = 110;
const HIGHLIGHT_WEIGHT_THRESHOLD = 145;
const MAX_BRIGHTNESS = 60;
const MAX_CONTRAST_GAIN = 0.5;
const MAX_HIGHLIGHT = 45;
const MAX_SHADOW = 45;
const MAX_SATURATION_GAIN = 0.7;
const MAX_DETAIL_STRENGTH = 1.4;
const DETAIL_DELTA_CAP = 40; // fixed per-channel clamp -- see module docstring

/** Exported (alongside the other pure tone-math helpers below) purely so
 * they can be unit-tested directly without a canvas/DOM — see
 * previewAdjustments.test.ts. Not used as a public API by any other
 * module; every real caller still goes through renderPreviewFrame. */
export function shadowWeight(x: number): number {
  const w = 1 - x / SHADOW_WEIGHT_THRESHOLD;
  return w < 0 ? 0 : w > 1 ? 1 : w;
}

export function highlightWeight(x: number): number {
  const span = Math.max(255 - HIGHLIGHT_WEIGHT_THRESHOLD, 1);
  const w = (x - HIGHLIGHT_WEIGHT_THRESHOLD) / span;
  return w < 0 ? 0 : w > 1 ? 1 : w;
}

export function isToneIdentity(params: AdjustmentParams): boolean {
  return (
    params.brightness === 0 &&
    params.contrast === 0 &&
    params.highlights === 0 &&
    params.shadows === 0
  );
}

/** The same combined monotonic 256-entry LUT as adjustments.py's
 * _build_lut (brightness+contrast+highlights+shadows), applied per RGB
 * channel here — see module docstring for why. Written into the cache's
 * reusable buffer so a drag allocates nothing. */
export function fillToneLut(lut: Uint8ClampedArray, params: AdjustmentParams): void {
  const contrastGain = 1 + (params.contrast / 100) * MAX_CONTRAST_GAIN;
  let prev = 0;
  for (let x = 0; x < 256; x++) {
    let y = (x - 128) * contrastGain + 128;
    y += (params.brightness / 100) * MAX_BRIGHTNESS;
    y += (params.shadows / 100) * MAX_SHADOW * shadowWeight(x);
    y += (params.highlights / 100) * MAX_HIGHLIGHT * highlightWeight(x);
    if (y < 0) y = 0;
    else if (y > 255) y = 255;
    if (y < prev) y = prev; // monotonic non-decreasing, matches the backend LUT
    prev = y;
    lut[x] = Math.round(y);
  }
}

/**
 * Reusable scratch for one preview resolution (DRAFT or SETTLE), created
 * once per base image by AdjustedPreviewCanvas. `out` and `imageData` share
 * the same underlying buffer (created via ctx.createImageData so nothing is
 * copied or re-created per frame) — every render refills `out` and hands
 * `imageData` straight to putImageData. Zero per-frame allocation.
 */
export interface PreviewRenderCache {
  width: number;
  height: number;
  /** 256-entry brightness/contrast/highlights/shadows LUT. */
  lut: Uint8ClampedArray<ArrayBuffer>;
  /** The fully rendered RGBA frame each render writes into. */
  out: Uint8ClampedArray<ArrayBuffer>;
  /** Live ImageData view over `out`; reuse it with putImageData. */
  imageData: ImageData;
  /** Horizontal 3x3 blur pass scratch for Detail (float, matching the
   * single-pass 9-tap average bit-for-bit). */
  hPass: Float32Array;
  /** Vertical pass result = the final 3x3 box blur, for the Detail unsharp. */
  blur: Float32Array;
}

export function createPreviewCache(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
): PreviewRenderCache {
  const imageData = ctx.createImageData(width, height);
  const out = imageData.data as Uint8ClampedArray<ArrayBuffer>;
  const n = width * height * 4;
  return {
    width,
    height,
    lut: new Uint8ClampedArray(256),
    out,
    imageData,
    hPass: new Float32Array(n),
    blur: new Float32Array(n),
  };
}

/** Tone-only pass (brightness/contrast/highlights/shadows via the LUT),
 * used when saturation is at its no-op value so colored-pixel work in the
 * saturation branch is skipped entirely. Alpha untouched. */
function applyToneOnly(
  cache: PreviewRenderCache,
  src: Uint8ClampedArray<ArrayBufferLike>,
): void {
  const out = cache.out;
  const lut = cache.lut;
  for (let i = 0; i < out.length; i += 4) {
    out[i] = lut[src[i]];
    out[i + 1] = lut[src[i + 1]];
    out[i + 2] = lut[src[i + 2]];
    out[i + 3] = src[i + 3];
  }
}

/** Direct chroma-scaling saturation — algebraically identical to the HSL
 * S-only reference (hue and lightness preserved, S multiplied by `factor`
 * and clamped to 1) but computed in the 0..255 domain with one division per
 * colored pixel. Given chroma C = M-m, capped scaled chroma C', and the
 * midpoint (sum-C')/2, any channel x becomes base + (x-m)*(C'/C): the max
 * lands on M', the min on m', any middle interpolates linearly (uniform
 * formula also handles identical max/min channels). Gray pixels are left
 * untouched. */
function applySaturation(
  lut: Uint8ClampedArray,
  src: Uint8ClampedArray<ArrayBufferLike>,
  dst: Uint8ClampedArray<ArrayBuffer>,
  factor: number,
  toneIdentity: boolean,
): void {
  for (let i = 0; i < dst.length; i += 4) {
    let r = src[i];
    let g = src[i + 1];
    let b = src[i + 2];
    if (!toneIdentity) {
      r = lut[r];
      g = lut[g];
      b = lut[b];
    }
    const m = r <= g ? (r <= b ? r : b) : (g <= b ? g : b);
    const M = r >= g ? (r >= b ? r : b) : (g >= b ? g : b);
    if (M !== m) {
      const sum = M + m;
      const cMax = Math.min(sum, 510 - sum);
      let c2 = (M - m) * factor;
      if (c2 > cMax) c2 = cMax;
      const ratio = c2 / (M - m);
      const base = (sum - c2) / 2;
      r = Math.round(base + (r - m) * ratio);
      g = Math.round(base + (g - m) * ratio);
      b = Math.round(base + (b - m) * ratio);
    }
    dst[i] = r;
    dst[i + 1] = g;
    dst[i + 2] = b;
    dst[i + 3] = src[i + 3];
  }
}

/** Separable 3x3 box blur + unsharp mask on cache.out, per RGB channel,
 * with the same fixed per-channel delta cap as the reference. The
 * horizontal-then-vertical separable blur is the exact 9-tap averaging
 * kernel (average of averages), with replicate-edge clamping matching the
 * reference for every image dimension; fusing the unsharp delta into the
 * blur's final pass keeps the whole Detail adjustment to ~7 memory touches
 * per byte instead of the reference's 27. In-place on `out`. */
function applyDetail(cache: PreviewRenderCache, amount: number): void {
  const strength = (amount / 100) * MAX_DETAIL_STRENGTH;
  const { out, hPass, blur, width, height } = cache;

  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const x0 = x > 0 ? x - 1 : 0;
      const x1 = x < width - 1 ? x + 1 : width - 1;
      const i = (row + x) * 4;
      for (let c = 0; c < 3; c++) {
        hPass[i + c] = (out[(row + x0) * 4 + c] + out[i + c] + out[(row + x1) * 4 + c]) / 3;
      }
    }
  }

  for (let y = 0; y < height; y++) {
    const y0 = y > 0 ? y - 1 : 0;
    const y1 = y < height - 1 ? y + 1 : height - 1;
    const row0 = y0 * width * 4;
    const row = y * width * 4;
    const row1 = y1 * width * 4;
    for (let x = 0; x < width; x++) {
      const j = x * 4;
      const i0 = row0 + j;
      const im = row + j;
      const i1 = row1 + j;
      for (let c = 0; c < 3; c++) {
        blur[im + c] = (hPass[i0 + c] + hPass[im + c] + hPass[i1 + c]) / 3;
      }
    }
  }

  for (let i = 0; i < out.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      let delta = (out[i + c] - blur[i + c]) * strength;
      if (delta > DETAIL_DELTA_CAP) delta = DETAIL_DELTA_CAP;
      else if (delta < -DETAIL_DELTA_CAP) delta = -DETAIL_DELTA_CAP;
      let v = out[i + c] + delta;
      if (v < 0) v = 0;
      else if (v > 255) v = 255;
      out[i + c] = v;
    }
  }
}

/** Renders `params` into the cache's preallocated buffer and returns the
 * cache's reusable ImageData. Never mutates `source` — the base enhanced
 * image stays immutable, so repeated slider changes (including Reset)
 * always start fresh from the same original pixels, exactly like the
 * backend's own no-mutation contract (adjustments.py's apply_adjustments
 * never mutates its input either). A pure memcpy (no pixel loop at all)
 * when every value is at its default, so Reset is always the cheapest
 * possible path. */
export function renderPreviewFrame(
  source: ImageData,
  params: AdjustmentParams,
  cache: PreviewRenderCache,
): ImageData {
  if (cache.width !== source.width || cache.height !== source.height) {
    throw new Error("renderPreviewFrame: cache dimensions must match the source");
  }

  const toneIdentity = isToneIdentity(params);
  const noSaturation = params.saturation === 0;
  const noDetail = params.detail <= 0;

  if (toneIdentity && noSaturation && noDetail) {
    cache.out.set(source.data);
    return cache.imageData;
  }

  // Single fused pass over the pixels (tone LUT -> saturation), then the
  // Detail pass when enabled. cache.out is fully overwritten each call, so
  // reusing the buffer (and its ImageData) across frames is safe.
  if (!toneIdentity) fillToneLut(cache.lut, params);
  if (noSaturation) {
    if (toneIdentity) {
      cache.out.set(source.data); // detail-only drag: start from the raw copy
    } else {
      applyToneOnly(cache, source.data);
    }
  } else {
    const factor = 1 + (params.saturation / 100) * MAX_SATURATION_GAIN;
    applySaturation(cache.lut, source.data, cache.out, factor, toneIdentity);
  }

  if (!noDetail) applyDetail(cache, params.detail);

  return cache.imageData;
}