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
 * Semantics: this mirrors image_enhancer/src/adjustments.py's constants,
 * ranges, and per-control direction/behavior exactly (same MAX_* ceilings,
 * same shadow/highlight weight thresholds, same saturation/detail bounds) —
 * "same adjustment semantics" as the backend. The one deliberate difference
 * is colorspace: the backend applies brightness/contrast/highlights/
 * shadows to the Lab L (luminance) channel only; this preview applies the
 * same tone curve to each RGB channel independently, because a real-time
 * per-pixel Lab<->RGB round trip on a multi-megapixel canvas, every
 * animation frame, is not fast enough for smooth dragging in JS. For these
 * bounded, moderate adjustments the visual difference is imperceptible;
 * the authoritative, pixel-exact output is always the backend call Export
 * makes once, on click — never anything computed here. Detail similarly
 * uses a fast 3x3 box-blur unsharp mask with a fixed delta cap instead of
 * the backend's local min/max clip, for the same real-time-performance
 * reason.
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

function shadowWeight(x: number): number {
  const w = 1 - x / SHADOW_WEIGHT_THRESHOLD;
  return w < 0 ? 0 : w > 1 ? 1 : w;
}

function highlightWeight(x: number): number {
  const span = Math.max(255 - HIGHLIGHT_WEIGHT_THRESHOLD, 1);
  const w = (x - HIGHLIGHT_WEIGHT_THRESHOLD) / span;
  return w < 0 ? 0 : w > 1 ? 1 : w;
}

/** The same combined monotonic 256-entry LUT as adjustments.py's
 * _build_lut (brightness+contrast+highlights+shadows), applied per RGB
 * channel here — see module docstring for why. */
export function buildToneLut(params: AdjustmentParams): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256);
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
  return lut;
}

function isToneIdentity(params: AdjustmentParams): boolean {
  return params.brightness === 0 && params.contrast === 0
    && params.highlights === 0 && params.shadows === 0;
}

/** In-place RGB->HSL->RGB saturation scale (H and L untouched — hue and
 * brightness are never affected by this control, matching the backend's
 * HSV-S-only behavior). */
function applySaturationInPlace(data: Uint8ClampedArray<ArrayBufferLike>, amount: number): void {
  if (amount === 0) return;
  const factor = 1 + (amount / 100) * MAX_SATURATION_GAIN;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) continue; // already gray -- nothing to scale
    let s = l > 0.5 ? (max - min) / (2 - max - min) : (max - min) / (max + min);
    s = Math.min(1, Math.max(0, s * factor));
    const d = max - min;
    let h: number;
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const hue2rgb = (t: number): number => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    data[i] = Math.round(hue2rgb(h + 1 / 3) * 255);
    data[i + 1] = Math.round(hue2rgb(h) * 255);
    data[i + 2] = Math.round(hue2rgb(h - 1 / 3) * 255);
  }
}

/** A fast 3x3-box-blur unsharp mask, per RGB channel, with a fixed delta
 * cap (see module docstring for why this differs from the backend's local
 * min/max clip). No-op at amount<=0. */
function applyDetail(
  src: Uint8ClampedArray<ArrayBufferLike>, width: number, height: number, amount: number,
): Uint8ClampedArray<ArrayBufferLike> {
  if (amount <= 0) return src;
  const strength = (amount / 100) * MAX_DETAIL_STRENGTH;
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < height; y++) {
    const y0 = y > 0 ? y - 1 : 0;
    const y1 = y < height - 1 ? y + 1 : height - 1;
    for (let x = 0; x < width; x++) {
      const x0 = x > 0 ? x - 1 : 0;
      const x1 = x < width - 1 ? x + 1 : width - 1;
      const idx = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        sum += src[(y0 * width + x0) * 4 + c] + src[(y0 * width + x) * 4 + c] + src[(y0 * width + x1) * 4 + c];
        sum += src[(y * width + x0) * 4 + c] + src[(y * width + x) * 4 + c] + src[(y * width + x1) * 4 + c];
        sum += src[(y1 * width + x0) * 4 + c] + src[(y1 * width + x) * 4 + c] + src[(y1 * width + x1) * 4 + c];
        const blur = sum / 9;
        const orig = src[idx + c];
        let delta = (orig - blur) * strength;
        if (delta > DETAIL_DELTA_CAP) delta = DETAIL_DELTA_CAP;
        else if (delta < -DETAIL_DELTA_CAP) delta = -DETAIL_DELTA_CAP;
        let v = orig + delta;
        if (v < 0) v = 0;
        else if (v > 255) v = 255;
        out[idx + c] = v;
      }
      out[idx + 3] = src[idx + 3];
    }
  }
  return out;
}

/** Renders `params` onto a COPY of `source` and returns the new ImageData.
 * Never mutates `source` — the base enhanced image stays immutable, so
 * repeated slider changes (including Reset) always start fresh from the
 * same original pixels, exactly like the backend's own no-mutation
 * contract (adjustments.py's apply_adjustments never mutates its input
 * either). A true identity copy (no pixel loop at all) when every value is
 * at its default, so Reset is always the cheapest possible path. */
export function renderPreviewFrame(source: ImageData, params: AdjustmentParams): ImageData {
  const toneIdentity = isToneIdentity(params);
  const noSaturation = params.saturation === 0;
  const noDetail = params.detail <= 0;

  if (toneIdentity && noSaturation && noDetail) {
    return new ImageData(
      new Uint8ClampedArray(source.data) as Uint8ClampedArray<ArrayBuffer>,
      source.width, source.height,
    );
  }

  let data: Uint8ClampedArray<ArrayBufferLike> = new Uint8ClampedArray(source.data);
  if (!toneIdentity) {
    const lut = buildToneLut(params);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = lut[data[i]];
      data[i + 1] = lut[data[i + 1]];
      data[i + 2] = lut[data[i + 2]];
    }
  }
  if (!noSaturation) applySaturationInPlace(data, params.saturation);
  if (!noDetail) data = applyDetail(data, source.width, source.height, params.detail);

  // Safe: every Uint8ClampedArray constructed in this module is freshly
  // allocated (never backed by a SharedArrayBuffer) -- ImageData's type
  // just doesn't express that distinction the same way TS's lib.dom does.
  return new ImageData(data as Uint8ClampedArray<ArrayBuffer>, source.width, source.height);
}
