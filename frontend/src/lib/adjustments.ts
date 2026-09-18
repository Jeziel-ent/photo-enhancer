/**
 * Manual post-processing adjustment types shared by every place the
 * frontend renders or exports an adjusted image (JobResultPanel's live
 * preview, api.ts's download/export requests, desktop.ts's native save
 * bridge). The actual pixel math lives ONLY on the backend
 * (image_enhancer/src/adjustments.py, via backend/adjustment_overlay.py) —
 * this file holds no image-processing logic of its own, so preview and
 * export can never compute a different result: both are the same backend
 * call, just made at different times (see JobResultPanel's debounced
 * preview fetch vs. its Export button).
 */

export interface AdjustmentParams {
  /** -100..100, default 0. */
  brightness: number;
  /** -100..100, default 0. */
  contrast: number;
  /** -100..100, default 0. Positive brightens bright tones, negative
   * recovers/protects them. */
  highlights: number;
  /** -100..100, default 0. Positive reveals dark tones, negative deepens
   * them. */
  shadows: number;
  /** -100..100, default 0. */
  saturation: number;
  /** 0..100, default 0 — only ever adds clarity, never removes it. */
  detail: number;
}

export const ADJUSTMENT_RANGE: Record<keyof AdjustmentParams, { min: number; max: number }> = {
  brightness: { min: -100, max: 100 },
  contrast: { min: -100, max: 100 },
  highlights: { min: -100, max: 100 },
  shadows: { min: -100, max: 100 },
  saturation: { min: -100, max: 100 },
  detail: { min: 0, max: 100 },
};

export const DEFAULT_ADJUSTMENTS: AdjustmentParams = {
  brightness: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  saturation: 0,
  detail: 0,
};

/** True when every value is at its default (0) — i.e. this would be a
 * total no-op if sent to the backend. Used to skip sending a redundant
 * `?adjust=` query/body field at all. */
export function isDefaultAdjustments(params: AdjustmentParams): boolean {
  return (Object.keys(DEFAULT_ADJUSTMENTS) as (keyof AdjustmentParams)[]).every(
    (key) => params[key] === DEFAULT_ADJUSTMENTS[key],
  );
}
