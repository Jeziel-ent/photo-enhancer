import { describe, expect, it } from "vitest";
import { ADJUSTMENT_RANGE, DEFAULT_ADJUSTMENTS, isDefaultAdjustments } from "./adjustments";
import type { AdjustmentParams } from "./adjustments";

describe("isDefaultAdjustments", () => {
  it("is true for the DEFAULT_ADJUSTMENTS constant itself", () => {
    expect(isDefaultAdjustments(DEFAULT_ADJUSTMENTS)).toBe(true);
  });

  it("is true for a fresh all-zero object (reset behavior)", () => {
    const reset: AdjustmentParams = {
      brightness: 0,
      contrast: 0,
      highlights: 0,
      shadows: 0,
      saturation: 0,
      detail: 0,
    };
    expect(isDefaultAdjustments(reset)).toBe(true);
  });

  it("is false when exactly one value differs from default", () => {
    for (const key of Object.keys(DEFAULT_ADJUSTMENTS) as (keyof AdjustmentParams)[]) {
      const params = { ...DEFAULT_ADJUSTMENTS, [key]: DEFAULT_ADJUSTMENTS[key] + 1 };
      expect(isDefaultAdjustments(params), `key=${key}`).toBe(false);
    }
  });

  it("is false for every field nudged away from default at once", () => {
    const params: AdjustmentParams = {
      brightness: 20,
      contrast: -10,
      highlights: 5,
      shadows: -5,
      saturation: 30,
      detail: 15,
    };
    expect(isDefaultAdjustments(params)).toBe(false);
  });

  // This is the exact gate App.tsx/api.ts use to decide whether `adjust`
  // is included in an export payload at all -- see App.tsx's
  // handleExportBatch and api.ts's exportBatch/downloadIndividualResult.
  it("drives the export-payload-omission contract: default -> omit, non-default -> include", () => {
    const buildPayload = (adjust: AdjustmentParams) =>
      isDefaultAdjustments(adjust) ? {} : { adjust: { ...adjust } };

    expect(buildPayload(DEFAULT_ADJUSTMENTS)).toEqual({});
    const custom = { ...DEFAULT_ADJUSTMENTS, brightness: 20 };
    expect(buildPayload(custom)).toEqual({ adjust: custom });
  });
});

describe("ADJUSTMENT_RANGE", () => {
  it("bounds every symmetric slider to -100..100", () => {
    for (const key of ["brightness", "contrast", "highlights", "shadows", "saturation"] as const) {
      expect(ADJUSTMENT_RANGE[key]).toEqual({ min: -100, max: 100 });
    }
  });

  it("bounds detail to 0..100 (never negative, per adjustments.py's clamp)", () => {
    expect(ADJUSTMENT_RANGE.detail).toEqual({ min: 0, max: 100 });
  });
});
