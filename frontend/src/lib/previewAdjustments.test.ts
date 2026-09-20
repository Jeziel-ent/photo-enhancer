import { describe, expect, it } from "vitest";
import { DEFAULT_ADJUSTMENTS } from "./adjustments";
import type { AdjustmentParams } from "./adjustments";
import { fillToneLut, highlightWeight, isToneIdentity, shadowWeight } from "./previewAdjustments";

describe("shadowWeight", () => {
  it("is 1 at x=0 (fully in shadow range)", () => {
    expect(shadowWeight(0)).toBe(1);
  });

  it("is 0 at and beyond the threshold (110)", () => {
    expect(shadowWeight(110)).toBe(0);
    expect(shadowWeight(255)).toBe(0);
  });

  it("clamps to [0, 1] for out-of-range inputs", () => {
    expect(shadowWeight(-50)).toBe(1); // would be >1 unclamped
    expect(shadowWeight(9999)).toBe(0);
  });

  it("decreases monotonically as x increases toward the threshold", () => {
    expect(shadowWeight(20)).toBeGreaterThan(shadowWeight(60));
    expect(shadowWeight(60)).toBeGreaterThan(shadowWeight(100));
  });
});

describe("highlightWeight", () => {
  it("is 0 at and below the threshold (145)", () => {
    expect(highlightWeight(145)).toBe(0);
    expect(highlightWeight(0)).toBe(0);
  });

  it("is 1 at 255 (the brightest possible value)", () => {
    expect(highlightWeight(255)).toBe(1);
  });

  it("clamps to [0, 1] for out-of-range inputs", () => {
    expect(highlightWeight(9999)).toBe(1);
    expect(highlightWeight(-100)).toBe(0);
  });

  it("increases monotonically as x increases toward 255", () => {
    expect(highlightWeight(160)).toBeLessThan(highlightWeight(200));
    expect(highlightWeight(200)).toBeLessThan(highlightWeight(250));
  });
});

describe("isToneIdentity", () => {
  it("is true for DEFAULT_ADJUSTMENTS", () => {
    expect(isToneIdentity(DEFAULT_ADJUSTMENTS)).toBe(true);
  });

  it("is true when only saturation/detail differ (not tone-affecting)", () => {
    const params: AdjustmentParams = { ...DEFAULT_ADJUSTMENTS, saturation: 40, detail: 60 };
    expect(isToneIdentity(params)).toBe(true);
  });

  it("is false when any of brightness/contrast/highlights/shadows is non-zero", () => {
    for (const key of ["brightness", "contrast", "highlights", "shadows"] as const) {
      const params = { ...DEFAULT_ADJUSTMENTS, [key]: 10 };
      expect(isToneIdentity(params), `key=${key}`).toBe(false);
    }
  });
});

describe("fillToneLut", () => {
  it("is the exact identity mapping (lut[x] === x) at default adjustments", () => {
    const lut = new Uint8ClampedArray(256);
    fillToneLut(lut, DEFAULT_ADJUSTMENTS);
    for (let x = 0; x < 256; x++) {
      expect(lut[x], `x=${x}`).toBe(x);
    }
  });

  it("is monotonic non-decreasing across the full input range, for extreme params", () => {
    const extreme: AdjustmentParams = {
      brightness: -100,
      contrast: 100,
      highlights: -100,
      shadows: 100,
      saturation: 0,
      detail: 0,
    };
    const lut = new Uint8ClampedArray(256);
    fillToneLut(lut, extreme);
    for (let x = 1; x < 256; x++) {
      expect(lut[x], `x=${x}`).toBeGreaterThanOrEqual(lut[x - 1]);
    }
  });

  it("max brightness shifts every input up by the full MAX_BRIGHTNESS ceiling (60), clamped at 255", () => {
    const params: AdjustmentParams = { ...DEFAULT_ADJUSTMENTS, brightness: 100 };
    const lut = new Uint8ClampedArray(256);
    fillToneLut(lut, params);
    expect(lut[0]).toBe(60);
    expect(lut[195]).toBe(255); // 195 + 60 = 255 exactly
    expect(lut[255]).toBe(255); // clamped, can't exceed 255
  });

  it("min brightness shifts every input down by the full ceiling, clamped at 0", () => {
    const params: AdjustmentParams = { ...DEFAULT_ADJUSTMENTS, brightness: -100 };
    const lut = new Uint8ClampedArray(256);
    fillToneLut(lut, params);
    expect(lut[255]).toBe(195); // 255 - 60
    expect(lut[0]).toBe(0); // clamped, can't go below 0
  });

  it("positive contrast pushes values away from the 128 midpoint", () => {
    const flat = new Uint8ClampedArray(256);
    fillToneLut(flat, DEFAULT_ADJUSTMENTS);
    const boosted = new Uint8ClampedArray(256);
    fillToneLut(boosted, { ...DEFAULT_ADJUSTMENTS, contrast: 100 });
    expect(boosted[200]).toBeGreaterThan(flat[200]);
    expect(boosted[50]).toBeLessThan(flat[50]);
    expect(boosted[128]).toBe(128); // midpoint is a fixed point of any contrast gain
  });
});
