import { describe, expect, it } from "vitest";
import { coverSourceRect, previewSizeForLongestSide } from "./AdjustedPreviewCanvas";

describe("previewSizeForLongestSide", () => {
  it("yields exactly the historical 480x270 / 1440x810 buffers for 16:9", () => {
    expect(previewSizeForLongestSide(480, { width: 16, height: 9 })).toEqual({
      width: 480,
      height: 270,
    });
    expect(previewSizeForLongestSide(1440, { width: 16, height: 9 })).toEqual({
      width: 1440,
      height: 810,
    });
  });

  it("keeps the longest side fixed and scales the other edge for a wide aspect", () => {
    const size = previewSizeForLongestSide(1000, { width: 2, height: 1 });
    expect(size).toEqual({ width: 1000, height: 500 });
  });

  it("keeps the longest side fixed and scales the other edge for a tall aspect", () => {
    const size = previewSizeForLongestSide(1000, { width: 1, height: 2 });
    expect(size).toEqual({ width: 500, height: 1000 });
  });

  it("produces a square buffer for a 1:1 aspect", () => {
    expect(previewSizeForLongestSide(300, { width: 1, height: 1 })).toEqual({
      width: 300,
      height: 300,
    });
  });
});

describe("coverSourceRect", () => {
  it("is a full, uncropped rect when source and destination share an aspect ratio", () => {
    const rect = coverSourceRect(1920, 1080, 960, 540);
    expect(rect).toEqual({ sx: 0, sy: 0, sw: 1920, sh: 1080 });
  });

  it("crops the source's width (centered) when the source is wider than the destination", () => {
    // 2:1 source into a 1:1 destination box -> crop left/right, keep full height
    const rect = coverSourceRect(2000, 1000, 500, 500);
    expect(rect.sh).toBe(1000);
    expect(rect.sw).toBe(1000); // srcH * dstRatio = 1000 * 1
    expect(rect.sx).toBe(500); // (2000 - 1000) / 2, centered
    expect(rect.sy).toBe(0);
  });

  it("crops the source's height (centered) when the source is taller than the destination", () => {
    // 1:2 source into a 1:1 destination box -> crop top/bottom, keep full width
    const rect = coverSourceRect(1000, 2000, 500, 500);
    expect(rect.sw).toBe(1000);
    expect(rect.sh).toBe(1000); // srcW / dstRatio = 1000 / 1
    expect(rect.sy).toBe(500); // (2000 - 1000) / 2, centered
    expect(rect.sx).toBe(0);
  });
});
