/**
 * Turns a real enhanced-image blob into a small JPEG data URL for the
 * Recent tab — a thumbnail reference, not a duplicate of the full 4K file.
 * Returns null on any failure (decoding, canvas) so callers can just omit
 * the thumbnail rather than fail the save.
 */
export async function makeThumbnailDataUrl(blob: Blob, maxSize = 160): Promise<string | null> {
  try {
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    return canvas.toDataURL("image/jpeg", 0.7);
  } catch {
    return null;
  }
}
