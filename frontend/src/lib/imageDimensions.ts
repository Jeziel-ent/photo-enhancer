/**
 * MVP 2: the enhanced result's real pixel dimensions are no longer always
 * 3840x2160 (aspect ratio is now preserved instead of stretched to 16:9 --
 * see backend/engine_adapter.py's aspect_preserving_target wiring). The
 * board editor needs each image's ACTUAL dimensions to convert between
 * on-screen percentages and real image-space pixel coordinates correctly.
 */
export interface ImageDimensions {
  width: number;
  height: number;
}

/** Loads `src` off-DOM just to read its natural pixel size. Rejects on any
 * load error so callers can fall back to a safe default rather than hang. */
export function loadImageDimensions(src: string): Promise<ImageDimensions> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error(`could not load image dimensions for ${src}`));
    img.src = src;
  });
}
