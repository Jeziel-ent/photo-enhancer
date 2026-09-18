import { useEffect, useRef, useState } from "react";
import { IconChevronLeft, IconChevronRight } from "../ui/Icon";

/**
 * Real before/after comparison slider for a completed job: `beforeSrc` is
 * the actual uploaded source image, `afterSrc` is the actual generated 4K
 * result — both object URLs the caller owns and revokes. No demo/decorative
 * imagery. The caller overlays the Before / After · 4K badges via its own
 * layout; this component renders only the draggable split.
 */
export function CompareSlider({
  beforeSrc,
  afterSrc,
}: {
  beforeSrc: string;
  afterSrc: string;
}) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const updateFromClientX = (clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.min(99, Math.max(1, pct)));
  };

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (draggingRef.current) updateFromClientX(event.clientX);
    };
    const onUp = () => {
      draggingRef.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative aspect-[16/9] w-full touch-none overflow-hidden rounded-xl border border-line select-none"
      onPointerDown={(event) => {
        draggingRef.current = true;
        updateFromClientX(event.clientX);
      }}
    >
      <img
        src={afterSrc}
        alt="Enhanced 4K result"
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        draggable={false}
      />
      <div
        className="pointer-events-none absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img
          src={beforeSrc}
          alt="Original uploaded photo"
          className="absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />
      </div>
      <div
        className="pointer-events-none absolute inset-y-0 w-px bg-white"
        style={{ left: `${position}%`, boxShadow: "0 0 0 1px rgba(0,0,0,0.08)" }}
      />
      <button
        type="button"
        role="slider"
        aria-label="Drag to compare before and after"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(position)}
        onPointerDown={(event) => {
          event.stopPropagation();
          draggingRef.current = true;
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setPosition((p) => Math.max(1, p - 4));
          if (event.key === "ArrowRight") setPosition((p) => Math.min(99, p + 4));
        }}
        className="absolute top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-ink shadow-pop focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
        style={{ left: `${position}%` }}
      >
        <IconChevronLeft className="h-3.5 w-3.5" />
        <IconChevronRight className="-ml-1.5 h-3.5 w-3.5" />
      </button>
    </div>
  );
}
