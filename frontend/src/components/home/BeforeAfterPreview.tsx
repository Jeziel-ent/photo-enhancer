import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import demoPhoto from "../../assets/demo-preview.jpg";
import { Card } from "../ui/Card";
import { IconChevronLeft, IconChevronRight, IconExpand, IconFocus, IconSliders } from "../ui/Icon";

function Capability({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <div className="flex flex-col items-center gap-1.5 text-center">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-canvas text-brand">
        {icon}
      </div>
      <span className="text-[11px] font-medium text-ink-2">{label}</span>
    </div>
  );
}

/**
 * Illustrative demo only — a real Adinn OOH sample photo
 * (image_enhancer/originals/3.jpeg) with a CSS filter standing in for the
 * "after" state. Not tied to job state; purely decorative, matching the
 * reference design's marketing panel.
 */
export function BeforeAfterPreview() {
  const [position, setPosition] = useState(56);
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const updateFromClientX = (clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.min(96, Math.max(4, pct)));
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
    <Card variant="glass" padding="md" className="animate-rise-in w-full space-y-4">
      <div
        ref={containerRef}
        className="relative aspect-[4/3] w-full touch-none overflow-hidden rounded-xl border border-line select-none"
        onPointerDown={(event) => {
          draggingRef.current = true;
          updateFromClientX(event.clientX);
        }}
      >
        <img
          src={demoPhoto}
          alt="Original photo before enhancement"
          className="pointer-events-none absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />
        <div
          className="pointer-events-none absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 0 0 ${position}%)` }}
        >
          <img
            src={demoPhoto}
            alt="Same photo after 4K enhancement"
            className="absolute inset-0 h-full w-full object-cover [filter:saturate(1.25)_contrast(1.2)_brightness(1.05)]"
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
            if (event.key === "ArrowLeft") setPosition((p) => Math.max(4, p - 4));
            if (event.key === "ArrowRight") setPosition((p) => Math.min(96, p + 4));
          }}
          className="absolute top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-ink shadow-pop focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
          style={{ left: `${position}%` }}
        >
          <IconChevronLeft className="h-3.5 w-3.5" />
          <IconChevronRight className="-ml-1.5 h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex items-center justify-between">
        <span className="rounded-full bg-ink/[0.06] px-3 py-1 text-[11px] font-medium text-ink-2">
          Before
        </span>
        <span className="rounded-full bg-brand px-3 py-1 text-[11px] font-semibold text-white">
          4K After
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 border-t border-line/70 pt-4">
        <Capability icon={<IconFocus className="h-[18px] w-[18px]" />} label="Denoise" />
        <Capability icon={<IconSliders className="h-[18px] w-[18px]" />} label="Enhance" />
        <Capability icon={<IconExpand className="h-[18px] w-[18px]" />} label="4K Upscale" />
      </div>
    </Card>
  );
}
