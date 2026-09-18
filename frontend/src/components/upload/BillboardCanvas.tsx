import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "../../lib/cn";

export interface BillboardRect {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** MVP 2 fallback only: used before a result's real dimensions have loaded
 * (see frontend/src/lib/imageDimensions.ts) -- matches the pre-MVP2 fixed
 * 16:9 output exactly, so a still-loading 16:9 image behaves identically to
 * before. Every real caller passes the image's actual imageWidth/imageHeight
 * once known. */
const DEFAULT_IMAGE_W = 3840;
const DEFAULT_IMAGE_H = 2160;
const MIN_SIZE_PX = 24;
const HANDLE_SIZE = 12;

interface PctRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface PctRectWithId extends PctRect {
  id: string;
}

let _nextId = 1;
export function createBillboardId(): string {
  return `bb-${_nextId++}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function pctFromPoints(ax: number, ay: number, bx: number, by: number): PctRect {
  const x = clamp(Math.min(ax, bx), 0, 100);
  const y = clamp(Math.min(ay, by), 0, 100);
  const x2 = clamp(Math.max(ax, bx), 0, 100);
  const y2 = clamp(Math.max(ay, by), 0, 100);
  return { x, y, w: x2 - x, h: y2 - y };
}

function pctToImageRect(
  pct: PctRect,
  imageW: number,
  imageH: number,
): { x: number; y: number; width: number; height: number } {
  const x = Math.round((pct.x / 100) * imageW);
  const y = Math.round((pct.y / 100) * imageH);
  const width = Math.round((pct.w / 100) * imageW);
  const height = Math.round((pct.h / 100) * imageH);
  return {
    x: clamp(x, 0, imageW - 1),
    y: clamp(y, 0, imageH - 1),
    width: clamp(width, 0, imageW - x),
    height: clamp(height, 0, imageH - y),
  };
}

function pctToBillboardRect(pct: PctRectWithId, imageW: number, imageH: number): BillboardRect {
  return { ...pctToImageRect(pct, imageW, imageH), id: pct.id };
}

function imageRectToPct(rect: BillboardRect, imageW: number, imageH: number): PctRectWithId {
  return {
    x: (rect.x / imageW) * 100,
    y: (rect.y / imageH) * 100,
    w: (rect.width / imageW) * 100,
    h: (rect.height / imageH) * 100,
    id: rect.id,
  };
}

type DragMode =
  | { kind: "draw"; newId: string }
  | { kind: "move"; id: string; startPct: PctRect; grabX: number; grabY: number }
  | {
      kind: "resize";
      id: string;
      corner: "nw" | "ne" | "sw" | "se";
      fixedX: number;
      fixedY: number;
    };

export function BillboardCanvas({
  imageSrc,
  imageWidth = DEFAULT_IMAGE_W,
  imageHeight = DEFAULT_IMAGE_H,
  rectangles,
  selectedId,
  onChange,
  onSelect,
  readOnly = false,
  addingMode = false,
  onDrawStart,
  onDrawCommit,
}: {
  imageSrc: string;
  /** The result image's REAL pixel dimensions (see
   * frontend/src/lib/imageDimensions.ts) -- rectangles are stored in this
   * same coordinate space. Defaults to the classic 3840x2160 for a caller
   * that hasn't loaded real dimensions yet. */
  imageWidth?: number;
  imageHeight?: number;
  rectangles: BillboardRect[];
  selectedId: string | null;
  onChange: (rects: BillboardRect[]) => void;
  onSelect: (id: string | null) => void;
  readOnly?: boolean;
  /** When true, shows the "draw a new board" hint even if boards exist. */
  addingMode?: boolean;
  /** Fired when the user begins drawing a new board (pointer down). */
  onDrawStart?: () => void;
  /** Fired when a board drawing gesture completes (pointer up). */
  onDrawCommit?: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [draftRects, setDraftRects] = useState<PctRectWithId[]>(() =>
    rectangles.map((r) => imageRectToPct(r, imageWidth, imageHeight)),
  );
  const dragRef = useRef<DragMode | null>(null);
  const startPointRef = useRef<{ x: number; y: number } | null>(null);
  const draftRef = useRef<PctRectWithId[]>(draftRects);
  const onDrawStartRef = useRef(onDrawStart);
  const onDrawCommitRef = useRef(onDrawCommit);

  useEffect(() => {
    onDrawStartRef.current = onDrawStart;
  }, [onDrawStart]);

  useEffect(() => {
    onDrawCommitRef.current = onDrawCommit;
  }, [onDrawCommit]);

  useEffect(() => {
    setDraftRects(rectangles.map((r) => imageRectToPct(r, imageWidth, imageHeight)));
  }, [rectangles, imageWidth, imageHeight]);

  useEffect(() => {
    draftRef.current = draftRects;
  }, [draftRects]);

  const commitToParent = useCallback(
    (next: PctRectWithId[]) => {
      onChange(next.map((r) => pctToBillboardRect(r, imageWidth, imageHeight)));
    },
    [onChange, imageWidth, imageHeight],
  );

  const pctFromClient = useCallback((clientX: number, clientY: number) => {
    const el = containerRef.current;
    if (!el) return { x: 0, y: 0 };
    const box = el.getBoundingClientRect();
    return {
      x: clamp(((clientX - box.left) / box.width) * 100, 0, 100),
      y: clamp(((clientY - box.top) / box.height) * 100, 0, 100),
    };
  }, []);

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (readOnly) return;
    const target = event.target as HTMLElement;
    (event.target as Element).setPointerCapture(event.pointerId);
    const pt = pctFromClient(event.clientX, event.clientY);

    const cornerAttr = target.getAttribute("data-corner");
    if (cornerAttr) {
      const rectId = target.closest("[data-rect-id]")?.getAttribute("data-rect-id");
      if (rectId) {
        const draft = draftRef.current.find((r) => r.id === rectId);
        if (draft) {
          const corner = cornerAttr as "nw" | "ne" | "sw" | "se";
          const fixedX =
            corner === "nw" || corner === "sw" ? draft.x + draft.w : draft.x;
          const fixedY =
            corner === "nw" || corner === "ne" ? draft.y + draft.h : draft.y;
          dragRef.current = {
            kind: "resize",
            id: rectId,
            corner,
            fixedX,
            fixedY,
          };
          startPointRef.current = pt;
          onSelect(rectId);
          return;
        }
      }
    }

    const bodyEl = target.closest("[data-body]");
    if (bodyEl) {
      const rectId = bodyEl.getAttribute("data-rect-id");
      if (rectId) {
        const draft = draftRef.current.find((r) => r.id === rectId);
        if (draft) {
          dragRef.current = {
            kind: "move",
            id: rectId,
            startPct: draft,
            grabX: pt.x,
            grabY: pt.y,
          };
          startPointRef.current = pt;
          onSelect(rectId);
          return;
        }
      }
    }

    const newId = createBillboardId();
    dragRef.current = { kind: "draw", newId };
    startPointRef.current = pt;
    onDrawStartRef.current?.();
    const newPct = { ...pctFromPoints(pt.x, pt.y, pt.x, pt.y), id: newId };
    setDraftRects((prev) => [...prev, newPct]);
    onSelect(newId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const mode = dragRef.current;
    const start = startPointRef.current;
    if (!mode || !start) return;
    const pt = pctFromClient(event.clientX, event.clientY);

    if (mode.kind === "draw") {
      setDraftRects((prev) =>
        prev.map((r) =>
          r.id === mode.newId
            ? { ...pctFromPoints(start.x, start.y, pt.x, pt.y), id: r.id }
            : r,
        ),
      );
      return;
    }
    if (mode.kind === "move") {
      const dx = pt.x - mode.grabX;
      const dy = pt.y - mode.grabY;
      const w = mode.startPct.w;
      const h = mode.startPct.h;
      const x = clamp(mode.startPct.x + dx, 0, 100 - w);
      const y = clamp(mode.startPct.y + dy, 0, 100 - h);
      setDraftRects((prev) =>
        prev.map((r) => (r.id === mode.id ? { ...r, x, y } : r)),
      );
      return;
    }
    if (mode.kind === "resize") {
      setDraftRects((prev) =>
        prev.map((r) =>
          r.id === mode.id
            ? { ...pctFromPoints(mode.fixedX, mode.fixedY, pt.x, pt.y), id: r.id }
            : r,
        ),
      );
    }
  };

  const finishDrag = () => {
    const mode = dragRef.current;
    dragRef.current = null;
    startPointRef.current = null;
    if (!mode) return;

    const prev = draftRef.current;
    const next = prev.filter((r) => {
      const wPx = (r.w / 100) * imageWidth;
      const hPx = (r.h / 100) * imageHeight;
      return wPx >= MIN_SIZE_PX && hPx >= MIN_SIZE_PX;
    });
    setDraftRects(next);
    commitToParent(next);
    if (mode.kind === "draw") {
      onDrawCommitRef.current?.();
    }

    if (
      selectedId &&
      !next.some((r) => r.id === selectedId)
    ) {
      onSelect(null);
    }
  };

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (readOnly) return;
      if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {
        e.preventDefault();
        const next = draftRef.current.filter((r) => r.id !== selectedId);
        setDraftRects(next);
        commitToParent(next);
        onSelect(null);
      }
    },
    [readOnly, selectedId, commitToParent, onSelect],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div
      ref={containerRef}
      className="relative w-full touch-none overflow-hidden rounded-xl border border-line bg-canvas select-none"
      style={{ aspectRatio: `${imageWidth} / ${imageHeight}` }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishDrag}
      onPointerCancel={finishDrag}
    >
      <img
        src={imageSrc}
        alt="Enhanced 4K result"
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        draggable={false}
      />

      {draftRects.map((r) => {
        const isSelected = r.id === selectedId;
        const labelAbove = r.y > 6;
        const labelOnLeft = r.x + r.w <= 88;
        return (
          <div
            key={r.id}
            data-body
            data-rect-id={r.id}
            className={cn(
              "absolute border-2",
              isSelected
                ? "border-brand"
                : readOnly
                  ? "border-red-500/80"
                  : "border-red-500/60",
              readOnly ? "" : "cursor-move",
              !readOnly && isSelected ? "z-20" : "z-10",
            )}
            style={{
              left: `${r.x}%`,
              top: `${r.y}%`,
              width: `${r.w}%`,
              height: `${r.h}%`,
            }}
          >
            <span
              className={cn(
                "absolute text-[10px] font-semibold tracking-wide",
                labelAbove ? "-top-[17px]" : "top-0.5",
                labelOnLeft ? "left-0" : "right-0",
              )}
              style={{
                color: "#d12126",
                textShadow:
                  "0 1px 2px rgba(255,255,255,0.95), 0 0 1px rgba(255,255,255,0.95)",
              }}
            >
              Board
            </span>

            {!readOnly &&
              isSelected &&
              (["nw", "ne", "sw", "se"] as const).map((corner) => (
                <span
                  key={corner}
                  data-corner={corner}
                  data-rect-id={r.id}
                  className="absolute rounded-full border-2 border-brand bg-white shadow-card"
                  style={{
                    width: HANDLE_SIZE,
                    height: HANDLE_SIZE,
                    zIndex: 30,
                    cursor:
                      corner === "nw" || corner === "se"
                        ? "nwse-resize"
                        : "nesw-resize",
                    top:
                      corner === "nw" || corner === "ne"
                        ? -HANDLE_SIZE / 2
                        : undefined,
                    bottom:
                      corner === "sw" || corner === "se"
                        ? -HANDLE_SIZE / 2
                        : undefined,
                    left:
                      corner === "nw" || corner === "sw"
                        ? -HANDLE_SIZE / 2
                        : undefined,
                    right:
                      corner === "ne" || corner === "se"
                        ? -HANDLE_SIZE / 2
                        : undefined,
                  }}
                />
              ))}
          </div>
        );
      })}

      {draftRects.length === 0 && !readOnly && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
          <p className="rounded-full bg-black/55 px-3.5 py-1.5 text-[12.5px] font-medium text-white">
            Click and drag to mark a board area
          </p>
        </div>
      )}

      {addingMode && !readOnly && draftRects.length > 0 && (
        <div className="pointer-events-none absolute inset-x-0 bottom-3 flex items-center justify-center">
          <p className="rounded-full bg-brand px-3 py-1 text-[11.5px] font-medium text-white shadow-pop">
            Click and drag on the photo to add a board
          </p>
        </div>
      )}
    </div>
  );
}

export function formatRectSize(rect: BillboardRect): string {
  return `${rect.width} × ${rect.height} px`;
}

export function formatRectPosition(rect: BillboardRect): string {
  return `Position ${rect.x}, ${rect.y}`;
}
