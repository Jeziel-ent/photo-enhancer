import type { ReactNode } from "react";
import type { JobStatusResponse } from "../../lib/api";
import { cn } from "../../lib/cn";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { IconEye, IconExpand, IconFocus, IconSliders } from "../ui/Icon";
import { FileErrorList } from "./FileErrorList";

const STATUS_LABEL: Record<JobStatusResponse["status"], string> = {
  queued: "Queued",
  processing: "Processing",
  completed: "Completed",
  failed: "Failed",
};

const STATUS_TONE: Record<JobStatusResponse["status"], "neutral" | "ok" | "warn" | "brand"> = {
  queued: "neutral",
  processing: "brand",
  completed: "ok",
  failed: "warn",
};

const RADIUS = 42;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

/**
 * The 4 stages shown around the ring, each mapped from the REAL backend
 * stage keys (backend/engine_adapter.py's STAGES, surfaced as
 * status.current_stage) -- never invented. "preparing" folds into
 * Analyzing and "finalizing" folds into Upscaling so the six real engine
 * stages fit the four-stage layout without fabricating anything: the
 * active/done/pending state below is always derived from the real
 * status.current_stage, and the percentage/label text always come
 * straight from status.progress / status.current_stage_label.
 */
const STAGE_GROUPS: { label: string; icon: ReactNode; match: string[] }[] = [
  { label: "Analyzing", icon: <IconEye className="h-5 w-5" />, match: ["preparing", "analyzing"] },
  { label: "Denoising", icon: <IconFocus className="h-5 w-5" />, match: ["restoring"] },
  { label: "Enhancing", icon: <IconSliders className="h-5 w-5" />, match: ["enhancing_details"] },
  { label: "Upscaling", icon: <IconExpand className="h-5 w-5" />, match: ["upscaling", "finalizing"] },
];

const DOT_COUNT = 28;

function ProgressDots({ percent }: { percent: number }) {
  const dots = Array.from({ length: DOT_COUNT }, (_, i) => i);
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0">
      {dots.map((i) => {
        const angleDeg = (i / DOT_COUNT) * 360;
        const lit = (i / DOT_COUNT) * 100 <= percent;
        const isLeading = lit && (i + 1) / DOT_COUNT * 100 > percent;
        return (
          <span
            key={i}
            className="absolute top-1/2 left-1/2"
            style={{ transform: `rotate(${angleDeg}deg) translateY(-124px)` }}
          >
            <span
              className={cn(
                "block h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full transition-colors duration-500",
                lit ? "bg-brand shadow-[0_0_6px_rgba(209,33,38,0.65)]" : "bg-ink/10",
                isLeading && "animate-soft-pulse",
              )}
            />
          </span>
        );
      })}
    </div>
  );
}

function StageStep({
  icon,
  label,
  state,
}: {
  icon: ReactNode;
  label: string;
  state: "done" | "active" | "pending";
}) {
  return (
    <div className="flex flex-1 flex-col items-center gap-2 text-center">
      <div
        className={cn(
          "flex h-12 w-12 items-center justify-center rounded-full border transition-colors duration-300",
          state === "done" && "border-transparent bg-brand text-white",
          state === "active" &&
            "animate-soft-pulse border-brand/30 bg-brand-soft text-brand shadow-[0_0_0_5px_rgba(209,33,38,0.1)]",
          state === "pending" && "border-line bg-canvas text-faint",
        )}
      >
        {icon}
      </div>
      <span
        className={cn(
          "text-[12px] tracking-wide",
          state === "active" && "font-semibold text-brand",
          state === "done" && "font-medium text-brand-dark",
          state === "pending" && "font-medium text-faint",
        )}
      >
        {label}
      </span>
    </div>
  );
}

export function JobProgressPanel({ status }: { status: JobStatusResponse }) {
  const percent = Math.min(100, Math.max(0, Math.round(status.progress * 100)));
  const offset = CIRCUMFERENCE - (percent / 100) * CIRCUMFERENCE;
  const isBatch = status.total_count > 1;
  // completed_count is 0-indexed files-done; the file in flight is the next one.
  const fileNumber = Math.min(status.total_count, status.completed_count + 1);

  const activeGroupIndex = STAGE_GROUPS.findIndex(
    (g) => status.current_stage != null && g.match.includes(status.current_stage),
  );

  return (
    <Card variant="glass" padding="lg" className="animate-rise-in space-y-7">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-[15px] font-semibold tracking-tight text-ink">
            Enhancing your photos
          </h3>
          <p className="text-[12.5px] text-muted">
            {status.completed_count} of {status.total_count} file
            {status.total_count === 1 ? "" : "s"} processed
          </p>
        </div>
        <Badge tone={STATUS_TONE[status.status]}>{STATUS_LABEL[status.status]}</Badge>
      </div>

      <div className="flex flex-col items-center gap-6 py-4">
        <div
          className="relative flex h-64 w-64 items-center justify-center"
          role="img"
          aria-label={`${percent}% complete`}
        >
          {/* Soft red/pink ambient glow behind the ring. */}
          <span
            aria-hidden="true"
            className="animate-soft-pulse absolute inset-4 rounded-full bg-brand/25 blur-2xl"
          />
          {/* Small progress/connection dots orbiting the ring, lit in step
              with the real percentage above. */}
          <ProgressDots percent={percent} />
          {/* Ambient rotating sheen — ornamental, independent of progress. */}
          <span
            aria-hidden="true"
            className="animate-spin-slow absolute inset-6 rounded-full opacity-70"
            style={{
              background:
                "conic-gradient(from 0deg, transparent 0%, rgba(209,33,38,0.18) 18%, transparent 32%)",
            }}
          />
          <svg viewBox="0 0 96 96" className="absolute inset-10 -rotate-90 drop-shadow-[0_2px_10px_rgba(209,33,38,0.35)]">
            <defs>
              <linearGradient id="ring-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ff8a3d" />
                <stop offset="100%" stopColor="var(--color-brand)" />
              </linearGradient>
            </defs>
            <circle cx="48" cy="48" r={RADIUS} fill="none" stroke="var(--color-line)" strokeWidth="7" />
            <circle
              cx="48"
              cy="48"
              r={RADIUS}
              fill="none"
              stroke="url(#ring-gradient)"
              strokeWidth="7"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={offset}
              className="transition-[stroke-dashoffset] duration-500 ease-out"
            />
          </svg>
          {/* Sits above the ring/glow/dots/sheen (all position:absolute
              siblings, which would otherwise paint on top of a plain flow
              element regardless of DOM order) so the percentage and stage
              label are never visually clipped or overlapped by the ring. */}
          <div className="relative z-10 flex max-w-[136px] flex-col items-center px-2">
            <span className="text-[40px] leading-none font-bold tabular-nums tracking-tight text-ink">
              {percent}%
            </span>
            <span className="mt-2.5 text-center text-[11px] leading-snug font-semibold uppercase tracking-[0.08em] text-brand-dark">
              {status.status === "queued" ? "Waiting" : status.current_stage_label || "Enhancing"}
            </span>
          </div>
        </div>

        <div className="flex w-full max-w-md items-start justify-between gap-3">
          {STAGE_GROUPS.map((group, i) => (
            <StageStep
              key={group.label}
              icon={group.icon}
              label={group.label}
              state={
                activeGroupIndex === -1
                  ? "pending"
                  : i < activeGroupIndex
                    ? "done"
                    : i === activeGroupIndex
                      ? "active"
                      : "pending"
              }
            />
          ))}
        </div>

        <div className="space-y-1 text-center">
          {/* status.current_file is the ORIGINAL uploaded filename (see
              backend/jobs.py: current_file = file_result.original_filename)
              -- never the enhanced/output filename. */}
          <p className="max-w-sm truncate text-[12.5px] text-muted">
            {status.current_file ? (
              <span className="font-medium text-ink-2">{status.current_file}</span>
            ) : (
              "Waiting to start…"
            )}
          </p>
          {isBatch ? (
            <p className="text-[11px] text-faint">
              File {fileNumber} of {status.total_count}
            </p>
          ) : null}
          <p className="text-[11px] text-faint">This may take a few moments…</p>
        </div>
      </div>

      <FileErrorList
        errors={status.errors}
        heading={`${status.errors.length} file${status.errors.length === 1 ? "" : "s"} failed to enhance`}
      />
    </Card>
  );
}
