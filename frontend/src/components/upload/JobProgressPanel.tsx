import type { JobStatusResponse } from "../../lib/api";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
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

export function JobProgressPanel({ status }: { status: JobStatusResponse }) {
  const percent = Math.min(100, Math.max(0, Math.round(status.progress * 100)));
  const offset = CIRCUMFERENCE - (percent / 100) * CIRCUMFERENCE;

  return (
    <Card variant="glass" padding="lg" className="animate-rise-in space-y-6">
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

      <div className="flex flex-col items-center gap-4 py-2">
        <div className="relative flex h-40 w-40 items-center justify-center" role="img" aria-label={`${percent}% complete`}>
          {/* Ambient orbital ring — continuous, ornamental, independent of progress. */}
          <span
            aria-hidden="true"
            className="animate-spin-slow absolute inset-0 rounded-full border-[3px] border-dashed border-brand/20"
          />
          <svg viewBox="0 0 96 96" className="absolute inset-2 -rotate-90">
            <circle cx="48" cy="48" r={RADIUS} fill="none" stroke="var(--color-line)" strokeWidth="6" />
            <circle
              cx="48"
              cy="48"
              r={RADIUS}
              fill="none"
              stroke="var(--color-brand)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={offset}
              className="transition-[stroke-dashoffset] duration-500 ease-out"
            />
          </svg>
          <div className="flex flex-col items-center">
            <span className="text-2xl font-semibold tabular-nums tracking-tight text-ink">
              {percent}%
            </span>
            <span className="text-[10.5px] font-medium uppercase tracking-[0.14em] text-faint">
              {status.status === "queued" ? "Waiting" : "Enhancing"}
            </span>
          </div>
        </div>

        <p className="max-w-xs truncate text-center text-[12.5px] text-muted">
          {status.current_file ? (
            <>
              Working on <span className="font-medium text-ink-2">{status.current_file}</span>
            </>
          ) : (
            "Waiting to start…"
          )}
        </p>
      </div>

      <FileErrorList
        errors={status.errors}
        heading={`${status.errors.length} file${status.errors.length === 1 ? "" : "s"} failed to enhance`}
      />
    </Card>
  );
}
