import type { JobStatusResponse } from "../../lib/api";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { IconImage } from "../ui/Icon";
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

export function JobProgressPanel({ status }: { status: JobStatusResponse }) {
  const percent = Math.round(status.progress * 100);

  return (
    <Card padding="lg" className="space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-canvas text-muted border border-line">
            <IconImage className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-ink">Enhancing your photos</h3>
            <p className="text-[12.5px] text-muted">
              {status.completed_count} of {status.total_count} file
              {status.total_count === 1 ? "" : "s"} processed
            </p>
          </div>
        </div>
        <Badge tone={STATUS_TONE[status.status]}>{STATUS_LABEL[status.status]}</Badge>
      </div>

      <div className="space-y-1.5">
        <div className="h-2 w-full overflow-hidden rounded-full bg-canvas">
          <div
            className="h-full rounded-full bg-brand transition-[width] duration-300 ease-out"
            style={{ width: `${percent}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-[12px] text-muted">
          <span className="truncate">
            {status.current_file ? `Working on ${status.current_file}` : "Waiting to start…"}
          </span>
          <span className="shrink-0 tabular-nums">{percent}%</span>
        </div>
      </div>

      <FileErrorList
        errors={status.errors}
        heading={`${status.errors.length} file${status.errors.length === 1 ? "" : "s"} failed to enhance`}
      />
    </Card>
  );
}
