import type { JobStatusResponse } from "../../lib/api";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { IconAlertTriangle, IconRefresh } from "../ui/Icon";
import { FileErrorList } from "./FileErrorList";

/**
 * The job-status endpoint doesn't expose the worker's internal
 * `fatal_error` string (see backend/jobs.py Job.to_status_dict) — only the
 * per-file `errors` array. When every file fails that array is the only
 * detail available, so that's what this shows.
 */
export function JobFailedPanel({
  status,
  onReset,
}: {
  status: JobStatusResponse;
  onReset: () => void;
}) {
  return (
    <Card variant="glass" padding="lg" className="animate-rise-in space-y-6">
      <div className="flex flex-col items-center gap-3 py-2 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-brand-soft text-brand border border-brand/15">
          <IconAlertTriangle className="h-6 w-6" />
        </div>
        <div className="space-y-1">
          <h3 className="text-[15px] font-semibold tracking-tight text-ink">
            Enhancement failed
          </h3>
          <p className="text-[12.5px] text-muted">
            None of the {status.total_count} file{status.total_count === 1 ? "" : "s"} could be
            enhanced.
          </p>
        </div>
        <Badge tone="warn">Failed</Badge>
      </div>

      <FileErrorList errors={status.errors} heading="Details" />

      <div className="flex items-center justify-end border-t border-line/70 pt-5">
        <Button variant="secondary" onClick={onReset} icon={<IconRefresh className="h-4 w-4" />}>
          Try again
        </Button>
      </div>
    </Card>
  );
}
