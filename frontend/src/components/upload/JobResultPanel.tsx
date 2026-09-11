import type { JobStatusResponse } from "../../lib/api";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { IconAlertTriangle, IconCheck, IconDownload, IconRefresh, IconSparkles } from "../ui/Icon";
import { FileErrorList } from "./FileErrorList";

export function JobResultPanel({
  status,
  onDownload,
  downloading,
  downloadError,
  onReset,
}: {
  status: JobStatusResponse;
  onDownload: () => void;
  downloading: boolean;
  downloadError: string | null;
  onReset: () => void;
}) {
  const isBatch = status.total_count > 1;
  const succeededCount = status.total_count - status.errors.length;

  return (
    <Card variant="glass" padding="lg" className="animate-rise-in space-y-6">
      <div className="flex flex-col items-center gap-4 py-2 text-center">
        <div className="relative flex h-16 w-16 items-center justify-center">
          <span
            aria-hidden="true"
            className="absolute inset-0 animate-soft-pulse rounded-full bg-ok/15 blur-md"
          />
          <div className="animate-pop-in relative flex h-16 w-16 items-center justify-center rounded-full bg-ok text-white shadow-card">
            <IconCheck className="h-7 w-7" />
          </div>
        </div>

        <div className="space-y-1.5">
          <p className="flex items-center justify-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.22em] text-brand">
            <IconSparkles className="h-3.5 w-3.5" aria-hidden="true" />
            4K Ready
          </p>
          <h3 className="text-[17px] font-semibold tracking-tight text-ink">
            {succeededCount} photo{succeededCount === 1 ? "" : "s"} enhanced to 4K
          </h3>
          <p className="text-[12.5px] text-muted">
            {isBatch ? "Ready to download as a ZIP archive." : "Ready to download as a PNG file."}
          </p>
        </div>

        <Badge tone="ok">Completed</Badge>
      </div>

      <FileErrorList
        errors={status.errors}
        heading={`${status.errors.length} file${status.errors.length === 1 ? "" : "s"} could not be enhanced and were left out`}
      />

      {downloadError ? (
        <div className="flex items-start gap-1.5 rounded-md border border-brand/20 bg-brand-soft px-3 py-2.5 text-[12.5px] text-brand-dark">
          <IconAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{downloadError}</span>
        </div>
      ) : null}

      <div className="flex items-center justify-between border-t border-line/70 pt-5">
        <Button variant="ghost" onClick={onReset} icon={<IconRefresh className="h-4 w-4" />}>
          Enhance more photos
        </Button>
        <Button
          variant="primary"
          size="lg"
          onClick={onDownload}
          disabled={downloading}
          icon={<IconDownload className="h-4 w-4" />}
        >
          {downloading ? "Preparing download…" : isBatch ? "Download ZIP" : "Download PNG"}
        </Button>
      </div>
    </Card>
  );
}
