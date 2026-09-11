import type { JobStatusResponse } from "../../lib/api";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { IconAlertTriangle, IconCheckCircle, IconDownload, IconRefresh } from "../ui/Icon";
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
    <Card padding="lg" className="space-y-5">
      <div className="flex items-center gap-2.5">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-ok-soft text-ok border border-ok/15">
          <IconCheckCircle className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-ink">
            {succeededCount} photo{succeededCount === 1 ? "" : "s"} enhanced to 4K
          </h3>
          <p className="text-[12.5px] text-muted">
            {isBatch ? "Ready to download as a ZIP archive." : "Ready to download as a PNG file."}
          </p>
        </div>
        <Badge tone="ok" className="ml-auto">
          Completed
        </Badge>
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

      <div className="flex items-center justify-between border-t border-line pt-4">
        <Button variant="ghost" onClick={onReset} icon={<IconRefresh className="h-4 w-4" />}>
          Enhance more photos
        </Button>
        <Button
          variant="primary"
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
