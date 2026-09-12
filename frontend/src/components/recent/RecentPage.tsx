import { useEffect, useState } from "react";
import { Card } from "../ui/Card";
import { EmptyState } from "../ui/EmptyState";
import { IconAlertTriangle, IconClock, IconFolder, IconImage } from "../ui/Icon";
import {
  getRecentHistory,
  isDesktopShell,
  openInExplorer,
  openSavedFile,
  type RecentEntry,
} from "../../lib/desktop";

function formatSavedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function Thumbnail({ entry }: { entry: RecentEntry }) {
  if (entry.kind === "image" && entry.thumbnail_data_url) {
    return (
      <img
        src={entry.thumbnail_data_url}
        alt=""
        className="h-16 w-16 shrink-0 rounded-lg border border-line object-cover"
        draggable={false}
      />
    );
  }
  return (
    <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-lg border border-line bg-canvas text-muted">
      <IconImage className="h-6 w-6" />
    </div>
  );
}

function RecentItem({ entry }: { entry: RecentEntry }) {
  const [actionError, setActionError] = useState<string | null>(null);

  const handleOpenDirectory = async () => {
    setActionError(null);
    const result = await openInExplorer(entry.directory);
    if (!result.ok) setActionError(result.error ?? "Could not open that folder.");
  };

  const handleOpenFile = async () => {
    setActionError(null);
    const result = await openSavedFile(entry.path);
    if (!result.ok) setActionError(result.error ?? "Could not open that file.");
  };

  return (
    <Card variant="glass" padding="md" className="flex items-start gap-4">
      <button
        type="button"
        onClick={handleOpenFile}
        aria-label={`Open ${entry.filename}`}
        className="shrink-0 rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
      >
        <Thumbnail entry={entry} />
      </button>

      <div className="min-w-0 flex-1 space-y-1">
        <button
          type="button"
          onClick={handleOpenFile}
          className="max-w-full truncate text-left text-[14px] font-semibold text-ink hover:text-brand-dark focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
          title={entry.filename}
        >
          {entry.filename}
        </button>

        <button
          type="button"
          onClick={handleOpenDirectory}
          className="flex max-w-full items-center gap-1.5 truncate text-[12.5px] text-brand hover:text-brand-dark hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
          title={entry.directory}
        >
          <IconFolder className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{entry.directory}</span>
        </button>

        <p className="flex items-center gap-1.5 text-[11.5px] text-faint">
          <IconClock className="h-3 w-3 shrink-0" />
          {formatSavedAt(entry.saved_at)}
          {entry.kind === "zip" && entry.file_count ? (
            <span> · {entry.file_count} image{entry.file_count === 1 ? "" : "s"} (ZIP)</span>
          ) : null}
        </p>

        {actionError ? (
          <p className="flex items-center gap-1.5 text-[11.5px] text-brand-dark">
            <IconAlertTriangle className="h-3 w-3 shrink-0" />
            {actionError}
          </p>
        ) : null}
      </div>
    </Card>
  );
}

export function RecentPage() {
  const [entries, setEntries] = useState<RecentEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const desktop = isDesktopShell();

  useEffect(() => {
    if (!desktop) {
      setEntries([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const result = await getRecentHistory();
        if (cancelled) return;
        if (result.ok) {
          setEntries(result.entries);
        } else {
          setEntries([]);
          setLoadError(result.error ?? "Could not load recent history.");
        }
      } catch {
        if (!cancelled) {
          setEntries([]);
          setLoadError("Could not load recent history.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [desktop]);

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <div>
        <h1 className="text-xl font-bold tracking-tight text-ink">Recent</h1>
        <p className="text-[13px] text-muted">Images you've enhanced and saved.</p>
      </div>

      {!desktop ? (
        <EmptyState
          icon={<IconClock className="h-5 w-5" />}
          title="Recent history is available in the desktop app"
          description="Saved images are tracked using the native Save As destination, which only the desktop app can see."
        />
      ) : loadError ? (
        <EmptyState
          icon={<IconAlertTriangle className="h-5 w-5" />}
          title="Could not load recent history"
          description={loadError}
        />
      ) : entries === null ? null : entries.length === 0 ? (
        <EmptyState
          icon={<IconClock className="h-5 w-5" />}
          title="Nothing saved yet"
          description="Enhanced images you save with Save As will show up here."
        />
      ) : (
        <div className="space-y-3">
          {entries.map((entry) => (
            <RecentItem key={entry.id} entry={entry} />
          ))}
        </div>
      )}
    </div>
  );
}
