import { useEffect, useRef, useState } from "react";
import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { SplashScreen } from "./components/splash/SplashScreen";
import { JobFailedPanel } from "./components/upload/JobFailedPanel";
import { JobProgressPanel } from "./components/upload/JobProgressPanel";
import { JobResultPanel } from "./components/upload/JobResultPanel";
import { UploadPanel } from "./components/upload/UploadPanel";
import { RecentPage } from "./components/recent/RecentPage";
import { IconAlertTriangle } from "./components/ui/Icon";
import {
  ApiError,
  createJob,
  downloadResult,
  getJobStatus,
  type JobStatusResponse,
} from "./lib/api";
import { isDesktopShell, recordSavedResult, saveResultNative } from "./lib/desktop";
import { makeThumbnailDataUrl } from "./lib/thumbnail";

const POLL_INTERVAL_MS = 1000;

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-1.5 rounded-md border border-brand/20 bg-brand-soft px-3 py-2.5 text-[12.5px] text-brand-dark">
      <IconAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

function messageFor(err: unknown, fallback: string): string {
  return err instanceof ApiError ? err.message : fallback;
}

function HomePage() {
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // The real uploaded source (single-file jobs only) and the real generated
  // result, kept as object URLs for the before/after comparison slider and
  // reused as the download payload — never a demo/placeholder image.
  const [comparison, setComparison] = useState<{ beforeUrl: string; afterUrl: string } | null>(
    null,
  );
  const resultBlobRef = useRef<{ blob: Blob; filename: string } | null>(null);
  const beforeUrlRef = useRef<string | null>(null);
  const afterUrlRef = useRef<string | null>(null);

  const clearComparison = () => {
    if (beforeUrlRef.current) URL.revokeObjectURL(beforeUrlRef.current);
    if (afterUrlRef.current) URL.revokeObjectURL(afterUrlRef.current);
    beforeUrlRef.current = null;
    afterUrlRef.current = null;
    resultBlobRef.current = null;
    setComparison(null);
  };

  useEffect(() => clearComparison, []);

  // Poll the job status endpoint while a job is queued/processing; stop
  // once it reaches a terminal state (completed/failed).
  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") return;
    let cancelled = false;

    const poll = async () => {
      try {
        const next = await getJobStatus(job.job_id);
        if (!cancelled) setJob(next);
      } catch {
        // Transient network hiccup mid-poll — try again next tick instead
        // of tearing down the progress view over one failed request.
      }
    };

    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [job?.job_id, job?.status]);

  const handleStart = async (files: File[]) => {
    setCreating(true);
    setCreateError(null);
    clearComparison();
    if (files.length === 1) {
      beforeUrlRef.current = URL.createObjectURL(files[0]);
    }
    try {
      const created = await createJob(files);
      setJob({
        ok: true,
        job_id: created.job_id,
        status: created.status,
        progress: 0,
        current_file: null,
        current_stage: null,
        current_stage_label: null,
        completed_count: 0,
        total_count: created.total_count,
        errors: [],
      });
    } catch (err) {
      setCreateError(messageFor(err, "Could not start the job."));
    } finally {
      setCreating(false);
    }
  };

  // Once a single-file job completes, fetch the actual generated result
  // once so it can back both the comparison slider and the download/save
  // action — no separate fake preview image, no redundant re-fetch.
  useEffect(() => {
    if (!job || job.status !== "completed" || job.total_count !== 1) return;
    if (resultBlobRef.current || !beforeUrlRef.current) return;
    let cancelled = false;

    (async () => {
      try {
        const result = await downloadResult(job.job_id);
        if (cancelled) return;
        resultBlobRef.current = { blob: result.blob, filename: result.filename };
        afterUrlRef.current = URL.createObjectURL(result.blob);
        setComparison({ beforeUrl: beforeUrlRef.current!, afterUrl: afterUrlRef.current });
      } catch {
        // The explicit Save/Download button still works and surfaces its
        // own error — this background preview fetch fails silently.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [job?.job_id, job?.status, job?.total_count]);

  const triggerBrowserDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const handleDownload = async () => {
    if (!job) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      if (isDesktopShell()) {
        const outcome = await saveResultNative(job.job_id);
        if (!outcome.ok && "error" in outcome) {
          setDownloadError(outcome.error);
          return;
        }
        if (outcome.ok) {
          const isBatch = job.total_count > 1;
          const thumbnailDataUrl = !isBatch && resultBlobRef.current
            ? await makeThumbnailDataUrl(resultBlobRef.current.blob)
            : null;
          void recordSavedResult({
            path: outcome.path,
            kind: isBatch ? "zip" : "image",
            file_count: isBatch ? job.total_count - job.errors.length : undefined,
            thumbnail_data_url: thumbnailDataUrl ?? undefined,
          });
        }
        return;
      }
      if (resultBlobRef.current) {
        triggerBrowserDownload(resultBlobRef.current.blob, resultBlobRef.current.filename);
        return;
      }
      const result = await downloadResult(job.job_id);
      triggerBrowserDownload(result.blob, result.filename);
    } catch (err) {
      setDownloadError(messageFor(err, "Could not save the result."));
    } finally {
      setDownloading(false);
    }
  };

  const handleReset = () => {
    setJob(null);
    setCreateError(null);
    setDownloadError(null);
    clearComparison();
  };

  if (!job) {
    return (
      <div className="mx-auto flex max-w-2xl flex-col items-center gap-6 pt-8 text-center">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-ink">
            Enhance Your Images to 4K
          </h1>
          <p className="text-[14px] text-muted">
            Upload your images to remove blur, reduce noise and upscale to 4K with AI.
          </p>
        </div>
        {createError ? (
          <div className="w-full">
            <ErrorBanner message={createError} />
          </div>
        ) : null}
        <UploadPanel onStart={handleStart} submitting={creating} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      {createError ? <ErrorBanner message={createError} /> : null}
      {job.status === "completed" ? (
        <JobResultPanel
          status={job}
          comparison={comparison}
          onDownload={handleDownload}
          downloading={downloading}
          downloadError={downloadError}
          onReset={handleReset}
        />
      ) : job.status === "failed" ? (
        <JobFailedPanel status={job} onReset={handleReset} />
      ) : (
        <JobProgressPanel status={job} />
      )}
    </div>
  );
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true);

  return (
    <>
      {showSplash ? <SplashScreen onDone={() => setShowSplash(false)} /> : null}
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/recent" element={<RecentPage />} />
        </Route>
      </Routes>
    </>
  );
}
