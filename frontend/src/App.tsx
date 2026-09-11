import { useEffect, useState } from "react";
import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { SplashScreen } from "./components/splash/SplashScreen";
import { BeforeAfterPreview } from "./components/home/BeforeAfterPreview";
import { JobFailedPanel } from "./components/upload/JobFailedPanel";
import { JobProgressPanel } from "./components/upload/JobProgressPanel";
import { JobResultPanel } from "./components/upload/JobResultPanel";
import { UploadPanel } from "./components/upload/UploadPanel";
import { IconAlertTriangle } from "./components/ui/Icon";
import {
  ApiError,
  createJob,
  downloadResult,
  getJobStatus,
  type JobStatusResponse,
} from "./lib/api";

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
    try {
      const created = await createJob(files);
      setJob({
        ok: true,
        job_id: created.job_id,
        status: created.status,
        progress: 0,
        current_file: null,
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

  const handleDownload = async () => {
    if (!job) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      const result = await downloadResult(job.job_id);
      const url = URL.createObjectURL(result.blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(messageFor(err, "Could not download the result."));
    } finally {
      setDownloading(false);
    }
  };

  const handleReset = () => {
    setJob(null);
    setCreateError(null);
    setDownloadError(null);
  };

  if (!job) {
    return (
      <div className="flex flex-col gap-6 xl:flex-row xl:items-start">
        <div className="min-w-0 flex-1 space-y-4">
          {createError ? <ErrorBanner message={createError} /> : null}
          <UploadPanel onStart={handleStart} submitting={creating} />
        </div>
        <div className="w-full shrink-0 xl:w-[300px]">
          <BeforeAfterPreview />
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      {createError ? <ErrorBanner message={createError} /> : null}
      {job.status === "completed" ? (
        <JobResultPanel
          status={job}
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
        </Route>
      </Routes>
    </>
  );
}
