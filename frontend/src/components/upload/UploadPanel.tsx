import { useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { cn } from "../../lib/cn";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { IconAlertTriangle, IconImage, IconUpload, IconX } from "../ui/Icon";
import {
  MAX_FILES_PER_JOB,
  formatBytes,
  validateBatch,
  validateFile,
} from "../../lib/uploadValidation";

interface SelectedFile {
  id: string;
  file: File;
  error: string | null;
}

function makeId(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}:${Math.random().toString(36).slice(2)}`;
}

export function UploadPanel({
  onStart,
  submitting,
}: {
  onStart: (files: File[]) => void;
  submitting: boolean;
}) {
  const [selected, setSelected] = useState<SelectedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (incoming: FileList | File[]) => {
    const existingKeys = new Set(
      selected.map((s) => `${s.file.name}:${s.file.size}:${s.file.lastModified}`),
    );
    const next: SelectedFile[] = [];
    for (const file of Array.from(incoming)) {
      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (existingKeys.has(key)) continue; // skip exact duplicates (same file picked twice)
      existingKeys.add(key);
      next.push({ id: makeId(file), file, error: validateFile(file) });
    }
    if (next.length > 0) {
      setSelected((prev) => [...prev, ...next]);
    }
  };

  const removeFile = (id: string) => {
    setSelected((prev) => prev.filter((s) => s.id !== id));
  };

  const clearInvalid = () => {
    setSelected((prev) => prev.filter((s) => s.error === null));
  };

  const clearAll = () => setSelected([]);

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      addFiles(event.target.files);
    }
    event.target.value = ""; // allow re-selecting the same file(s) later
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      addFiles(event.dataTransfer.files);
    }
  };

  const invalidCount = selected.filter((s) => s.error !== null).length;
  const validFiles = useMemo(() => selected.filter((s) => s.error === null), [selected]);
  const batchErrors = useMemo(
    () => validateBatch(validFiles.map((s) => s.file)),
    [validFiles],
  );
  const canStart = !submitting && validFiles.length > 0 && batchErrors.length === 0;

  return (
    <Card padding="lg" className="space-y-5">
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-12 text-center transition-colors",
          dragActive
            ? "border-brand bg-brand-soft/60"
            : "border-line-strong bg-canvas/60 hover:border-brand/45 hover:bg-canvas",
        )}
      >
        <div className="mb-1 flex h-11 w-11 items-center justify-center rounded-full bg-panel text-muted border border-line shadow-card">
          <IconUpload className="h-5 w-5" />
        </div>
        <h3 className="text-sm font-semibold text-ink">
          Drag and drop photos here, or click to browse
        </h3>
        <p className="max-w-sm text-[13px] leading-relaxed text-muted">
          JPG or PNG, up to {formatBytes(40 * 1024 * 1024)} per file, {MAX_FILES_PER_JOB} files max.
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>

      {selected.length > 0 ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-[13px] font-medium text-ink-2">
              {selected.length} file{selected.length === 1 ? "" : "s"} selected
              {invalidCount > 0 ? (
                <span className="text-brand"> · {invalidCount} invalid</span>
              ) : null}
            </p>
            <div className="flex items-center gap-2">
              {invalidCount > 0 ? (
                <Button variant="ghost" size="sm" onClick={clearInvalid}>
                  Clear invalid
                </Button>
              ) : null}
              <Button variant="ghost" size="sm" onClick={clearAll}>
                Clear all
              </Button>
            </div>
          </div>

          <ul className="max-h-64 space-y-1.5 overflow-y-auto pr-1">
            {selected.map((item) => (
              <li
                key={item.id}
                className={cn(
                  "flex items-center gap-3 rounded-md border px-3 py-2",
                  item.error ? "border-brand/25 bg-brand-soft/40" : "border-line bg-white",
                )}
              >
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded bg-canvas text-muted">
                  <IconImage className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] font-medium text-ink">{item.file.name}</p>
                  {item.error ? (
                    <p className="flex items-center gap-1 text-[11.5px] text-brand">
                      <IconAlertTriangle className="h-3 w-3 shrink-0" />
                      {item.error}
                    </p>
                  ) : (
                    <p className="text-[11.5px] text-faint">{formatBytes(item.file.size)}</p>
                  )}
                </div>
                <button
                  type="button"
                  aria-label={`Remove ${item.file.name}`}
                  onClick={() => removeFile(item.id)}
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted transition-colors hover:bg-ink/[0.06] hover:text-ink"
                >
                  <IconX className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>

          {batchErrors.length > 0 ? (
            <div className="space-y-1 rounded-md border border-brand/25 bg-brand-soft/40 px-3 py-2">
              {batchErrors.map((message) => (
                <p key={message} className="flex items-center gap-1.5 text-[12.5px] text-brand-dark">
                  <IconAlertTriangle className="h-3.5 w-3.5 shrink-0" />
                  {message}
                </p>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="flex items-center justify-between border-t border-line pt-4">
        <p className="text-[12.5px] text-muted">
          {validFiles.length > 0
            ? `${validFiles.length} file${validFiles.length === 1 ? "" : "s"} ready to enhance`
            : "Select at least one photo to continue"}
        </p>
        <Button
          variant="primary"
          disabled={!canStart}
          onClick={() => onStart(validFiles.map((s) => s.file))}
        >
          {submitting ? "Starting…" : "Enhance photos"}
        </Button>
      </div>
    </Card>
  );
}
