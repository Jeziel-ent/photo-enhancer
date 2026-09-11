import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { cn } from "../../lib/cn";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import {
  IconAlertTriangle,
  IconArrowRight,
  IconFolder,
  IconPlus,
  IconUpload,
  IconX,
} from "../ui/Icon";
import { validateBatch, validateFile } from "../../lib/uploadValidation";

interface SelectedFile {
  id: string;
  file: File;
  error: string | null;
  previewUrl: string;
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

  // Keep a live ref of the current selection so the unmount cleanup below
  // can revoke every outstanding object URL, not just the ones from the
  // render that installed the effect.
  const selectedRef = useRef<SelectedFile[]>([]);
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);
  useEffect(() => {
    return () => {
      selectedRef.current.forEach((s) => URL.revokeObjectURL(s.previewUrl));
    };
  }, []);

  const addFiles = (incoming: FileList | File[]) => {
    const existingKeys = new Set(
      selected.map((s) => `${s.file.name}:${s.file.size}:${s.file.lastModified}`),
    );
    const next: SelectedFile[] = [];
    for (const file of Array.from(incoming)) {
      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (existingKeys.has(key)) continue; // skip exact duplicates (same file picked twice)
      existingKeys.add(key);
      next.push({
        id: makeId(file),
        file,
        error: validateFile(file),
        previewUrl: URL.createObjectURL(file),
      });
    }
    if (next.length > 0) {
      setSelected((prev) => [...prev, ...next]);
    }
  };

  const removeFile = (id: string) => {
    setSelected((prev) => {
      const target = prev.find((s) => s.id === id);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((s) => s.id !== id);
    });
  };

  const clearInvalid = () => {
    setSelected((prev) => {
      prev.filter((s) => s.error !== null).forEach((s) => URL.revokeObjectURL(s.previewUrl));
      return prev.filter((s) => s.error === null);
    });
  };

  const clearAll = () => {
    selected.forEach((s) => URL.revokeObjectURL(s.previewUrl));
    setSelected([]);
  };

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

  const openPicker = () => inputRef.current?.click();

  const invalidCount = selected.filter((s) => s.error !== null).length;
  const validFiles = useMemo(() => selected.filter((s) => s.error === null), [selected]);
  const batchErrors = useMemo(
    () => validateBatch(validFiles.map((s) => s.file)),
    [validFiles],
  );
  const canStart = !submitting && validFiles.length > 0 && batchErrors.length === 0;

  return (
    <Card variant="glass" padding="lg" className="animate-rise-in space-y-6">
      <div
        role="button"
        tabIndex={0}
        aria-label="Choose photos to enhance, or drag and drop them here"
        onClick={openPicker}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openPicker();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-16 text-center transition-all duration-200",
          dragActive
            ? "border-brand bg-brand-soft/50 shadow-[0_0_0_4px_rgba(209,33,38,0.08)]"
            : "border-brand/35 bg-white/50 hover:border-brand/55 hover:bg-white/70",
        )}
      >
        <div className="mb-1 flex h-20 w-20 items-center justify-center rounded-full bg-brand-soft">
          <IconUpload className="h-8 w-8 text-brand" />
        </div>
        <h3 className="text-xl font-bold tracking-tight text-ink">Drop images here</h3>
        <p className="text-[14px] text-muted">or browse</p>
        <p className="text-[12px] text-faint">JPG · PNG</p>
        <Button
          variant="primary"
          size="lg"
          rounded="full"
          className="mt-2"
          icon={<IconFolder className="h-4 w-4" />}
          onClick={(event) => {
            event.stopPropagation();
            openPicker();
          }}
        >
          Browse Images
        </Button>
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
          {invalidCount > 0 ? (
            <div className="flex items-center justify-between">
              <p className="text-[12.5px] text-brand">
                {invalidCount} file{invalidCount === 1 ? "" : "s"} can't be used
              </p>
              <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" onClick={clearInvalid}>
                  Clear invalid
                </Button>
                <Button variant="ghost" size="sm" onClick={clearAll}>
                  Clear all
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex justify-end">
              <Button variant="ghost" size="sm" onClick={clearAll}>
                Clear all
              </Button>
            </div>
          )}

          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5">
            {selected.map((item) => (
              <div key={item.id} className="relative">
                <div
                  className={cn(
                    "aspect-square w-full overflow-hidden rounded-xl border bg-canvas",
                    item.error ? "border-brand ring-2 ring-brand/25" : "border-line",
                  )}
                >
                  <img
                    src={item.previewUrl}
                    alt=""
                    className="h-full w-full object-cover"
                    draggable={false}
                  />
                </div>
                {item.error ? (
                  <>
                    <span
                      className="absolute bottom-1.5 left-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-brand text-white"
                      title={item.error}
                      aria-hidden="true"
                    >
                      <IconAlertTriangle className="h-3 w-3" />
                    </span>
                    <p className="sr-only">
                      {item.file.name}: {item.error}
                    </p>
                  </>
                ) : null}
                <button
                  type="button"
                  aria-label={`Remove ${item.file.name}`}
                  onClick={() => removeFile(item.id)}
                  className="absolute -top-2 -right-2 flex h-6 w-6 items-center justify-center rounded-full border border-line bg-white text-ink-2 shadow-card transition-colors hover:bg-brand hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
                >
                  <IconX className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}

            <button
              type="button"
              onClick={openPicker}
              aria-label="Add more photos"
              className="flex aspect-square w-full flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed border-line-strong text-muted transition-colors hover:border-brand/40 hover:text-brand"
            >
              <IconPlus className="h-5 w-5" />
              <span className="text-[11px] font-medium">Add More</span>
            </button>
          </div>

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

      <div className="flex items-center justify-between border-t border-line/70 pt-5">
        <p className="text-[13px] text-muted">
          {validFiles.length > 0
            ? `${validFiles.length} image${validFiles.length === 1 ? "" : "s"} selected`
            : "Select at least one photo to continue"}
        </p>
        <Button
          variant="primary"
          size="lg"
          rounded="full"
          disabled={!canStart}
          icon={<IconArrowRight className="h-4 w-4" />}
          className="flex-row-reverse"
          onClick={() => onStart(validFiles.map((s) => s.file))}
        >
          {submitting ? "Starting…" : "Enhance Images"}
        </Button>
      </div>
    </Card>
  );
}
