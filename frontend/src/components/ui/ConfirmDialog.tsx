import { useEffect } from "react";
import { cn } from "../../lib/cn";
import { Button } from "./Button";

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Remove",
  cancelLabel = "Cancel",
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/35 p-4"
      onMouseDown={onCancel}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div
        className="w-full max-w-sm rounded-lg bg-panel border border-line shadow-pop"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="px-5 pt-5 pb-4">
          <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
          <p className="mt-1.5 text-[13px] leading-relaxed text-muted">{message}</p>
        </div>
        <div className={cn("flex justify-end gap-2 border-t border-line px-5 py-3.5")}>
          <Button variant="secondary" size="sm" onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button variant="danger" size="sm" onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}