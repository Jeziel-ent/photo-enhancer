import type { ReactNode } from "react";
import { cn } from "../../lib/cn";

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line-strong bg-canvas/60 px-6 py-12 text-center",
        className,
      )}
    >
      {icon ? (
        <div className="mb-1 flex h-11 w-11 items-center justify-center rounded-full bg-panel text-muted border border-line shadow-card">
          {icon}
        </div>
      ) : null}
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {description ? (
        <p className="max-w-sm text-[13px] leading-relaxed text-muted">{description}</p>
      ) : null}
      {action ? <div className="mt-3">{action}</div> : null}
    </div>
  );
}