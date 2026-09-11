import type { ReactNode } from "react";
import { cn } from "../../lib/cn";

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
  className,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-end justify-between gap-4", className)}>
      <div className="space-y-1">
        {eyebrow ? (
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-brand">
            {eyebrow}
          </p>
        ) : null}
        <h1 className="text-[26px] font-semibold leading-tight tracking-tight text-ink">
          {title}
        </h1>
        {subtitle ? (
          <p className="max-w-2xl text-[13.5px] leading-relaxed text-muted">
            {subtitle}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      ) : null}
    </div>
  );
}