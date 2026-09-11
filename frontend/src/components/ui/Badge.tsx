import type { ReactNode } from "react";
import { cn } from "../../lib/cn";

type Tone = "neutral" | "ok" | "warn" | "brand";

const TONES: Record<Tone, string> = {
  neutral: "bg-ink/[0.04] text-ink-2 border border-ink/[0.08]",
  ok: "bg-ok-soft text-ok border border-ok/15",
  warn: "bg-warn-soft text-warn border border-warn/15",
  brand: "bg-brand-soft text-brand-dark border border-brand/15",
};

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-medium tracking-wide",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}