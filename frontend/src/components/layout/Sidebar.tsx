import type { ReactNode } from "react";
import { cn } from "../../lib/cn";
import { IconClock, IconHome, IconImage, IconSettings } from "../ui/Icon";

function SidebarButton({
  label,
  active,
  children,
}: {
  label: string;
  active?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex h-11 w-11 items-center justify-center rounded-xl transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70",
        active
          ? "bg-brand text-white shadow-card"
          : "text-muted hover:bg-ink/[0.05] hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}

/**
 * Narrow left nav rail. Only "Home" leads anywhere today (the app has a
 * single route) — the rest are presented per the reference design but are
 * inert until there's a second screen to route to.
 */
export function Sidebar() {
  return (
    <aside className="relative flex w-24 shrink-0 flex-col items-center overflow-hidden border-r border-line bg-white/80 pb-0 pt-5">
      <nav className="flex flex-col items-center gap-3" aria-label="Primary">
        <SidebarButton label="Home" active>
          <IconHome className="h-5 w-5" />
        </SidebarButton>
        <SidebarButton label="Gallery">
          <IconImage className="h-5 w-5" />
        </SidebarButton>
        <SidebarButton label="History">
          <IconClock className="h-5 w-5" />
        </SidebarButton>
        <SidebarButton label="Settings">
          <IconSettings className="h-5 w-5" />
        </SidebarButton>
      </nav>

      <div className="relative mt-auto flex h-36 w-full items-end justify-center">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -bottom-16 -left-10 h-44 w-44 rounded-full"
          style={{
            background:
              "radial-gradient(circle, var(--color-brand) 0%, var(--color-brand-dark) 55%, transparent 78%)",
          }}
        />
        <div className="relative pb-5 text-center">
          <p className="text-[13px] font-bold tracking-tight text-white">Adinn</p>
          <p className="text-[10px] text-white/80">v1.0.0</p>
        </div>
      </div>
    </aside>
  );
}
