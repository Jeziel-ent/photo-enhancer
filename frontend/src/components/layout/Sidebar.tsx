import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { cn } from "../../lib/cn";
import { IconClock, IconHome } from "../ui/Icon";

function SidebarLink({
  to,
  label,
  children,
}: {
  to: string;
  label: string;
  children: ReactNode;
}) {
  return (
    <NavLink
      to={to}
      end
      aria-label={label}
      className={({ isActive }) =>
        cn(
          "flex h-11 w-full items-center gap-2.5 rounded-xl px-4 text-[13.5px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70",
          isActive
            ? "bg-brand text-white shadow-card"
            : "text-muted hover:bg-ink/[0.05] hover:text-ink",
        )
      }
    >
      {children}
      {label}
    </NavLink>
  );
}

/** Left nav rail: Home and Recent are the app's only two screens. */
export function Sidebar() {
  return (
    <aside className="relative flex w-52 shrink-0 flex-col overflow-hidden border-r border-line bg-white/80 px-3 pb-0 pt-5">
      <nav className="flex flex-col gap-1.5" aria-label="Primary">
        <SidebarLink to="/" label="Home">
          <IconHome className="h-4.5 w-4.5" />
        </SidebarLink>
        <SidebarLink to="/recent" label="Recent">
          <IconClock className="h-4.5 w-4.5" />
        </SidebarLink>
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
