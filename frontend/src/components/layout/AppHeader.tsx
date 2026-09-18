import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import adinnLogo from "../../assets/adinn-logo.png";
import { IconHelpCircle, IconSettings, IconShieldCheck } from "../ui/Icon";

function IconButton({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      className="flex h-8 w-8 items-center justify-center rounded-md text-muted transition-colors hover:bg-ink/[0.05] hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand/70"
    >
      {children}
    </button>
  );
}

export function AppHeader() {
  return (
    <header className="relative z-40 flex h-[76px] w-full shrink-0 items-center justify-between border-b border-line/70 bg-white/70 px-6 backdrop-blur-xl">
      <Link to="/" className="flex items-center gap-4" aria-label="Adinn home">
        <img src={adinnLogo} alt="Adinn" className="h-9 w-auto select-none" draggable={false} />
        <span className="h-9 w-px bg-line" aria-hidden="true" />
        <span className="flex flex-col gap-0.5">
          <span className="text-[16px] font-semibold leading-tight tracking-tight text-ink">
            4K Image Enhancer
          </span>
          <span className="text-[11px] tracking-wide text-faint">
            Enhance · Denoise · Upscale
          </span>
        </span>
      </Link>

      <div className="flex items-center gap-4">
        <span className="flex items-center gap-1.5 rounded-full border border-ok/15 bg-ok-soft px-3 py-1.5 text-[12px] font-medium text-ok">
          <IconShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
          Processed Locally
        </span>
        <span className="h-6 w-px bg-line" aria-hidden="true" />
        <IconButton label="Settings">
          <IconSettings className="h-[18px] w-[18px]" />
        </IconButton>
        <span className="h-6 w-px bg-line" aria-hidden="true" />
        <IconButton label="Help">
          <IconHelpCircle className="h-[18px] w-[18px]" />
        </IconButton>
      </div>
    </header>
  );
}
