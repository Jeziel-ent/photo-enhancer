import { Link } from "react-router-dom";
import adinnLogo from "../../assets/adinn-logo.png";

export function AppHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-[1180px] items-center justify-between px-6">
        <div className="flex items-center gap-3.5">
          <Link to="/" className="flex items-center gap-3.5" aria-label="Adinn home">
            <img
              src={adinnLogo}
              alt="Adinn"
              className="h-7 w-auto select-none"
              draggable={false}
            />
            <span className="hidden h-7 w-px bg-line sm:block" aria-hidden="true" />
            <span className="hidden flex-col sm:flex">
              <span className="text-[13.5px] font-semibold leading-tight tracking-tight text-ink">
                4K Image Enhancer
              </span>
              <span className="text-[11px] leading-tight text-muted">
                Adinn
              </span>
            </span>
          </Link>
        </div>
      </div>
    </header>
  );
}