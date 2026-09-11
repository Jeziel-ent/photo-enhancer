import { useEffect, useState } from "react";
import adinnIcon from "../../assets/adinn-icon-512.png";
import { cn } from "../../lib/cn";

// Total on-screen time stays inside the ~2.5–2.8s budget: a held reveal
// (with a deliberate, unhurried pause once everything has settled), then a
// smooth cross-fade into the (already-mounted) app underneath.
const HOLD_MS = 2200;
const FADE_MS = 450;

/**
 * One-time opening animation. Purely decorative and non-blocking: the real
 * app is already mounted underneath (see App.tsx), so this is
 * `pointer-events-none` and `aria-hidden` from the start — keyboard/screen
 * reader users reach the real UI immediately, sighted users just see it a
 * beat later under the fading mark.
 */
export function SplashScreen({ onDone }: { onDone: () => void }) {
  const [fading, setFading] = useState(false);

  useEffect(() => {
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const hold = reduceMotion ? 150 : HOLD_MS;
    const fade = reduceMotion ? 0 : FADE_MS;

    const fadeTimer = window.setTimeout(() => setFading(true), hold);
    const doneTimer = window.setTimeout(onDone, hold + fade);
    return () => {
      window.clearTimeout(fadeTimer);
      window.clearTimeout(doneTimer);
    };
  }, [onDone]);

  return (
    <div
      aria-hidden="true"
      className={cn(
        "pointer-events-none fixed inset-0 z-[100] flex flex-col items-center justify-center gap-7 overflow-hidden",
        fading && "animate-fade-out",
      )}
    >
      {/* Full-window Adinn-red canvas: a soft circular flood reveal, then a
          subtle darker/lighter wash for depth (not a flat fill). */}
      <div
        aria-hidden="true"
        className="animate-bg-reveal absolute inset-0"
        style={{
          background:
            "radial-gradient(120% 85% at 28% 12%, rgba(255,255,255,0.12), transparent 55%)," +
            "radial-gradient(130% 100% at 75% 105%, rgba(0,0,0,0.22), transparent 60%)," +
            "linear-gradient(160deg, var(--color-brand) 0%, var(--color-brand-dark) 55%, var(--color-brand-deep) 100%)",
        }}
      />

      <div className="relative flex h-32 w-32 items-center justify-center">
        <span
          aria-hidden="true"
          className="animate-soft-pulse absolute inset-0 rounded-full bg-white/20 blur-2xl"
          style={{ animationDelay: "650ms" }}
        />
        <img
          src={adinnIcon}
          alt=""
          draggable={false}
          className="animate-logo-reveal relative h-28 w-28 select-none drop-shadow-[0_10px_30px_rgba(0,0,0,0.32)]"
          style={{ animationDelay: "150ms" }}
        />
        {/* One-pass diagonal light sweep across the settled icon — a single
            restrained catch of light, not a shine/gloss loop. */}
        <span
          aria-hidden="true"
          className="animate-light-sweep pointer-events-none absolute inset-0 mix-blend-overlay"
          style={{
            animationDelay: "750ms",
            background:
              "linear-gradient(115deg, transparent 40%, rgba(255,255,255,0.6) 50%, transparent 60%)",
          }}
        />
      </div>

      <div className="relative flex flex-col items-center gap-3">
        <span
          className="animate-line-draw h-[2px] w-14 rounded-full bg-white/85"
          style={{ animationDelay: "950ms" }}
        />
        <p
          className="animate-rise-in text-[22px] font-bold tracking-[0.18em] text-white"
          style={{ animationDelay: "1100ms" }}
        >
          ADINN
        </p>
        <p
          className="animate-rise-in text-[11px] font-semibold uppercase tracking-[0.34em] text-white/75"
          style={{ animationDelay: "1230ms" }}
        >
          4K Image Enhancer
        </p>
      </div>
    </div>
  );
}
