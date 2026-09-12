import { useEffect, useState } from "react";
import adinnIcon from "../../assets/adinn-icon-512.png";
import { cn } from "../../lib/cn";

// Total on-screen time stays inside the ~2.5–2.8s budget: a held reveal
// (with a deliberate, unhurried pause once everything has settled), then a
// smooth cross-fade into the (already-mounted) app underneath.
const HOLD_MS = 2200;
const FADE_MS = 450;

const ORBIT_PARTICLES = [0, 120, 240];

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
        "pointer-events-none fixed inset-0 z-[100] flex flex-col items-center justify-center gap-8 overflow-hidden",
        fading && "animate-fade-out",
      )}
    >
      {/* Full-window Adinn-red canvas: a soft circular flood reveal, a
          top-left highlight wash, and a darker vignette toward the edges
          for depth (not a flat fill). */}
      <div
        aria-hidden="true"
        className="animate-bg-reveal absolute inset-0"
        style={{
          background:
            "radial-gradient(120% 85% at 28% 12%, rgba(255,255,255,0.12), transparent 55%)," +
            "radial-gradient(85% 70% at 50% 45%, transparent 55%, rgba(0,0,0,0.30) 100%)," +
            "radial-gradient(130% 100% at 75% 105%, rgba(0,0,0,0.22), transparent 60%)," +
            "linear-gradient(160deg, var(--color-brand) 0%, var(--color-brand-dark) 55%, var(--color-brand-deep) 100%)",
        }}
      />

      {/* Large, very-slow-drifting glowing ribbon curves anchored near the
          bottom corners — ambient background motion only. */}
      <div
        aria-hidden="true"
        className="animate-ribbon-drift absolute -bottom-32 -left-24 h-96 w-[34rem] rounded-[50%] opacity-60 blur-3xl"
        style={{
          background:
            "radial-gradient(closest-side, rgba(255,138,61,0.35), rgba(209,33,38,0.18) 60%, transparent 80%)",
        }}
      />
      <div
        aria-hidden="true"
        className="animate-ribbon-drift absolute -right-24 -bottom-40 h-[26rem] w-[38rem] rounded-[50%] opacity-50 blur-3xl"
        style={{
          animationDelay: "-8s",
          background:
            "radial-gradient(closest-side, rgba(255,255,255,0.16), rgba(130,14,19,0.35) 55%, transparent 80%)",
        }}
      />

      {/* Soft ambient bloom behind the whole mark. */}
      <span
        aria-hidden="true"
        className="animate-soft-pulse absolute h-[26rem] w-[26rem] rounded-full bg-white/10 blur-[90px]"
      />

      <div className="relative flex h-44 w-44 items-center justify-center">
        {/* Glowing 3D platform the icon appears to float above. */}
        <span
          aria-hidden="true"
          className="animate-soft-pulse absolute bottom-2 h-6 w-28 rounded-[50%] bg-white/70 blur-md"
          style={{ animationDelay: "300ms" }}
        />
        <span
          aria-hidden="true"
          className="absolute bottom-3 h-3 w-24 rounded-[50%]"
          style={{
            background:
              "radial-gradient(closest-side, rgba(255,255,255,0.55), rgba(255,138,61,0.25) 70%, transparent 100%)",
          }}
        />
        {/* Soft red glow pooling under the platform. */}
        <span
          aria-hidden="true"
          className="animate-soft-pulse absolute bottom-0 h-10 w-36 rounded-[50%] bg-brand-deep/50 blur-xl"
          style={{ animationDelay: "150ms" }}
        />

        {/* Thin orbit ring with 3 small glowing particles, rotating slowly. */}
        <div className="animate-orbit-spin absolute inset-2 rounded-full border border-white/25">
          {ORBIT_PARTICLES.map((deg) => (
            <span
              key={deg}
              aria-hidden="true"
              className="absolute top-1/2 left-1/2 h-1.5 w-1.5 rounded-full bg-white shadow-[0_0_8px_2px_rgba(255,255,255,0.65)]"
              style={{ transform: `rotate(${deg}deg) translate(78px) rotate(-${deg}deg)` }}
            />
          ))}
        </div>
        <div className="animate-orbit-spin-slow absolute inset-6 rounded-full border border-dashed border-white/15" />

        {/* Icon floats gently up/down above the platform. */}
        <div className="animate-float relative flex h-28 w-28 items-center justify-center">
          <span
            aria-hidden="true"
            className="animate-soft-pulse absolute inset-0 rounded-full bg-white/20 blur-2xl"
            style={{ animationDelay: "650ms" }}
          />
          <img
            src={adinnIcon}
            alt=""
            draggable={false}
            className="animate-logo-reveal relative h-28 w-28 select-none drop-shadow-[0_16px_28px_rgba(0,0,0,0.4)]"
            style={{ animationDelay: "150ms" }}
          />
          {/* One-pass diagonal light sweep across the settled icon — a
              single restrained catch of light, not a shine/gloss loop. */}
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
      </div>

      <div className="relative flex flex-col items-center gap-3">
        <span
          className="animate-line-draw h-[2px] w-14 rounded-full bg-white/85"
          style={{ animationDelay: "950ms" }}
        />
        <p
          className="animate-rise-in text-[26px] font-bold tracking-[0.18em] text-white"
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

      {/* Glowing indeterminate loading bar + label. */}
      <div
        className="animate-rise-in relative flex flex-col items-center gap-2"
        style={{ animationDelay: "1380ms" }}
      >
        <div className="relative h-[3px] w-40 overflow-hidden rounded-full bg-white/15">
          <span
            aria-hidden="true"
            className="animate-loading-sweep absolute inset-y-0 w-1/3 rounded-full bg-white shadow-[0_0_10px_2px_rgba(255,255,255,0.75)]"
          />
        </div>
        <p className="text-[10px] font-semibold uppercase tracking-[0.3em] text-white/60">
          Loading…
        </p>
      </div>

      <p
        className="animate-rise-in absolute bottom-8 text-[10px] font-semibold uppercase tracking-[0.32em] text-white/50"
        style={{ animationDelay: "1500ms" }}
      >
        Enhance&nbsp;&nbsp;•&nbsp;&nbsp;Denoise&nbsp;&nbsp;•&nbsp;&nbsp;Upscale
      </p>
    </div>
  );
}
