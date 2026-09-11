import type { HTMLAttributes } from "react";
import { cn } from "../../lib/cn";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  padding?: "none" | "sm" | "md" | "lg";
  /** "glass" adds the translucent, blurred surface used by the enhancement flow. */
  variant?: "solid" | "glass";
}

const PADDING = {
  none: "",
  sm: "p-4",
  md: "p-5",
  lg: "p-7",
};

const VARIANTS = {
  solid: "bg-panel border border-line shadow-card",
  glass: "border border-white/60 bg-white/70 shadow-glass backdrop-blur-xl",
};

export function Card({ padding = "md", variant = "solid", className, children, ...props }: CardProps) {
  return (
    <div
      className={cn("rounded-xl", VARIANTS[variant], PADDING[padding], className)}
      {...props}
    >
      {children}
    </div>
  );
}