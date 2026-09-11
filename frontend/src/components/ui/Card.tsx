import type { HTMLAttributes } from "react";
import { cn } from "../../lib/cn";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  padding?: "none" | "sm" | "md" | "lg";
}

const PADDING = {
  none: "",
  sm: "p-4",
  md: "p-5",
  lg: "p-7",
};

export function Card({ padding = "md", className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-lg bg-panel border border-line shadow-card",
        PADDING[padding],
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}