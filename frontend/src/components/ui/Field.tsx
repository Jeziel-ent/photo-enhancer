import type {
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";
import { forwardRef } from "react";
import { cn } from "../../lib/cn";

const CONTROL_BASE =
  "w-full rounded-md border bg-white px-3 text-sm text-ink placeholder:text-faint " +
  "transition-colors focus:outline-none focus:ring-2 focus:ring-brand/25 focus:border-brand " +
  "disabled:bg-canvas disabled:text-muted";

export const inputClass = (hasError: boolean) =>
  cn(CONTROL_BASE, hasError ? "border-brand" : "border-line-strong");

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }>(
  function Input({ className, invalid, ...props }, ref) {
    return (
      <input
        ref={ref}
        className={cn(inputClass(Boolean(invalid)), "h-9.5", className)}
        {...props}
      />
    );
  },
);

export const TextArea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement> & { invalid?: boolean }>(
  function TextArea({ className, invalid, ...props }, ref) {
    return (
      <textarea
        ref={ref}
        className={cn(inputClass(Boolean(invalid)), "py-2 min-h-24", className)}
        {...props}
      />
    );
  },
);

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement> & { invalid?: boolean }>(
  function Select({ className, invalid, children, ...props }, ref) {
    return (
      <select
        ref={ref}
        className={cn(inputClass(Boolean(invalid)), "h-9.5 pr-8", className)}
        {...props}
      >
        {children}
      </select>
    );
  },
);

export interface FieldProps {
  label: string;
  required?: boolean;
  htmlFor?: string;
  hint?: ReactNode;
  error?: string | null;
  children: ReactNode;
  className?: string;
}

export function Field({
  label,
  required,
  htmlFor,
  hint,
  error,
  children,
  className,
}: FieldProps) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <label
        htmlFor={htmlFor}
        className="flex items-baseline justify-between text-[13px] font-medium text-ink-2"
      >
        <span>
          {label}
          {required ? <span className="text-brand ml-0.5">*</span> : null}
        </span>
        {hint ? (
          <span className="text-xs font-normal text-faint">{hint}</span>
        ) : null}
      </label>
      {children}
      {error ? (
        <p className="text-xs text-brand mt-1" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}