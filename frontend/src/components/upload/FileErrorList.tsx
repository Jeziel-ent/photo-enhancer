import type { JobFileError } from "../../lib/api";
import { IconAlertTriangle } from "../ui/Icon";

export function FileErrorList({ errors, heading }: { errors: JobFileError[]; heading: string }) {
  if (errors.length === 0) return null;
  return (
    <div className="space-y-1.5 rounded-lg border border-warn/20 bg-warn-soft px-3 py-2.5">
      <p className="text-[12.5px] font-medium text-warn">{heading}</p>
      <ul className="space-y-1">
        {errors.map((err) => (
          <li key={err.filename} className="flex items-start gap-1.5 text-[12px] text-warn/90">
            <IconAlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
            <span>
              <span className="font-medium">{err.filename}:</span> {err.message}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
