import { ADJUSTMENT_RANGE, DEFAULT_ADJUSTMENTS, isDefaultAdjustments, type AdjustmentParams } from "../../lib/adjustments";
import { cn } from "../../lib/cn";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { IconRefresh, IconSliders } from "../ui/Icon";

const SLIDERS: { key: keyof AdjustmentParams; label: string }[] = [
  { key: "brightness", label: "Brightness" },
  { key: "contrast", label: "Contrast" },
  { key: "highlights", label: "Highlights" },
  { key: "shadows", label: "Shadows" },
  { key: "saturation", label: "Saturation" },
  { key: "detail", label: "Detail" },
];

function SliderRow({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (next: number) => void;
}) {
  const zero = Math.round(((0 - min) / (max - min)) * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[12.5px] font-medium text-ink-2">{label}</span>
        <span className="tabular-nums text-[11.5px] text-faint">{value}</span>
      </div>
      <div className="relative">
        {min < 0 ? (
          <span
            aria-hidden="true"
            className="pointer-events-none absolute top-1/2 h-3 w-px -translate-y-1/2 bg-line-strong"
            style={{ left: `${zero}%` }}
          />
        ) : null}
        <input
          type="range"
          aria-label={label}
          min={min}
          max={max}
          step={1}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-line accent-brand"
        />
      </div>
    </div>
  );
}

/**
 * The six manual post-processing sliders (MVP 2), applied AFTER the
 * automatic AI enhancement and never re-running Restormer/SwinIR-M — see
 * image_enhancer/src/adjustments.py, the one deterministic implementation
 * the backend uses for both live preview and final export. Default (0)
 * means no additional change; this component is purely controlled -- the
 * caller (JobResultPanel) owns the value and debounces preview fetches.
 */
export function AdjustmentControls({
  value,
  onChange,
  disabled,
}: {
  value: AdjustmentParams;
  onChange: (next: AdjustmentParams) => void;
  disabled?: boolean;
}) {
  const atDefault = isDefaultAdjustments(value);
  return (
    <Card padding="md" className={cn("animate-fade-in", disabled && "opacity-50")}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-[13px] font-semibold tracking-tight text-ink">
          <IconSliders className="h-4 w-4 text-brand" />
          Adjustments
        </h3>
        <Button
          variant="ghost"
          size="sm"
          disabled={disabled || atDefault}
          icon={<IconRefresh className="h-3.5 w-3.5" />}
          onClick={() => onChange({ ...DEFAULT_ADJUSTMENTS })}
        >
          Reset
        </Button>
      </div>
      <fieldset disabled={disabled} className="space-y-3.5">
        {SLIDERS.map(({ key, label }) => (
          <SliderRow
            key={key}
            label={label}
            value={value[key]}
            min={ADJUSTMENT_RANGE[key].min}
            max={ADJUSTMENT_RANGE[key].max}
            onChange={(next) => onChange({ ...value, [key]: next })}
          />
        ))}
      </fieldset>
    </Card>
  );
}
