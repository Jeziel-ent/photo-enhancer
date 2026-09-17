import { useEffect, useState } from "react";
import { Card } from "../ui/Card";
import { Field, Select } from "../ui/Field";
import { IconAlertTriangle, IconCheckCircle } from "../ui/Icon";
import {
  ApiError,
  getSettings,
  updateSettings,
  type ProcessingDevicePreference,
  type SettingsResponse,
} from "../../lib/api";

const DEVICE_LABELS: Record<ProcessingDevicePreference, string> = {
  auto: "Auto (Recommended)",
  gpu: "GPU (NVIDIA CUDA)",
  cpu: "CPU",
};

function detectedLabel(settings: SettingsResponse | null): string {
  if (!settings) return "Detecting...";
  return settings.detected_gpu
    ? `NVIDIA ${settings.detected_gpu}`.replace(/^NVIDIA NVIDIA/, "NVIDIA")
    : "No supported NVIDIA GPU detected";
}

function activeLabel(settings: SettingsResponse | null): string {
  if (!settings || settings.effective_device === null) return "Detecting...";
  return settings.effective_device === "gpu" ? "GPU" : "CPU";
}

export function SettingsPage() {
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justSaved, setJustSaved] = useState(false);

  const load = async () => {
    try {
      const next = await getSettings();
      setSettings(next);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load settings.");
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const handleChange = async (value: ProcessingDevicePreference) => {
    setSaving(true);
    setError(null);
    setJustSaved(false);
    try {
      const next = await updateSettings(value);
      setSettings(next);
      setJustSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save this setting.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-xl space-y-4">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight text-ink">Settings</h1>
        <p className="text-[13.5px] text-muted">
          Choose which hardware processes your images.
        </p>
      </div>

      <Card>
        <Field
          label="Processing Device"
          htmlFor="processing-device"
          hint={saving ? "Saving..." : undefined}
        >
          <Select
            id="processing-device"
            value={settings?.processing_device ?? "auto"}
            disabled={!settings || saving}
            onChange={(e) => void handleChange(e.target.value as ProcessingDevicePreference)}
          >
            {(Object.keys(DEVICE_LABELS) as ProcessingDevicePreference[]).map((key) => (
              <option key={key} value={key}>
                {DEVICE_LABELS[key]}
              </option>
            ))}
          </Select>
        </Field>

        <dl className="mt-4 space-y-1.5 text-[13px]">
          <div className="flex items-center justify-between">
            <dt className="text-muted">Detected</dt>
            <dd className="font-medium text-ink-2">{detectedLabel(settings)}</dd>
          </div>
          <div className="flex items-center justify-between">
            <dt className="text-muted">Currently active</dt>
            <dd className="font-medium text-ink-2">{activeLabel(settings)}</dd>
          </div>
        </dl>

        {settings?.warning ? (
          <div className="mt-4 flex items-start gap-1.5 rounded-md border border-brand/20 bg-brand-soft px-3 py-2.5 text-[12.5px] text-brand-dark">
            <IconAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{settings.warning}</span>
          </div>
        ) : null}

        {settings?.restart_required ? (
          <div className="mt-4 flex items-start gap-1.5 rounded-md border border-line-strong bg-canvas px-3 py-2.5 text-[12.5px] text-ink-2">
            <IconAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>Restart the app for this change to take effect.</span>
          </div>
        ) : null}

        {justSaved && !settings?.restart_required ? (
          <div className="mt-4 flex items-center gap-1.5 text-[12.5px] text-ink-2">
            <IconCheckCircle className="h-3.5 w-3.5 shrink-0 text-brand" />
            <span>Saved.</span>
          </div>
        ) : null}

        {error ? (
          <div className="mt-4 flex items-start gap-1.5 rounded-md border border-brand/20 bg-brand-soft px-3 py-2.5 text-[12.5px] text-brand-dark">
            <IconAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
