"use client";

import {
  Aperture,
  ChevronLeft,
  ChevronRight,
  Dices,
  Gauge,
  Layers,
  RotateCcw,
  Shapes,
  Sparkles,
  Sun,
  Thermometer,
  Upload,
  Waves,
} from "lucide-react";
import { useRef, useState } from "react";
import { GEOMETRY, MAGNIFICATION, NOISE_GROUPS, SOURCE } from "@/lib/params";
import type { Defaults, ObjectSpec, ParamDict, ParamValue, ProfileId } from "@/lib/types";
import { ArrangementCard } from "./ArrangementCard";
import { Badge, NumberField, Panel, Segmented, SliderField, Toggle, cx } from "./ui";

export const PROFILE_IDS: ProfileId[] = ["A", "B", "C", "D"];

export type NoisePreset = "off" | "mild" | "moderate" | "strong";
export const PRESET_SCALE: Record<NoisePreset, number> = { off: 0, mild: 0.5, moderate: 1, strong: 2 };

const num = (v: unknown, fallback = 0) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);

export function ArrangementsPanel({
  defaults,
  selected,
  onChange,
  archEnlarge,
  onToggleArchEnlarge,
}: {
  defaults: Defaults;
  selected: ProfileId[];
  onChange: (next: ProfileId[]) => void;
  archEnlarge?: ProfileId | null;
  onToggleArchEnlarge?: (id: ProfileId) => void;
}) {
  const toggle = (id: ProfileId) =>
    onChange(selected.includes(id) ? selected.filter((p) => p !== id) : PROFILE_IDS.filter((p) => p === id || selected.includes(p)));
  return (
    <Panel
      icon={<Layers className="h-4 w-4" />}
      title="Arrangements"
      subtitle="Architecture × imaging mode"
      right={
        <button
          type="button"
          className="text-[11px] text-sky-300 hover:text-sky-200"
          onClick={() => onChange(selected.length === 4 ? [] : [...PROFILE_IDS])}
        >
          {selected.length === 4 ? "Clear" : "All"}
        </button>
      }
    >
      <div className="scroll-thin max-h-[min(42vh,360px)] overflow-y-auto pr-0.5">
        <div className="grid grid-cols-2 gap-2">
          {PROFILE_IDS.map((id) => (
            <ArrangementCard
              key={id}
              profile={defaults.profiles[id]}
              selected={selected.includes(id)}
              onToggle={() => toggle(id)}
              archEnlarged={archEnlarge === id}
              onToggleArchEnlarge={onToggleArchEnlarge ? () => onToggleArchEnlarge(id) : undefined}
            />
          ))}
        </div>
      </div>
    </Panel>
  );
}

type ObjectSourceTab = ObjectSpec["source"];

function fileToPngBase64(file: File): Promise<{ png_base64: string; file_name: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("failed to read file"));
    reader.onload = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      const png_base64 = comma >= 0 ? result.slice(comma + 1) : result;
      if (!png_base64) {
        reject(new Error("empty file"));
        return;
      }
      resolve({ png_base64, file_name: file.name || "upload.png" });
    };
    reader.readAsDataURL(file);
  });
}

export function ObjectPanel({
  object,
  onChange,
  maxIndex,
}: {
  object: ObjectSpec;
  onChange: (next: ObjectSpec) => void;
  maxIndex: number;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const index = object.source === "mnist" ? object.mnist_index : 5;
  const maskId = object.source === "mask" ? object.mask_id : 5;
  const setIndex = (i: number) =>
    onChange({ source: "mnist", mnist_index: Math.min(maxIndex - 1, Math.max(0, Math.round(i))) });

  const setSource = (source: ObjectSourceTab) => {
    setUploadError(null);
    if (source === "mnist") onChange({ source, mnist_index: index });
    else if (source === "mask") onChange({ source, mask_id: maskId });
    else if (source === "rabbit") onChange({ source: "rabbit" });
    else if (source === "upload") {
      if (object.source === "upload") return;
      onChange({ source: "upload", png_base64: "", file_name: undefined });
    }
  };

  const onPickFile = async (file: File | null) => {
    if (!file) return;
    setUploadError(null);
    try {
      if (!file.type.startsWith("image/")) {
        throw new Error("please choose an image file (PNG preferred)");
      }
      const { png_base64, file_name } = await fileToPngBase64(file);
      onChange({ source: "upload", png_base64, file_name });
    } catch (e) {
      setUploadError((e as Error).message);
    }
  };

  return (
    <Panel icon={<Shapes className="h-4 w-4" />} title="SLM object" subtitle="Amplitude pattern written on the SLM">
      <div className="grid grid-cols-2 gap-1 rounded-lg bg-ink-900/80 p-1 ring-1 ring-white/5">
        {(
          [
            { value: "mnist", label: "MNIST" },
            { value: "mask", label: "Catalog" },
            { value: "rabbit", label: "Rabbit" },
            { value: "upload", label: "Upload" },
          ] as const
        ).map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => setSource(opt.value)}
            className={cx(
              "rounded-md px-2 py-1.5 text-[11px] font-medium transition-all",
              object.source === opt.value
                ? "bg-ink-700 text-white shadow ring-1 ring-white/10"
                : "text-ink-400 hover:text-ink-200",
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {object.source === "mnist" && (
        <div className="flex items-end gap-2">
          <div className="min-w-0 flex-1">
            <NumberField
              label="MNIST index"
              value={index}
              min={0}
              max={maxIndex - 1}
              onChange={setIndex}
              hint="Index into the shipped MNIST subset. The digit label is not the index."
            />
          </div>
          <div className="flex gap-1 pb-0.5">
            {[
              { icon: <ChevronLeft className="h-4 w-4" />, title: "Previous", on: () => setIndex(index - 1) },
              { icon: <ChevronRight className="h-4 w-4" />, title: "Next", on: () => setIndex(index + 1) },
              { icon: <Dices className="h-4 w-4" />, title: "Random", on: () => setIndex(Math.floor(Math.random() * maxIndex)) },
            ].map((b) => (
              <button
                key={b.title}
                type="button"
                title={b.title}
                onClick={b.on}
                className="grid h-9 w-9 place-items-center rounded-lg bg-ink-800 text-ink-300 ring-1 ring-white/5 hover:bg-ink-700 hover:text-white"
              >
                {b.icon}
              </button>
            ))}
          </div>
        </div>
      )}

      {object.source === "mask" && (
        <div className="grid grid-cols-5 gap-1.5">
          {Array.from({ length: 10 }, (_, i) => i + 1).map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => onChange({ source: "mask", mask_id: id })}
              className={cx(
                "rounded-lg py-2 font-mono text-sm transition-all",
                object.mask_id === id
                  ? "bg-sky-400 font-bold text-ink-950"
                  : "bg-ink-800 text-ink-300 ring-1 ring-white/5 hover:bg-ink-700",
              )}
            >
              {id}
            </button>
          ))}
        </div>
      )}

      {object.source === "rabbit" && (
        <div className="rounded-lg bg-ink-900/60 p-3 text-[11px] leading-relaxed text-ink-300 ring-1 ring-white/5">
          Built-in rabbit amplitude mask from <code className="break-all text-sky-300">public/test-masks/rabbit-slm-mask.png</code>.
          Resized to the simulation grid with aspect ratio preserved and zero padding.
        </div>
      )}

      {object.source === "upload" && (
        <div className="space-y-2">
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp,.png,.jpg,.jpeg,.webp,.bmp"
            className="hidden"
            onChange={(e) => void onPickFile(e.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-ink-800 px-3 py-2.5 text-[12px] text-ink-200 ring-1 ring-white/5 hover:bg-ink-700 hover:text-white"
          >
            <Upload className="h-4 w-4" />
            {object.png_base64 ? "Replace mask image" : "Upload SLM mask"}
          </button>
          {object.file_name && (
            <p className="break-all font-mono text-[11px] text-sky-300">{object.file_name}</p>
          )}
          {!object.png_base64 && (
            <p className="text-[11px] text-ink-400">Choose a grayscale / binary amplitude mask (not an RGB photo).</p>
          )}
          {uploadError && <p className="text-[11px] text-rose-300">{uploadError}</p>}
        </div>
      )}
    </Panel>
  );
}

export function RunSettingsPanel({
  seed,
  size,
  modes,
  sizeRange,
  onSeed,
  onSize,
  onModes,
}: {
  seed: number;
  size: number;
  modes: { ideal: boolean; noisy: boolean };
  sizeRange: [number, number];
  onSeed: (v: number) => void;
  onSize: (v: number) => void;
  onModes: (v: { ideal: boolean; noisy: boolean }) => void;
}) {
  return (
    <Panel icon={<Gauge className="h-4 w-4" />} title="Run settings" subtitle="Grid, RNG seed and output modes">
      <div className="grid grid-cols-2 gap-2">
        <NumberField label="Grid size" value={size} min={sizeRange[0]} max={sizeRange[1]} step={8} unit="px"
          onChange={(v) => onSize(Math.round(v))} hint={`Square simulation grid, ${sizeRange[0]}–${sizeRange[1]} px.`} />
        <NumberField label="Seed" value={seed} min={0} step={1} onChange={(v) => onSeed(Math.round(v))}
          hint="Random seed for detector and drift noise; same seed → identical noisy image." />
      </div>
      <div className="rounded-lg bg-ink-900/60 p-2 ring-1 ring-white/5">
        <Toggle label="Ideal" description="All optical, drift and detector noise off" checked={modes.ideal}
          color="bg-emerald-400" onChange={(ideal) => onModes({ ...modes, ideal })} />
        <Toggle label="Noisy" description="Drift latents, optical blur and EMCCD noise on" checked={modes.noisy}
          color="bg-rose-400" onChange={(noisy) => onModes({ ...modes, noisy })} />
      </div>
    </Panel>
  );
}

export function SourcePanel({
  mode,
  onMode,
  latents,
  onChange,
}: {
  mode: "ideal" | "noisy";
  onMode: (m: "ideal" | "noisy") => void;
  latents: ParamDict;
  onChange: (key: string, value: number) => void;
}) {
  return (
    <Panel icon={<Sun className="h-4 w-4" />} title="Source & exposure" subtitle="SPDC brightness and camera integration" collapsible>
      <Segmented
        value={mode}
        onChange={onMode}
        options={[
          { value: "ideal", label: "Ideal run" },
          { value: "noisy", label: "Noisy run" },
        ]}
      />
      {SOURCE.map((p) => (
        <SliderField key={p.key} param={p} value={num(latents[p.key], p.min)} onChange={(v) => onChange(p.key, v)}
          accent={mode === "ideal" ? "#34d399" : "#fb7185"} />
      ))}
    </Panel>
  );
}

export function GeometryPanel({ setup, onChange }: { setup: ParamDict; onChange: (key: string, value: number) => void }) {
  return (
    <Panel icon={<Aperture className="h-4 w-4" />} title="Geometry" subtitle="Applied to both ideal and noisy runs" collapsible>
      {GEOMETRY.map((p) => (
        <SliderField key={p.key} param={p} value={num(setup[p.key], p.min)} onChange={(v) => onChange(p.key, v)} />
      ))}
      {(["near_field", "far_field"] as const).map((mode) => (
        <div key={mode} className="rounded-lg bg-ink-900/50 p-2.5 ring-1 ring-white/5">
          <div className="mb-2 flex items-center gap-2">
            <Badge tone={mode === "far_field" ? "violet" : "sky"}>{mode === "far_field" ? "Far field · A, C" : "Near field · B, D"}</Badge>
          </div>
          <div className="space-y-3">
            {MAGNIFICATION[mode].map((p) => (
              <SliderField key={p.key} param={p} value={num(setup[p.key], 1)} onChange={(v) => onChange(p.key, v)} />
            ))}
          </div>
        </div>
      ))}
    </Panel>
  );
}

export function ThermalPanel({
  defaults,
  selected,
  thermal,
  onChange,
  onReset,
}: {
  defaults: Defaults;
  selected: ProfileId[];
  thermal: Record<ProfileId, Record<string, number>>;
  onChange: (profile: ProfileId, key: string, value: number) => void;
  onReset: () => void;
}) {
  return (
    <Panel
      icon={<Thermometer className="h-4 w-4" />}
      title="Crystal temperature"
      subtitle="T_crystal = T_pm0 + ΔT per crystal"
      collapsible
      right={
        <span
          role="button"
          tabIndex={0}
          title="Reset to profile defaults"
          onClick={(e) => {
            e.stopPropagation();
            onReset();
          }}
          className="rounded-md p-1 text-ink-400 hover:bg-ink-700 hover:text-white"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </span>
      }
    >
      {selected.length === 0 && <p className="text-[11px] text-ink-400">Select an arrangement to edit its crystals.</p>}
      {selected.map((id) => {
        const profile = defaults.profiles[id];
        return (
          <div key={id} className="rounded-lg bg-ink-900/50 p-2.5 ring-1 ring-white/5">
            <div className="mb-2 flex items-center gap-2">
              <span className="grid h-5 w-5 place-items-center rounded bg-ink-700 font-mono text-[11px] font-bold text-white">{id}</span>
              <span className="text-[11px] text-ink-300">{profile.label}</span>
            </div>
            {profile.crystals.map((crystal) => (
              <div key={crystal} className="grid grid-cols-[2.5rem_1fr_1fr] items-center gap-2 py-0.5">
                <span className="font-mono text-[11px] text-violet-300">{crystal}</span>
                <NumberField label="T_pm0" unit="°C" step={0.1} value={thermal[id]?.[`T_pm0_${crystal}_C`] ?? 0}
                  onChange={(v) => onChange(id, `T_pm0_${crystal}_C`, v)} />
                <NumberField label="ΔT" unit="°C" step={0.05} value={thermal[id]?.[`dT_${crystal}_C`] ?? 0}
                  onChange={(v) => onChange(id, `dT_${crystal}_C`, v)} />
              </div>
            ))}
          </div>
        );
      })}
    </Panel>
  );
}

export function NoisePanel({
  setup,
  latents,
  preset,
  onPreset,
  onSetup,
  onLatent,
}: {
  setup: ParamDict;
  latents: ParamDict;
  preset: NoisePreset | "custom";
  onPreset: (p: NoisePreset) => void;
  onSetup: (key: string, value: ParamValue) => void;
  onLatent: (key: string, value: number) => void;
}) {
  return (
    <Panel icon={<Waves className="h-4 w-4" />} title="Noise model" subtitle="Noisy run only — ideal keeps everything at zero" collapsible>
      <div className="grid grid-cols-4 gap-1 rounded-lg bg-ink-900/80 p-1 ring-1 ring-white/5">
        {(["off", "mild", "moderate", "strong"] as NoisePreset[]).map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPreset(p)}
            className={cx(
              "rounded-md py-1.5 text-[11px] font-medium capitalize transition-all",
              preset === p ? "bg-rose-500/20 text-rose-200 ring-1 ring-rose-400/40" : "text-ink-400 hover:text-ink-200",
            )}
          >
            {p}
          </button>
        ))}
      </div>
      {preset === "custom" && (
        <p className="flex items-center gap-1.5 text-[11px] text-amber-300/90">
          <Sparkles className="h-3 w-3" /> Custom values
        </p>
      )}
      {NOISE_GROUPS.map((group) => (
        <div key={group.id} className="rounded-lg bg-ink-900/50 p-2.5 ring-1 ring-white/5">
          <div className="mb-2">
            <div className="text-[12px] font-semibold text-ink-200">{group.title}</div>
            <div className="text-[10px] text-ink-400">{group.description}</div>
          </div>
          <div className="space-y-3">
            {group.numeric.map((p) => {
              const source = p.target === "setup" ? setup : latents;
              return (
                <SliderField
                  key={p.key}
                  param={p}
                  accent="#fb7185"
                  value={num(source[p.key], 0)}
                  onChange={(v) => (p.target === "setup" ? onSetup(p.key, v) : onLatent(p.key, v))}
                />
              );
            })}
            {group.toggles?.map((t) => (
              <Toggle key={t.key} label={t.label} checked={Boolean(setup[t.key])} color="bg-rose-400"
                onChange={(v) => onSetup(t.key, v)} />
            ))}
          </div>
        </div>
      ))}
      <Toggle
        label="SPDC correlation PSF (double blur)"
        description="Physical two-photon blur; washes out fine detail"
        checked={setup.blur_model === "double_blur"}
        color="bg-rose-400"
        onChange={(v) => onSetup("blur_model", v ? "double_blur" : "single_blur")}
      />
    </Panel>
  );
}
