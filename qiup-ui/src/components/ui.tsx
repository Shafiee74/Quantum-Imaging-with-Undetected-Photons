"use client";

import { ChevronDown, Info } from "lucide-react";
import { useState, type ReactNode } from "react";
import { formatValue, type NumericParam } from "@/lib/params";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function Panel({
  icon,
  title,
  subtitle,
  right,
  children,
  collapsible = false,
  defaultOpen = true,
}: {
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const header = (
    <div className="flex items-center gap-3">
      {icon && (
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-ink-800 text-accent ring-1 ring-white/5">
          {icon}
        </div>
      )}
      <div className="min-w-0 flex-1 text-left">
        <h3 className="text-[13px] font-semibold tracking-wide text-white">{title}</h3>
        {subtitle && <p className="truncate text-[11px] text-ink-400">{subtitle}</p>}
      </div>
      {right}
      {collapsible && (
        <ChevronDown className={cx("h-4 w-4 text-ink-400 transition-transform", open && "rotate-180")} />
      )}
    </div>
  );
  return (
    <section className="glass rounded-2xl p-4">
      {collapsible ? (
        <button type="button" className="w-full" onClick={() => setOpen((o) => !o)}>
          {header}
        </button>
      ) : (
        header
      )}
      {(!collapsible || open) && <div className="mt-4 space-y-3">{children}</div>}
    </section>
  );
}

export function Hint({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <Info className="h-3 w-3 text-ink-400 transition-colors group-hover:text-accent" />
      <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-white/10 bg-ink-900 px-3 py-2 text-[11px] leading-relaxed text-ink-200 opacity-0 shadow-xl transition-opacity group-hover:opacity-100">
        {text}
      </span>
    </span>
  );
}

export function SliderField({
  param,
  value,
  onChange,
  disabled,
  accent,
}: {
  param: NumericParam;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  accent?: string;
}) {
  const clamped = Math.min(param.max, Math.max(param.min, value));
  const fill = ((clamped - param.min) / (param.max - param.min)) * 100;
  return (
    <div className={cx("group", disabled && "pointer-events-none opacity-40")}>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
          {param.label}
          {param.hint && <Hint text={param.hint} />}
        </label>
        <div className="flex items-center gap-1 rounded-md bg-ink-900/80 px-1.5 py-0.5 ring-1 ring-white/5 focus-within:ring-accent/60">
          <input
            type="number"
            className="w-16 bg-transparent text-right font-mono text-[12px] text-white outline-none"
            value={formatValue(value, param.step)}
            min={param.min}
            max={param.max}
            step={param.step}
            onChange={(e) => {
              const next = Number(e.target.value);
              if (Number.isFinite(next)) onChange(param.integer ? Math.round(next) : next);
            }}
          />
          {param.unit && <span className="text-[10px] text-ink-400">{param.unit}</span>}
        </div>
      </div>
      <input
        type="range"
        min={param.min}
        max={param.max}
        step={param.step}
        value={clamped}
        style={{ ["--fill" as string]: `${fill}%`, ...(accent ? { ["--color-accent" as string]: accent } : {}) }}
        onChange={(e) => {
          const next = Number(e.target.value);
          onChange(param.integer ? Math.round(next) : next);
        }}
      />
    </div>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  hint?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-ink-400">
        {label}
        {hint && <Hint text={hint} />}
      </span>
      <div className="flex items-center gap-2 rounded-lg bg-ink-900/80 px-3 py-2 ring-1 ring-white/5 focus-within:ring-accent/60">
        <input
          type="number"
          className="w-full bg-transparent font-mono text-sm text-white outline-none"
          value={Number.isFinite(value) ? value : ""}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            const next = Number(e.target.value);
            if (Number.isFinite(next)) onChange(next);
          }}
        />
        {unit && <span className="text-[11px] text-ink-400">{unit}</span>}
      </div>
    </label>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
  description,
  color = "bg-accent",
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  description?: string;
  color?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-3 rounded-lg px-1 py-1 text-left"
    >
      <span>
        <span className="block text-[12px] text-ink-200">{label}</span>
        {description && <span className="block text-[11px] text-ink-400">{description}</span>}
      </span>
      <span
        className={cx(
          "relative h-5 w-9 shrink-0 rounded-full transition-colors",
          checked ? color : "bg-ink-700",
        )}
      >
        <span
          className={cx(
            "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all",
            checked ? "left-[18px]" : "left-0.5",
          )}
        />
      </span>
    </button>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: Array<{ value: T; label: ReactNode }>;
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex rounded-lg bg-ink-900/80 p-1 ring-1 ring-white/5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={cx(
            "flex-1 rounded-md px-3 py-1.5 text-[12px] font-medium transition-all",
            value === option.value
              ? "bg-ink-700 text-white shadow ring-1 ring-white/10"
              : "text-ink-400 hover:text-ink-200",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "green" | "violet" | "rose" | "sky" | "amber" }) {
  const tones: Record<string, string> = {
    slate: "bg-ink-800 text-ink-300 ring-white/10",
    green: "bg-emerald-500/10 text-emerald-300 ring-emerald-400/20",
    violet: "bg-violet-500/10 text-violet-300 ring-violet-400/20",
    rose: "bg-rose-500/10 text-rose-300 ring-rose-400/20",
    sky: "bg-sky-500/10 text-sky-300 ring-sky-400/20",
    amber: "bg-amber-500/10 text-amber-300 ring-amber-400/20",
  };
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium ring-1", tones[tone])}>
      {children}
    </span>
  );
}
