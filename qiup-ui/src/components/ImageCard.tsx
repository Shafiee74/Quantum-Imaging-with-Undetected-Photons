"use client";

import { AlertTriangle, Download, Loader2, Maximize2, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { formatStat } from "@/lib/params";
import type { ImagePayload, ThermalRow } from "@/lib/types";
import { Badge, cx } from "./ui";

export type CardState =
  | { status: "idle" }
  | { status: "queued" }
  | { status: "running" }
  | { status: "done"; image: ImagePayload; elapsed?: number; sha?: string | null; thermal?: ThermalRow[] }
  | { status: "error"; message: string };

function Colorbar({ min, max }: { min: number; max: number }) {
  return (
    <div className="mt-2">
      <div className="inferno-bar h-1.5 w-full rounded-full" />
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-400">
        <span>{formatStat(min)}</span>
        <span>{formatStat(max)}</span>
      </div>
    </div>
  );
}

function Lightbox({ src, title, onClose }: { src: string; title: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div
      className="fixed inset-0 z-[100] grid place-items-center bg-black/80 p-8 backdrop-blur-sm"
      onClick={onClose}
    >
      <div className="glass fade-up rounded-2xl p-4" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between gap-6">
          <h4 className="text-sm font-semibold text-white">{title}</h4>
          <button type="button" onClick={onClose} className="rounded-md p-1 text-ink-400 hover:bg-ink-700 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={title} className="pixelated h-[min(75vh,720px)] w-[min(75vh,720px)] rounded-lg" />
      </div>
    </div>
  );
}

export function ImageCard({
  title,
  subtitle,
  state,
  tone,
  fileName,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  state: CardState;
  tone: "ideal" | "noisy" | "object";
  fileName: string;
  footer?: ReactNode;
}) {
  const [zoom, setZoom] = useState(false);
  const src = state.status === "done" ? `data:image/png;base64,${state.image.png}` : null;
  const toneRing = { ideal: "ring-emerald-400/25", noisy: "ring-rose-400/25", object: "ring-sky-400/25" }[tone];
  const toneBadge = { ideal: "green", noisy: "rose", object: "sky" }[tone] as "green" | "rose" | "sky";

  return (
    <div className={cx("glass fade-up flex flex-col rounded-xl p-3 ring-1", toneRing)}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <Badge tone={toneBadge}>{tone === "object" ? "SLM" : tone}</Badge>
            <span className="truncate text-[12px] font-semibold text-white">{title}</span>
          </div>
          {subtitle && <div className="mt-0.5 truncate text-[10px] text-ink-400">{subtitle}</div>}
        </div>
        {src && (
          <div className="flex shrink-0 gap-0.5">
            <button
              type="button"
              title="Enlarge"
              onClick={() => setZoom(true)}
              className="rounded-md p-1 text-ink-400 hover:bg-ink-700 hover:text-white"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
            <a
              title="Download PNG"
              href={src}
              download={fileName}
              className="rounded-md p-1 text-ink-400 hover:bg-ink-700 hover:text-white"
            >
              <Download className="h-3.5 w-3.5" />
            </a>
          </div>
        )}
      </div>

      <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-ink-950 ring-1 ring-white/5">
        {src && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={src}
            alt={title}
            className="pixelated h-full w-full cursor-zoom-in"
            onClick={() => setZoom(true)}
          />
        )}
        {state.status === "running" && (
          <div className="shimmer absolute inset-0 grid place-items-center">
            <div className="flex items-center gap-2 text-[11px] text-sky-300">
              <Loader2 className="h-4 w-4 animate-spin" /> simulating…
            </div>
          </div>
        )}
        {state.status === "queued" && (
          <div className="absolute inset-0 grid place-items-center text-[11px] text-ink-400">queued</div>
        )}
        {state.status === "idle" && (
          <div className="grid-bg absolute inset-0 grid place-items-center text-[11px] text-ink-600">no data</div>
        )}
        {state.status === "error" && (
          <div className="absolute inset-0 grid place-items-center p-3 text-center">
            <div>
              <AlertTriangle className="mx-auto mb-1 h-5 w-5 text-rose-400" />
              <p className="line-clamp-5 text-[10px] text-rose-300">{state.message}</p>
            </div>
          </div>
        )}
      </div>

      {state.status === "done" && (
        <>
          <Colorbar min={state.image.min} max={state.image.max} />
          <div className="mt-2 grid grid-cols-3 gap-1 font-mono text-[10px]">
            <Stat label="mean" value={formatStat(state.image.mean)} />
            <Stat label="std" value={formatStat(state.image.std)} />
            <Stat label="time" value={state.elapsed !== undefined ? `${state.elapsed.toFixed(2)}s` : "—"} />
          </div>
          {state.thermal && state.thermal.length > 0 && (
            <div className="mt-2 space-y-0.5 rounded-md bg-ink-950/60 px-2 py-1.5 font-mono text-[10px] text-ink-300">
              {state.thermal.map((row) => (
                <div key={row.crystal} className="flex justify-between">
                  <span className="text-ink-400">{row.crystal}</span>
                  <span>
                    T<sub>pm</sub> {row.T_pm_C?.toFixed(2) ?? "—"} · T {row.T_crystal_C?.toFixed(2) ?? "—"} °C
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
      {footer}
      {zoom && src && <Lightbox src={src} title={title} onClose={() => setZoom(false)} />}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-ink-950/60 px-1.5 py-1">
      <div className="text-[9px] uppercase tracking-wider text-ink-400">{label}</div>
      <div className="truncate text-ink-200">{value}</div>
    </div>
  );
}
