"use client";

import {
  Activity,
  AlertTriangle,
  Atom,
  CheckCircle2,
  FileJson,
  FolderOpen,
  Loader2,
  Play,
  RefreshCw,
  Square,
  Terminal,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { NOISE_GROUPS, NOISE_KEYS } from "@/lib/params";
import type {
  Defaults,
  Mode,
  ObjectEvent,
  ObjectSpec,
  ParamDict,
  ParamValue,
  ProfileId,
  RunConfig,
  RunEvent,
} from "@/lib/types";
import {
  ArrangementsPanel,
  GeometryPanel,
  NoisePanel,
  NoisePreset,
  ObjectPanel,
  PRESET_SCALE,
  PROFILE_IDS,
  RunSettingsPanel,
  SourcePanel,
  ThermalPanel,
} from "./ControlPanels";
import { ImageCard, type CardState } from "./ImageCard";
import { Badge, cx } from "./ui";

type DefaultsResponse = Defaults & { backend?: { python: string; root: string }; error?: string };
type LogLine = { t: number; kind: "info" | "ok" | "err"; text: string };

const MODES: Mode[] = ["ideal", "noisy"];

function initialThermal(defaults: Defaults): Record<ProfileId, Record<string, number>> {
  return Object.fromEntries(PROFILE_IDS.map((id) => [id, { ...defaults.profiles[id].thermal }])) as Record<
    ProfileId,
    Record<string, number>
  >;
}

function applyPreset(defaults: Defaults, preset: NoisePreset): { setup: ParamDict; latents: ParamDict } {
  const scale = PRESET_SCALE[preset];
  const setup: ParamDict = { ...defaults.noisy_setup };
  const latents: ParamDict = { ...defaults.noisy_latents };
  for (const p of NOISE_KEYS) {
    const source = p.target === "setup" ? defaults.noisy_setup : defaults.noisy_latents;
    const base = typeof source[p.key] === "number" ? (source[p.key] as number) : 0;
    const value = Math.min(p.max, Math.max(p.min, base * scale));
    if (p.target === "setup") setup[p.key] = value;
    else latents[p.key] = value;
  }
  if (preset === "off") {
    for (const g of NOISE_GROUPS) for (const t of g.toggles ?? []) setup[t.key] = false;
  }
  return { setup, latents };
}

export default function Simulator() {
  const [defaults, setDefaults] = useState<DefaultsResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [profiles, setProfiles] = useState<ProfileId[]>([...PROFILE_IDS]);
  const [object, setObject] = useState<ObjectSpec>({ source: "mnist", mnist_index: 5 });
  const [size, setSize] = useState(64);
  const [seed, setSeed] = useState(15000);
  const [modes, setModes] = useState({ ideal: true, noisy: true });
  const [idealSetup, setIdealSetup] = useState<ParamDict>({});
  const [idealLatents, setIdealLatents] = useState<ParamDict>({});
  const [noisySetup, setNoisySetup] = useState<ParamDict>({});
  const [noisyLatents, setNoisyLatents] = useState<ParamDict>({});
  const [thermal, setThermal] = useState<Record<ProfileId, Record<string, number>>>({} as never);
  const [preset, setPreset] = useState<NoisePreset | "custom">("moderate");
  const [sourceMode, setSourceMode] = useState<"ideal" | "noisy">("noisy");

  const [preview, setPreview] = useState<ObjectEvent | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [cards, setCards] = useState<Record<string, CardState>>({});
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const pushLog = useCallback((kind: LogLine["kind"], text: string) => {
    setLog((l) => [...l.slice(-199), { t: Date.now(), kind, text }]);
  }, []);

  const resetFromDefaults = useCallback((d: Defaults) => {
    setIdealSetup({ ...d.ideal_setup });
    setIdealLatents({ ...d.ideal_latents });
    setNoisySetup({ ...d.noisy_setup });
    setNoisyLatents({ ...d.noisy_latents });
    setThermal(initialThermal(d));
    setSize(d.size);
    setSeed(d.seed);
    setObject({ source: "mnist", mnist_index: d.mnist_index });
    setPreset("moderate");
  }, []);

  useEffect(() => {
    fetch("/api/defaults")
      .then((r) => r.json())
      .then((d: DefaultsResponse) => {
        if (d.error) throw new Error(d.error);
        setDefaults(d);
        resetFromDefaults(d);
      })
      .catch((e: Error) => setLoadError(e.message));
  }, [resetFromDefaults]);

  useEffect(() => {
    if (!defaults) return;
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      setPreviewBusy(true);
      fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object, size }),
        signal: ctrl.signal,
      })
        .then((r) => r.json())
        .then((ev) => {
          if (ev.error) throw new Error(ev.error);
          setPreview(ev as ObjectEvent);
        })
        .catch((e: Error) => e.name !== "AbortError" && pushLog("err", `preview: ${e.message}`))
        .finally(() => !ctrl.signal.aborted && setPreviewBusy(false));
    }, 450);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [defaults, object, size, pushLog]);

  const config: RunConfig = useMemo(
    () => ({
      profiles,
      modes,
      object,
      size,
      seed,
      thermal: Object.fromEntries(profiles.map((p) => [p, thermal[p]])),
      ideal_setup: idealSetup,
      ideal_latents: idealLatents,
      noisy_setup: noisySetup,
      noisy_latents: noisyLatents,
    }),
    [profiles, modes, object, size, seed, thermal, idealSetup, idealLatents, noisySetup, noisyLatents],
  );

  const handleEvent = useCallback(
    (ev: RunEvent) => {
      switch (ev.type) {
        case "start":
          setProgress({ done: 0, total: ev.total });
          setOutputDir(ev.output_dir);
          pushLog("info", `run started: ${ev.profiles.join(", ")} × ${ev.modes.join(" + ")}`);
          break;
        case "object":
          setPreview(ev);
          pushLog("info", `SLM object: ${ev.source === "mnist" ? `MNIST #${ev.mnist_index} (label ${ev.label})` : ev.label}`);
          break;
        case "progress":
          setCards((c) => ({ ...c, [`${ev.profile}_${ev.mode}`]: { status: "running" } }));
          break;
        case "result":
          setCards((c) => ({
            ...c,
            [`${ev.profile}_${ev.mode}`]: {
              status: "done",
              image: ev.image,
              elapsed: ev.elapsed_s,
              sha: ev.sha256,
              thermal: ev.thermal,
            },
          }));
          setProgress((p) => ({ ...p, done: p.done + 1 }));
          pushLog("ok", `${ev.profile} ${ev.mode} done in ${ev.elapsed_s.toFixed(2)} s`);
          break;
        case "error":
          if (ev.profile && ev.mode) {
            setCards((c) => ({ ...c, [`${ev.profile}_${ev.mode}`]: { status: "error", message: ev.message } }));
            setProgress((p) => ({ ...p, done: p.done + 1 }));
          }
          pushLog("err", ev.profile ? `${ev.profile} ${ev.mode}: ${ev.message}` : ev.message);
          break;
        case "done":
          setElapsed(ev.elapsed_s);
          setOutputDir(ev.output_dir);
          pushLog("ok", `finished in ${ev.elapsed_s.toFixed(2)} s`);
          break;
      }
    },
    [pushLog],
  );

  const run = useCallback(async () => {
    if (running || profiles.length === 0 || (!modes.ideal && !modes.noisy)) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setElapsed(null);
    setOutputDir(null);
    setCards(
      Object.fromEntries(
        profiles.flatMap((p) => MODES.filter((m) => modes[m]).map((m) => [`${p}_${m}`, { status: "queued" } as CardState])),
      ),
    );
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
        signal: ctrl.signal,
      });
      if (!res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            handleEvent(JSON.parse(line) as RunEvent);
          } catch {
            pushLog("err", `unparsable line: ${line.slice(0, 120)}`);
          }
        }
      }
      if (buffer.trim()) handleEvent(JSON.parse(buffer) as RunEvent);
    } catch (e) {
      const err = e as Error;
      pushLog("err", err.name === "AbortError" ? "run cancelled" : err.message);
      setCards((c) =>
        Object.fromEntries(
          Object.entries(c).map(([k, v]) => [k, v.status === "running" || v.status === "queued" ? { status: "idle" } : v]),
        ),
      );
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [running, profiles, modes, config, handleEvent, pushLog]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void run();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run]);

  const setGeometry = (key: string, value: number) => {
    setIdealSetup((s) => ({ ...s, [key]: value }));
    setNoisySetup((s) => ({ ...s, [key]: value }));
  };
  const setSourceLatent = (key: string, value: number) =>
    (sourceMode === "ideal" ? setIdealLatents : setNoisyLatents)((s) => ({ ...s, [key]: value }));
  const onPreset = (p: NoisePreset) => {
    if (!defaults) return;
    const { setup, latents } = applyPreset(defaults, p);
    const keepSetup = (s: ParamDict) => {
      const next = { ...s };
      for (const pk of NOISE_KEYS) if (pk.target === "setup") next[pk.key] = setup[pk.key];
      for (const g of NOISE_GROUPS) for (const t of g.toggles ?? []) next[t.key] = setup[t.key];
      return next;
    };
    const keepLatents = (s: ParamDict) => {
      const next = { ...s };
      for (const pk of NOISE_KEYS) if (pk.target === "latents") next[pk.key] = latents[pk.key];
      return next;
    };
    setNoisySetup(keepSetup);
    setNoisyLatents(keepLatents);
    setPreset(p);
  };

  const downloadConfig = () => {
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "qiup_run_config.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loadError) {
    return (
      <div className="grid min-h-screen place-items-center p-8">
        <div className="glass max-w-xl rounded-2xl p-6">
          <div className="mb-3 flex items-center gap-2 text-rose-300">
            <AlertTriangle className="h-5 w-5" />
            <h2 className="font-semibold">Python backend unavailable</h2>
          </div>
          <pre className="scroll-thin max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-ink-950 p-3 font-mono text-[11px] text-ink-300">
            {loadError}
          </pre>
          <p className="mt-3 text-[12px] text-ink-400">
            Set <code className="text-sky-300">QIUP_PYTHON</code> to a Python with numpy/scipy/torch installed, then reload.
          </p>
        </div>
      </div>
    );
  }

  if (!defaults) {
    return (
      <div className="grid min-h-screen place-items-center">
        <div className="flex flex-col items-center gap-3 text-ink-300">
          <Atom className="h-10 w-10 animate-spin text-sky-400 [animation-duration:3s]" />
          <p className="text-sm">Starting QIUP simulator backend…</p>
        </div>
      </div>
    );
  }

  const canRun = !running && profiles.length > 0 && (modes.ideal || modes.noisy);
  const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
  const shownModes = MODES.filter((m) => modes[m]);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-white/5 bg-ink-950/75 backdrop-blur-xl">
        <div className="flex items-center gap-4 px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-violet-500 via-sky-500 to-emerald-400 shadow-lg shadow-sky-500/20">
              <Atom className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight text-white">
                QIUP Simulator
                <span className="ml-2 align-middle">
                  <Badge tone="sky">v{defaults.package_version}</Badge>
                </span>
                <span className="ml-2 align-middle">
                  <Badge tone="amber">prototype · pre-validation</Badge>
                </span>
              </h1>
              <p className="text-[11px] text-ink-400">
                Quantum imaging with undetected photons · four-arrangement simulator ·{" "}
                {defaults.calibration_status ?? "NOT YET EXPERIMENTALLY CALIBRATED"}
              </p>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="hidden items-center gap-1.5 rounded-lg bg-ink-900/80 px-2.5 py-1.5 text-[11px] text-ink-400 ring-1 ring-white/5 lg:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px] shadow-emerald-400" />
              backend · {defaults.backend?.python ?? "python"}
            </span>
            <button
              type="button"
              onClick={() => resetFromDefaults(defaults)}
              title="Reset all parameters"
              className="flex items-center gap-1.5 rounded-lg bg-ink-800 px-3 py-2 text-[12px] text-ink-300 ring-1 ring-white/5 hover:bg-ink-700 hover:text-white"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Reset
            </button>
            <button
              type="button"
              onClick={downloadConfig}
              title="Download run config JSON"
              className="flex items-center gap-1.5 rounded-lg bg-ink-800 px-3 py-2 text-[12px] text-ink-300 ring-1 ring-white/5 hover:bg-ink-700 hover:text-white"
            >
              <FileJson className="h-3.5 w-3.5" /> Config
            </button>
            {running ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="flex items-center gap-2 rounded-lg bg-rose-500/90 px-4 py-2 text-[13px] font-semibold text-white shadow-lg shadow-rose-500/20 hover:bg-rose-500"
              >
                <Square className="h-3.5 w-3.5 fill-current" /> Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void run()}
                disabled={!canRun}
                title="Run simulation (Ctrl+Enter)"
                className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-sky-500 to-violet-500 px-4 py-2 text-[13px] font-semibold text-white shadow-lg shadow-sky-500/25 transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Play className="h-3.5 w-3.5 fill-current" /> Run simulation
              </button>
            )}
          </div>
        </div>
        <div className="h-0.5 w-full bg-transparent">
          <div
            className={cx("h-full bg-gradient-to-r from-sky-400 via-violet-400 to-rose-400 transition-all duration-500", !running && pct === 100 && "opacity-0")}
            style={{ width: `${running || pct < 100 ? pct : 100}%` }}
          />
        </div>
      </header>

      <div className="flex flex-1 flex-col lg:flex-row">
        <aside className="scroll-thin w-full shrink-0 space-y-3 overflow-y-auto border-r border-white/5 p-4 lg:sticky lg:top-[61px] lg:h-[calc(100vh-61px)] lg:w-[400px]">
          <ArrangementsPanel defaults={defaults} selected={profiles} onChange={setProfiles} />
          <ObjectPanel object={object} onChange={setObject} maxIndex={defaults.mnist_train_count} />
          <RunSettingsPanel seed={seed} size={size} modes={modes} sizeRange={defaults.size_range}
            onSeed={setSeed} onSize={setSize} onModes={setModes} />
          <SourcePanel mode={sourceMode} onMode={setSourceMode}
            latents={sourceMode === "ideal" ? idealLatents : noisyLatents} onChange={setSourceLatent} />
          <GeometryPanel setup={noisySetup} onChange={setGeometry} />
          <ThermalPanel defaults={defaults} selected={profiles} thermal={thermal}
            onChange={(p, k, v) => setThermal((t) => ({ ...t, [p]: { ...t[p], [k]: v } }))}
            onReset={() => setThermal(initialThermal(defaults))} />
          <NoisePanel setup={noisySetup} latents={noisyLatents} preset={preset} onPreset={onPreset}
            onSetup={(k: string, v: ParamValue) => {
              setNoisySetup((s) => ({ ...s, [k]: v }));
              if (k !== "blur_model") setPreset("custom");
            }}
            onLatent={(k, v) => {
              setNoisyLatents((s) => ({ ...s, [k]: v }));
              setPreset("custom");
            }} />
        </aside>

        <main className="min-w-0 flex-1 space-y-5 p-5">
          <section className="grid gap-4 xl:grid-cols-[280px_1fr]">
            <ImageCard
              title={
                preview
                  ? preview.source === "mnist"
                    ? `MNIST #${preview.mnist_index}`
                    : String(preview.label)
                  : "SLM object"
              }
              subtitle={
                preview?.source === "mnist"
                  ? `digit label ${preview.label} · ${size}×${size} px`
                  : `${size}×${size} px amplitude mask`
              }
              tone="object"
              fileName="slm_object.png"
              state={preview ? { status: "done", image: preview.image } : previewBusy ? { status: "running" } : { status: "idle" }}
            />
            <div className="glass flex flex-col rounded-xl p-4">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-sky-400" />
                <h2 className="text-[13px] font-semibold text-white">Run status</h2>
                <span className="ml-auto">
                  {running ? (
                    <Badge tone="sky"><Loader2 className="h-3 w-3 animate-spin" /> running</Badge>
                  ) : elapsed !== null ? (
                    <Badge tone="green"><CheckCircle2 className="h-3 w-3" /> complete</Badge>
                  ) : (
                    <Badge>idle</Badge>
                  )}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Metric label="Arrangements" value={profiles.length ? profiles.join(" · ") : "—"} />
                <Metric label="Progress" value={progress.total ? `${progress.done} / ${progress.total}` : "—"} />
                <Metric label="Wall time" value={elapsed !== null ? `${elapsed.toFixed(1)} s` : running ? "…" : "—"} />
                <Metric label="Noise preset" value={modes.noisy ? preset : "n/a"} />
              </div>
              {outputDir && (
                <div className="mt-3 flex items-center gap-2 rounded-lg bg-ink-950/60 px-3 py-2 font-mono text-[11px] text-ink-300 ring-1 ring-white/5">
                  <FolderOpen className="h-3.5 w-3.5 shrink-0 text-amber-300" />
                  <span className="truncate" title={outputDir}>{outputDir}</span>
                </div>
              )}
              <div className="mt-3 flex min-h-0 flex-1 flex-col">
                <div className="mb-1 flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-ink-400">
                  <Terminal className="h-3 w-3" /> Event log
                </div>
                <div className="scroll-thin h-28 flex-1 overflow-y-auto rounded-lg bg-ink-950/70 p-2 font-mono text-[11px] ring-1 ring-white/5">
                  {log.length === 0 && <div className="text-ink-600">Press Run simulation or Ctrl+Enter.</div>}
                  {log.map((l, i) => (
                    <div key={i} className={cx(l.kind === "err" ? "text-rose-300" : l.kind === "ok" ? "text-emerald-300" : "text-ink-300")}>
                      <span className="text-ink-600">{new Date(l.t).toLocaleTimeString()} </span>
                      {l.text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {profiles.length === 0 || shownModes.length === 0 ? (
            <div className="glass grid place-items-center rounded-xl p-12 text-sm text-ink-400">
              Select at least one arrangement and one output mode.
            </div>
          ) : (
            shownModes.map((mode) => (
              <section key={mode}>
                <div className="mb-2 flex items-center gap-2">
                  <span className={cx("h-2 w-2 rounded-full", mode === "ideal" ? "bg-emerald-400" : "bg-rose-400")} />
                  <h2 className="text-[13px] font-semibold capitalize text-white">{mode} camera output</h2>
                  <span className="text-[11px] text-ink-400">
                    {mode === "ideal"
                      ? "noise-free signal image — all drift, optical blur and detector noise disabled"
                      : "detector frame with drift latents, optical blur and EMCCD noise"}
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 2xl:grid-cols-4">
                  {profiles.map((p) => {
                    const info = defaults.profiles[p];
                    return (
                      <ImageCard
                        key={`${p}_${mode}`}
                        title={`${p} · ${info.label}`}
                        subtitle={`${info.architecture.replaceAll("_", " ")} · ${info.imaging_mode.replace("_", " ")}`}
                        tone={mode}
                        fileName={`${p}_${mode}.png`}
                        state={cards[`${p}_${mode}`] ?? { status: "idle" }}
                      />
                    );
                  })}
                </div>
              </section>
            ))
          )}
        </main>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-ink-950/60 px-3 py-2 ring-1 ring-white/5">
      <div className="text-[10px] uppercase tracking-wider text-ink-400">{label}</div>
      <div className="mt-0.5 truncate font-mono text-[13px] capitalize text-white">{value}</div>
    </div>
  );
}
