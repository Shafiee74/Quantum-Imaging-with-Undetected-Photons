(() => {
  const PROFILE_IDS = ["A", "B", "C", "D"];
  const PRESET_SCALE = { off: 0, mild: 0.5, moderate: 1, strong: 2 };

  const GEOMETRY = [
    { key: "pump_waist_um", label: "Pump waist w₀", unit: "µm", min: 100, max: 350, step: 5, target: "setup",
      hint: "Synced to NL1/NL2. Image change is clearest with SPDC correlation PSF (double blur) on." },
    { key: "pixel_um", label: "Camera pixel", unit: "µm", min: 6.5, max: 24, step: 0.5, target: "setup",
      hint: "Detector pixel pitch. UI default is 6.5 µm; typical lab cameras often use ~13–16 µm." },
    { key: "object_map_pixel_um", label: "SLM map pixel", unit: "µm", min: 2, max: 24, step: 0.5, target: "setup",
      hint: "Physical size of one object-map pixel on the SLM." },
  ];

  const MAGNIFICATION = {
    near_field: [
      { key: "near_field_idler_magnification", label: "Idler magnification", min: 0.25, max: 4, step: 0.05, target: "setup" },
      { key: "near_field_signal_magnification", label: "Signal magnification", min: 0.25, max: 4, step: 0.05, target: "setup" },
    ],
    far_field: [
      { key: "far_field_idler_relay_magnification", label: "Idler relay mag.", min: 0.25, max: 4, step: 0.05, target: "setup" },
      { key: "far_field_signal_relay_magnification", label: "Signal relay mag.", min: 0.25, max: 4, step: 0.05, target: "setup" },
    ],
  };

  const SOURCE = [
    { key: "mean_photons_per_mode", label: "Mean photons / mode n̄", min: 0.1, max: 20, step: 0.1, target: "latents",
      hint: "SPDC brightness. Higher n̄ raises signal above the detector floor." },
    { key: "exposure_s", label: "Exposure", unit: "s", min: 0.005, max: 1, step: 0.005, target: "latents" },
    { key: "em_gain", label: "EM gain", min: 1, max: 1000, step: 1, target: "latents", integer: true,
      hint: "EMCCD multiplication gain (≥ 1)." },
    { key: "num_frames", label: "Frames averaged", min: 1, max: 64, step: 1, target: "latents", integer: true,
      hint: "Number of frames requested for averaging (UI capped at 64)." },
  ];

  const NOISE_GROUPS = [
    {
      id: "drift",
      title: "Physical drift latents",
      description: "Dimensionless stress levels s ∈ [0, 1] for each subsystem.",
      numeric: [
        { key: "s_pump", label: "Pump drift", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "s_gain", label: "SPDC gain drift", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "s_align", label: "Alignment", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "s_thermal", label: "Thermal", min: 0, max: 0.99, step: 0.01, target: "latents",
          hint: "Thermal drift latent. Values near 1 can be rejected for short acquisitions; UI max is 0.99." },
        {
          key: "s_focus",
          label: "Focus",
          min: 0,
          max: 1,
          step: 0.01,
          target: "latents",
          hint: "Maps to mechanical Δz on A–D (Δz = s_focus · Δz_max).",
        },
        {
          key: "s_slm",
          label: "SLM",
          min: 0,
          max: 1,
          step: 0.01,
          target: "latents",
          hint: "Phase-calibration latent; little effect on amplitude-only (φ≡0) objects.",
        },
        { key: "s_det", label: "Detector", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "s_cool", label: "Cooling", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "delta_phi_rms", label: "Phase jitter δφ", unit: "rad", min: 0, max: 0.5, step: 0.005, target: "latents" },
      ],
    },
    {
      id: "optics",
      title: "Optical blur & misalignment",
      description: "Point-spread and geometric imperfections of the imaging path.",
      numeric: [
        { key: "sigma_blur_extra_px", label: "Extra blur σ", unit: "px", min: 0, max: 3, step: 0.05, target: "setup" },
        { key: "sigma_diffraction_px", label: "Diffraction σ", unit: "px", min: 0, max: 3, step: 0.05, target: "setup" },
        { key: "sigma_aberration_px", label: "Aberration σ", unit: "px", min: 0, max: 3, step: 0.05, target: "setup" },
        {
          key: "delta_phi_align_span_rad",
          label: "Align phase span",
          unit: "rad",
          min: 0,
          max: 1,
          step: 0.01,
          target: "setup",
          hint: "Coupled to wavefront when raised (enables couple_align_to_wavefront).",
        },
        { key: "image_rotation_deg", label: "Image rotation", unit: "°", min: -5, max: 5, step: 0.1, target: "setup" },
        { key: "slm_misreg_x_px", label: "SLM misreg. x", unit: "px", min: -4, max: 4, step: 0.1, target: "setup" },
        { key: "slm_misreg_y_px", label: "SLM misreg. y", unit: "px", min: -4, max: 4, step: 0.1, target: "setup" },
        {
          key: "lens_shift_x_um",
          label: "Lens shift x",
          unit: "µm",
          min: -80,
          max: 80,
          step: 1,
          target: "setup",
          hint: "Needed with Lens rings (lateral idler-lens shift).",
        },
        {
          key: "lens_shift_y_um",
          label: "Lens shift y",
          unit: "µm",
          min: -80,
          max: 80,
          step: 1,
          target: "setup",
        },
        {
          key: "delta_z_um",
          label: "Defocus Δz",
          unit: "µm",
          min: -100,
          max: 100,
          step: 1,
          target: "setup",
          hint: "Mechanical defocus; also activates Lens rings when non-zero.",
        },
      ],
      toggles: [
        {
          key: "enable_lens_rings",
          label: "Lens rings",
          target: "setup",
          desc: "Needs non-zero lens shift or Δz. Turning on seeds Lens shift x = 20 µm if both are zero.",
        },
        { key: "enable_etalon_fringes", label: "Etalon fringes", target: "setup" },
      ],
    },
    {
      id: "detector",
      title: "EMCCD detector",
      description: "Readout, clock-induced charge, dark current and background.",
      numeric: [
        { key: "read_noise0_e", label: "Read noise (base)", unit: "e⁻", min: 0, max: 40, step: 0.5, target: "setup" },
        { key: "read_noise_span_e", label: "Read noise span", unit: "e⁻", min: 0, max: 80, step: 0.5, target: "setup" },
        { key: "prnu_sigma", label: "PRNU σ", min: 0, max: 0.1, step: 0.001, target: "setup" },
        { key: "cic0", label: "Clock-induced charge", unit: "e⁻/px", min: 0, max: 0.05, step: 0.0005, target: "setup" },
        { key: "dark0", label: "Dark current", unit: "e⁻/px/s", min: 0, max: 0.02, step: 0.0001, target: "setup" },
        { key: "hot_frac_at_s1", label: "Hot-pixel fraction", min: 0, max: 0.01, step: 0.0001, target: "setup" },
        {
          key: "accidental_e_at_s1",
          label: "Accidentals",
          unit: "e⁻",
          min: 0,
          max: 0.05,
          step: 0.0005,
          target: "setup",
          hint: "Present in the twin model; currently little/no visible effect on the web EMCCD image path.",
        },
      ],
    },
  ];

  const NOISE_KEYS = NOISE_GROUPS.flatMap((g) => g.numeric);
  const STATIC_BASE = (window.QIUP_STATIC || "/static/ui/").replace(/\/?$/, "/");
  const ARCHITECTURE_SRC = {
    A: `${STATIC_BASE}Architectures/Fig1-A.png`,
    B: `${STATIC_BASE}Architectures/Fig1-B.png`,
    C: `${STATIC_BASE}Architectures/Fig1-C.png`,
    D: `${STATIC_BASE}Architectures/Fig1-D.png`,
  };

  const state = {
    defaults: null,
    profiles: [...PROFILE_IDS],
    object: { source: "mnist", mnist_index: 5 },
    size: 64,
    seed: 15000,
    modes: { ideal: true, noisy: true },
    idealSetup: {},
    idealLatents: {},
    noisySetup: {},
    noisyLatents: {},
    thermal: {},
    preset: "moderate",
    sourceMode: "noisy",
    collapsed: { arrangements: false },
    abort: null,
    running: false,
    previewTimer: null,
    enlarge: null, // { type: 'preview'|'result'|'architecture', ... }
    images: {}, // key -> { png, ... } payload
    diagnostics: {}, // `${profile}_${mode}` -> result diagnostics bundle
    stages: [], // stage-ladder results
    mechanisms: [], // ablation mechanism matrix results
    labPhase: null, // phase-step result diagnostics
    sweepCurve: null, // { param, label, curve: [...] }
    fringeCurve: null, // { label, curve: [{phi_rad, mean_I_plus, ...}] }
    chartCache: {}, // plot id -> last draw payload for hi-res enlarge
    scrollBeforeEnlarge: null, // restore main/page scroll after closing enlarge
    viewport: { w: window.innerWidth || 1280, h: window.innerHeight || 800 },
  };

  const $ = (id) => document.getElementById(id);
  const boot = $("boot");

  function csrfToken() {
    const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  }

  function apiHeaders(extra = {}) {
    const headers = { ...extra };
    const token = csrfToken();
    if (token) headers["X-CSRFToken"] = token;
    return headers;
  }

  async function apiFetch(url, options = {}) {
    const opts = { credentials: "same-origin", ...options };
    opts.headers = apiHeaders(opts.headers || {});
    return fetch(url, opts);
  }
  const errorBox = $("error");
  const workspace = $("workspace");
  const logEl = $("log");
  const sidebar = $("sidebar");

  function updateViewportCss() {
    const w = window.innerWidth || 1280;
    const h = window.innerHeight || 800;
    state.viewport = { w, h };
    const sidebarW = w >= 1100 ? Math.min(400, Math.max(300, Math.round(w * 0.3))) : w;
    document.documentElement.style.setProperty("--sidebar-w", w >= 1100 ? `${sidebarW}px` : "100%");
    const header = document.querySelector(".topbar");
    const headerH = header ? Math.ceil(header.getBoundingClientRect().height) : 70;
    document.documentElement.style.setProperty("--header-h", `${headerH}px`);
    const pill = $("viewport-pill");
    if (pill) pill.textContent = `${w}×${h}`;
  }

  function objectReady(obj) {
    if (!obj) return false;
    if (obj.source === "upload") return Boolean(obj.png_base64);
    return true;
  }

  function syncEnlargeButtons() {
    document.querySelectorAll("[data-enlarge]").forEach((btn) => {
      const kind = btn.getAttribute("data-enlarge");
      let active = false;
      if (state.enlarge) {
        if (kind === "preview") active = state.enlarge.type === "preview";
        else if (kind === "architecture") {
          active = state.enlarge.type === "architecture" && state.enlarge.id === btn.getAttribute("data-arch");
        } else if (kind === "result") {
          active =
            state.enlarge.type === "result" &&
            state.enlarge.profile === btn.getAttribute("data-profile") &&
            state.enlarge.mode === btn.getAttribute("data-mode");
        } else if (kind === "diag") {
          active =
            state.enlarge.type === "diag" &&
            state.enlarge.plot === btn.getAttribute("data-plot") &&
            (btn.getAttribute("data-stage-id") == null ||
              state.enlarge.stageId === btn.getAttribute("data-stage-id")) &&
            (btn.getAttribute("data-profile") == null ||
              state.enlarge.profile === btn.getAttribute("data-profile")) &&
            (btn.getAttribute("data-mode") == null ||
              state.enlarge.mode === btn.getAttribute("data-mode"));
        }
      }
      btn.classList.toggle("active", active);
    });
  }

  function mainScrollEl() {
    return document.querySelector(".main");
  }

  function captureScrollBeforeEnlarge() {
    const main = mainScrollEl();
    state.scrollBeforeEnlarge = {
      main: main ? main.scrollTop : 0,
      window: window.scrollY || document.documentElement.scrollTop || 0,
    };
  }

  function restoreScrollAfterEnlarge() {
    const saved = state.scrollBeforeEnlarge;
    state.scrollBeforeEnlarge = null;
    if (!saved) return;
    const apply = () => {
      const main = mainScrollEl();
      if (main && Number.isFinite(saved.main)) main.scrollTop = saved.main;
      if (Number.isFinite(saved.window)) {
        window.scrollTo(0, saved.window);
      }
    };
    // Restore after layout switches back from enlarge view.
    apply();
    window.requestAnimationFrame(() => {
      apply();
      window.setTimeout(apply, 0);
    });
  }

  function clearEnlarge() {
    state.enlarge = null;
    renderEnlarge();
    syncEnlargeButtons();
    restoreScrollAfterEnlarge();
  }

  function setEnlarge(next) {
    const cur = state.enlarge;
    const same =
      cur &&
      next &&
      cur.type === next.type &&
      cur.profile === next.profile &&
      cur.mode === next.mode &&
      cur.id === next.id &&
      cur.plot === next.plot &&
      cur.stageId === next.stageId;
    if (!cur && next) captureScrollBeforeEnlarge();
    state.enlarge = same ? null : next;
    renderEnlarge();
    syncEnlargeButtons();
    if (same) {
      restoreScrollAfterEnlarge();
    } else if (state.enlarge) {
      const main = mainScrollEl();
      if (main) main.scrollTop = 0;
      window.scrollTo(0, 0);
    }
  }

  function teardownEnlargeChart() {
    const cleanup = state.enlargeChartCleanup;
    state.enlargeChartCleanup = null;
    if (typeof cleanup === "function") {
      try {
        cleanup();
      } catch (_) {
        /* ignore */
      }
    }
  }

  function lineChartGeometry(series, opts, w, h) {
    const pad = { l: 58, r: 14, t: 22, b: 46 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const all = series.flatMap((s) => (s.y || []).filter((v) => Number.isFinite(v)));
    if (!all.length) return null;
    let ymin = opts.ymin != null ? opts.ymin : Math.min(...all);
    let ymax = opts.ymax != null ? opts.ymax : Math.max(...all);
    if (ymax <= ymin) {
      ymin -= 1;
      ymax += 1;
    }
    const xAll = series.flatMap((s) => {
      const y = s.y || [];
      return y.map((_, i) => (s.x && Number.isFinite(s.x[i]) ? Number(s.x[i]) : i));
    });
    let xmin = opts.xmin != null ? opts.xmin : Math.min(...xAll);
    let xmax = opts.xmax != null ? opts.xmax : Math.max(...xAll);
    if (xmax <= xmin) xmax = xmin + 1;
    return { pad, plotW, plotH, xmin, xmax, ymin, ymax };
  }

  function nearestLineHover(series, legend, geom, mx, my) {
    if (!geom) return null;
    const { pad, plotW, plotH, xmin, xmax, ymin, ymax } = geom;
    let best = null;
    series.forEach((s, si) => {
      const y = s.y || [];
      y.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const xv = s.x && Number.isFinite(s.x[i]) ? Number(s.x[i]) : i;
        const px = pad.l + ((xv - xmin) / (xmax - xmin)) * plotW;
        const py = pad.t + (1 - (v - ymin) / (ymax - ymin)) * plotH;
        const d = (px - mx) * (px - mx) + (py - my) * (py - my);
        if (!best || d < best.d) best = { si, i, xv, v, px, py, d, color: s.color || "#38bdf8" };
      });
    });
    if (!best || best.d > 36 * 36) return null;
    // Show every series at the nearest shared x (or closest sample to that x).
    const rows = series.map((s, si) => {
      const y = s.y || [];
      let bi = -1;
      let bd = Infinity;
      let bx = null;
      let by = null;
      y.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const xv = s.x && Number.isFinite(s.x[i]) ? Number(s.x[i]) : i;
        const dx = Math.abs(xv - best.xv);
        if (dx < bd) {
          bd = dx;
          bi = i;
          bx = xv;
          by = v;
        }
      });
      const label = legend?.[si]?.label || `Series ${si + 1}`;
      const color = legend?.[si]?.color || s.color || "#38bdf8";
      return bi < 0
        ? null
        : {
            label,
            color,
            x: bx,
            y: by,
            px: pad.l + ((bx - xmin) / (xmax - xmin)) * plotW,
            py: pad.t + (1 - (by - ymin) / (ymax - ymin)) * plotH,
          };
    }).filter(Boolean);
    return { xv: best.xv, px: best.px, rows };
  }

  function nearestBarHover(groups, legend, geom, mx, my) {
    if (!geom || !groups?.length) return null;
    const { pad, plotW, plotH, ymax, n, gap, cluster } = geom;
    let best = null;
    groups.forEach((g, gi) => {
      const x0 = pad.l + gi * (cluster + gap) + gap / 2;
      const bw = cluster / Math.max(g.values.length, 1);
      g.values.forEach((v, vi) => {
        if (!Number.isFinite(v.value)) return;
        const bh = (v.value / ymax) * plotH;
        const x = x0 + vi * bw;
        const y = pad.t + plotH - bh;
        const cx = x + Math.max(bw - 2, 2) / 2;
        const cy = y + bh / 2;
        const inside =
          mx >= x &&
          mx <= x + Math.max(bw - 2, 2) &&
          my >= y &&
          my <= pad.t + plotH;
        const d = inside ? 0 : (cx - mx) * (cx - mx) + (cy - my) * (cy - my);
        if (!best || d < best.d) {
          best = {
            d,
            gi,
            vi,
            label: g.label || `Group ${gi + 1}`,
            seriesLabel: legend?.[vi]?.label || `Series ${vi + 1}`,
            color: v.color || legend?.[vi]?.color || "#38bdf8",
            value: v.value,
            px: cx,
            py: y,
          };
        }
      });
    });
    if (!best || best.d > 28 * 28) return null;
    return best;
  }

  function nearestHistHover(hists, legend, geom, mx) {
    if (!geom || !hists?.length) return null;
    const { pad, plotW, n, edges } = geom;
    if (n < 1) return null;
    const bw = plotW / n;
    const i = Math.max(0, Math.min(n - 1, Math.floor((mx - pad.l) / bw)));
    if (mx < pad.l || mx > pad.l + plotW) return null;
    const x0 = edges?.[i];
    const x1 = edges?.[i + 1];
    const rows = hists.map((hh, hi) => ({
      label: legend?.[hi]?.label || hh.label || `Series ${hi + 1}`,
      color: legend?.[hi]?.color || hh.color || "#38bdf8",
      count: hh.counts?.[i],
    }));
    return {
      i,
      px: pad.l + i * bw + bw * 0.5,
      x0,
      x1,
      rows,
    };
  }

  function mountEnlargeChart(bodyEl, plotKey, alt) {
    teardownEnlargeChart();
    const cached = state.chartCache[plotKey];
    if (!cached) {
      bodyEl.innerHTML = `<div class="muted">Chart not available yet</div>`;
      return;
    }
    bodyEl.innerHTML = `
      <div class="enlarge-chart-wrap" id="enlarge-chart-wrap">
        <canvas id="enlarge-chart-canvas" aria-label="${escapeHtml(alt)}"></canvas>
        <div id="enlarge-chart-tip" class="enlarge-chart-tip hidden" role="status"></div>
        <div class="enlarge-chart-hint muted">Hover the plot to read live values</div>
      </div>`;
    const wrap = $("enlarge-chart-wrap");
    const canvas = $("enlarge-chart-canvas");
    const tip = $("enlarge-chart-tip");
    if (!wrap || !canvas || !tip) return;

    let layout = { cssW: 900, cssH: 480 };
    let geom = null;

    const paint = (hover) => {
      const cssW = Math.max(640, Math.min(1600, Math.round(wrap.clientWidth || 980)));
      const cssH = Math.max(420, Math.min(900, Math.round((wrap.clientHeight || 640) - 28)));
      layout = { cssW, cssH };
      const baseOpts = { ...cached.opts, cssW, cssH, dpr: 2, cacheKey: null, hover: null };
      try {
        if (cached.kind === "line") {
          geom = lineChartGeometry(cached.series, baseOpts, cssW, cssH);
          drawLineSeries(canvas, cached.series, { ...baseOpts, hover });
        } else if (cached.kind === "bars") {
          const pad = { l: 58, r: 14, t: 28, b: 56 };
          const plotW = cssW - pad.l - pad.r;
          const plotH = cssH - pad.t - pad.b;
          const vals = (cached.groups || []).flatMap((g) =>
            g.values.map((v) => v.value).filter((v) => Number.isFinite(v))
          );
          const ymax = Math.max(...vals, 1e-12);
          const n = (cached.groups || []).length;
          const gap = 12;
          const cluster = Math.max(plotW / Math.max(n, 1) - gap, 10);
          geom = { pad, plotW, plotH, ymax, n, gap, cluster };
          drawBars(canvas, cached.groups, { ...baseOpts, hover });
        } else if (cached.kind === "hist") {
          const pad = { l: 58, r: 14, t: 24, b: 46 };
          const usable = (cached.hists || []).filter((hh) => hh?.counts?.length);
          const n = usable[0]?.counts?.length || 0;
          geom = {
            pad,
            plotW: cssW - pad.l - pad.r,
            plotH: cssH - pad.t - pad.b,
            n,
            edges: usable[0]?.edges,
          };
          drawHistogram(canvas, cached.hists, { ...baseOpts, hover });
        } else {
          bodyEl.innerHTML = `<div class="muted">Chart not available</div>`;
          return;
        }
      } catch (_) {
        bodyEl.innerHTML = `<div class="muted">Unable to enlarge chart</div>`;
      }
    };

    const hideTip = () => {
      tip.classList.add("hidden");
      tip.innerHTML = "";
      paint(null);
    };

    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const mx = ((e.clientX - rect.left) / rect.width) * layout.cssW;
      const my = ((e.clientY - rect.top) / rect.height) * layout.cssH;
      const legend = cached.opts?.legend || [];
      let html = "";
      let hover = null;

      if (cached.kind === "line") {
        const hit = nearestLineHover(cached.series || [], legend, geom, mx, my);
        if (!hit) {
          if (!tip.classList.contains("hidden")) hideTip();
          return;
        }
        hover = hit;
        const xLabel = cached.opts?.xLabel || "x";
        html = `<div class="tip-x">${escapeHtml(xLabel)} = <strong>${escapeHtml(fmtNum(hit.xv, 4))}</strong></div>` +
          hit.rows
            .map(
              (r) =>
                `<div class="tip-row"><span class="swatch" style="background:${escapeHtml(r.color)}"></span>${escapeHtml(r.label)}: <strong>${escapeHtml(fmtNum(r.y, 4))}</strong></div>`
            )
            .join("");
      } else if (cached.kind === "bars") {
        const hit = nearestBarHover(cached.groups || [], legend, geom, mx, my);
        if (!hit) {
          if (!tip.classList.contains("hidden")) hideTip();
          return;
        }
        hover = hit;
        html =
          `<div class="tip-x">${escapeHtml(hit.label)}</div>` +
          `<div class="tip-row"><span class="swatch" style="background:${escapeHtml(hit.color)}"></span>${escapeHtml(hit.seriesLabel)}: <strong>${escapeHtml(fmtNum(hit.value, 4))}</strong></div>`;
      } else if (cached.kind === "hist") {
        const usable = (cached.hists || []).filter((hh) => hh?.counts?.length);
        const hit = nearestHistHover(usable, legend, geom, mx);
        if (!hit) {
          if (!tip.classList.contains("hidden")) hideTip();
          return;
        }
        hover = hit;
        const bin =
          hit.x0 != null && hit.x1 != null
            ? `${fmtNum(hit.x0, 3)} … ${fmtNum(hit.x1, 3)}`
            : `bin ${hit.i + 1}`;
        html =
          `<div class="tip-x">${escapeHtml(cached.opts?.xLabel || "intensity")} · ${escapeHtml(bin)}</div>` +
          hit.rows
            .map(
              (r) =>
                `<div class="tip-row"><span class="swatch" style="background:${escapeHtml(r.color)}"></span>${escapeHtml(r.label)}: <strong>${escapeHtml(fmtNum(r.count, 0))}</strong></div>`
            )
            .join("");
      }

      paint(hover);
      tip.innerHTML = html;
      tip.classList.remove("hidden");
      const tipW = tip.offsetWidth || 160;
      const tipH = tip.offsetHeight || 60;
      let left = e.clientX - rect.left + 14;
      let top = e.clientY - rect.top + 14;
      if (left + tipW > rect.width - 8) left = Math.max(8, e.clientX - rect.left - tipW - 12);
      if (top + tipH > rect.height - 8) top = Math.max(8, e.clientY - rect.top - tipH - 12);
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    };

    const onLeave = () => hideTip();
    const onResize = () => paint(null);
    paint(null);
    // Re-paint after flex layout settles so the chart fills the enlarge panel.
    window.requestAnimationFrame(() => {
      paint(null);
      window.requestAnimationFrame(() => paint(null));
    });
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);
    window.addEventListener("resize", onResize);
    state.enlargeChartCleanup = () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("resize", onResize);
    };
  }

  function enlargeImageHtml(src, alt, opts = {}) {
    if (!src) return `<div class="muted">No image yet</div>`;
    const pixelated = opts.pixelated !== false;
    const imgClass = pixelated ? "pixelated diag-enlarge-img" : "diag-enlarge-img smooth";
    return `
      <div class="enlarge-img-frame">
        <img class="${imgClass}" alt="${escapeHtml(alt || "enlarged")}" src="${src}" />
      </div>`;
  }

  function findStageCard(plot, stageId) {
    const profile = selectedDiagProfile();
    const list =
      plot === "mech-img"
        ? (state.mechanisms || []).filter((s) => s.profile === profile)
        : (state.stages || []).filter((s) => s.profile === profile);
    return list.find((s) => String(s.stage_id) === String(stageId)) || null;
  }

  function stageCardEnlargeHtml(stage) {
    if (!stage) return `<div class="muted">Image not available</div>`;
    const d = stage.diagnostics || {};
    const snr = fmtNum(d.snr, 2);
    const v = fmtNum(d.visibility?.V_50_50, 3);
    const rms = d.residual_rms != null ? fmtNum(d.residual_rms, 3) : "—";
    const png = stage.image?.png;
    if (!png) return `<div class="muted">No image for this stage</div>`;
    return `
      <div class="stage-enlarge-single">
        ${enlargeImageHtml(`data:image/png;base64,${png}`, stage.stage_label || stage.stage_id)}
        <div class="meta" style="margin-top:0.75rem;text-align:center">
          SNR ${escapeHtml(snr)} · V ${escapeHtml(v)} · RMSΔ ${escapeHtml(rms)}
          ${stage.stage_explain ? `<div class="muted" style="margin-top:0.35rem">${escapeHtml(stage.stage_explain)}</div>` : ""}
        </div>
      </div>`;
  }

  function renderEnlarge() {
    const view = $("enlarge-view");
    const normal = $("sim-normal");
    const body = $("enlarge-body");
    const title = $("enlarge-title");
    const sub = $("enlarge-sub");
    const simPanel = $("sim-panel");
    if (!view || !normal || !body) return;
    if (!state.enlarge) {
      teardownEnlargeChart();
      view.classList.add("hidden");
      normal.classList.remove("hidden");
      simPanel?.classList.remove("is-enlarged");
      workspace?.classList.remove("has-enlarge");
      body.classList.remove("enlarge-body-light");
      view.classList.remove("is-architecture");
      body.innerHTML = "";
      return;
    }
    teardownEnlargeChart();
    view.classList.remove("hidden");
    normal.classList.add("hidden");
    simPanel?.classList.add("is-enlarged");
    workspace?.classList.add("has-enlarge");
    const en = state.enlarge;
    const isArchitecture = en.type === "architecture";
    body.classList.toggle("enlarge-body-light", isArchitecture);
    view.classList.toggle("is-architecture", isArchitecture);
    if (isArchitecture) {
      title.textContent = `Arrangement ${en.id} · full architecture`;
      sub.textContent = `${state.defaults?.profiles?.[en.id]?.label || ""} · shown in simulation panel`;
      body.innerHTML = enlargeImageHtml(
        ARCHITECTURE_SRC[en.id],
        `Architecture ${en.id}`,
        { pixelated: false }
      );
      return;
    }
    if (en.type === "preview") {
      const img = state.images.preview;
      title.textContent = $("preview-title")?.textContent || "SLM object";
      sub.textContent = $("preview-sub")?.textContent || "";
      body.innerHTML = img?.png
        ? enlargeImageHtml(pngSrc(img), "preview")
        : `<div class="muted">No preview image</div>`;
      return;
    }
    if (en.type === "result") {
      const key = `${en.profile}_${en.mode}`;
      const img = state.images[key];
      const info = state.defaults?.profiles?.[en.profile];
      title.textContent = `${en.profile} · ${info?.label || ""}`;
      sub.textContent = `${en.mode} · ${(info?.imaging_mode || "").replaceAll("_", " ")}`;
      body.innerHTML = img?.png
        ? enlargeImageHtml(pngSrc(img), key)
        : `<div class="muted">No result image yet</div>`;
      return;
    }
    if (en.type === "diag") {
      const profile = selectedDiagProfile();
      const caption = typeof archCaption === "function" ? archCaption(profile) : profile;
      if (en.plot === "visibility") {
        title.textContent = "Visibility & thermal";
        sub.textContent = caption;
        const vis = $("diag-visibility")?.innerHTML || "";
        const th = $("diag-thermal")?.innerHTML || "";
        body.innerHTML = `<div class="diag-table" style="font-size:0.95rem;max-width:40rem;margin:0 auto">${vis}${th}</div>`;
        return;
      }
      if (en.plot === "transfer") {
        title.textContent = "Effective object transfer";
        sub.textContent = caption;
        body.innerHTML = `<div class="diag-table" style="font-size:0.95rem;max-width:40rem;margin:0 auto">${$("diag-transfer")?.innerHTML || ""}</div>`;
        return;
      }
      if (en.plot === "arch-thumb") {
        const key = `${en.profile}_${en.mode}`;
        const img = state.images[key];
        const diag = state.diagnostics[key]?.diagnostics;
        title.textContent = `${archCaption(en.profile)} · ${en.mode === "noisy" ? "Noisy" : "Ideal"}`;
        sub.textContent = `SNR ${fmtNum(diag?.snr, 2)} · V ${fmtNum(diag?.visibility?.V_50_50, 3)}`;
        body.innerHTML = enlargeImageHtml(pngSrc(img), key);
        return;
      }
      if (en.plot === "residual" || en.plot === "diffport" || en.plot === "teff" || en.plot === "phase-map" || en.plot === "phase-mod" || en.plot === "iplus" || en.plot === "iminus") {
        const titles = {
          residual: "Ideal − noisy residual",
          diffport: "Difference port D = I₊ − I₋",
          teff: "T_eff map",
          "phase-map": "Phase map φ(x, y)",
          "phase-mod": "Phase modulation",
          iplus: "Plus port I₊",
          iminus: "Minus port I₋",
        };
        const payload = diagMapPayload(en.plot);
        title.textContent = titles[en.plot] || "Map";
        sub.textContent = payload
          ? `${caption} · min=${fmtNum(payload.min)} · max=${fmtNum(payload.max)} · mean=${fmtNum(payload.mean)}${payload.constant ? " · constant field" : ""}`
          : caption;
        body.innerHTML = payload?.png
          ? enlargeImageHtml(`data:image/png;base64,${payload.png}`, titles[en.plot])
          : `<div class="muted">${en.plot === "residual" ? "Run both Ideal and Noisy for this arrangement." : "No image yet — run a simulation first."}</div>`;
        return;
      }
      if (en.plot === "stage-img" || en.plot === "mech-img") {
        const stage = findStageCard(en.plot, en.stageId);
        title.textContent = stage?.stage_label || stage?.stage_id || "Stage image";
        sub.textContent = `${caption} · ${en.plot === "mech-img" ? "mechanism" : "noise stage"}`;
        body.innerHTML = stageCardEnlargeHtml(stage);
        return;
      }
      if (en.plot === "sweep") {
        title.textContent = "Parameter sweep";
        sub.textContent = state.sweepCurve?.label || caption;
        mountEnlargeChart(body, "sweep", "Sweep curve");
        return;
      }
      if (en.plot === "fringe") {
        title.textContent = "I(φ) fringe curve";
        sub.textContent = state.fringeCurve?.label || caption;
        mountEnlargeChart(body, "fringe", "I(phi) fringe");
        return;
      }
      if (en.plot === "arch") {
        title.textContent = "A–D comparison";
        sub.textContent = "Ideal vs noisy · NF/FF · T/R";
        mountEnlargeChart(body, "arch", "A-D comparison");
        return;
      }
      if (en.plot === "profiles") {
        title.textContent = "Line profiles";
        sub.textContent = `${caption} · SLM / ideal / noisy mid-row`;
        mountEnlargeChart(body, "profiles", "Line profiles");
        return;
      }
      if (en.plot === "hist") {
        title.textContent = "Intensity histograms";
        sub.textContent = `${caption} · ideal vs noisy`;
        mountEnlargeChart(body, "hist", "Histograms");
        return;
      }
      if (en.plot === "stage-kpi") {
        title.textContent = "Stage KPI bars";
        sub.textContent = `${caption} · SNR / V / RMS residual`;
        mountEnlargeChart(body, "stage-kpi", "Stage KPI chart");
        return;
      }
      title.textContent = "Diagnostics";
      sub.textContent = "";
      body.textContent = "Unknown diagnostic plot";
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function num(v, fallback = 0) {
    return typeof v === "number" && Number.isFinite(v) ? v : fallback;
  }

  function formatValue(value, step) {
    if (!Number.isFinite(value)) return "—";
    const decimals = step >= 1 ? 0 : Math.min(4, Math.max(0, Math.ceil(-Math.log10(step))));
    return value.toFixed(decimals);
  }

  function log(kind, text) {
    const stamp = new Date().toLocaleTimeString();
    const color = kind === "err" ? "#fda4af" : kind === "ok" ? "#6ee7b7" : "#cbd5e1";
    if (logEl.textContent.startsWith("Press Run")) logEl.textContent = "";
    logEl.innerHTML += `<div style="color:${color}"><span style="color:#64748b">${stamp} </span>${escapeHtml(text)}</div>`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function pngSrc(image) {
    return image?.png ? `data:image/png;base64,${image.png}` : "";
  }

  function setImage(box, image, fallback, alt = "simulation result") {
    if (!image?.png) {
      box.textContent = fallback || "—";
      return;
    }
    box.innerHTML = `<img alt="${escapeHtml(alt)}" src="${pngSrc(image)}" />`;
  }

  function clone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function initialThermal(defaults) {
    return Object.fromEntries(PROFILE_IDS.map((id) => [id, { ...defaults.profiles[id].thermal }]));
  }

  function applyPreset(defaults, preset) {
    const scale = PRESET_SCALE[preset];
    const setup = { ...defaults.noisy_setup };
    const latents = { ...defaults.noisy_latents };
    for (const p of NOISE_KEYS) {
      const source = p.target === "setup" ? defaults.noisy_setup : defaults.noisy_latents;
      const base = typeof source[p.key] === "number" ? source[p.key] : 0;
      const value = Math.min(p.max, Math.max(p.min, base * scale));
      if (p.target === "setup") setup[p.key] = value;
      else latents[p.key] = value;
    }
    if (preset === "off") {
      for (const g of NOISE_GROUPS) for (const t of g.toggles || []) setup[t.key] = false;
    }
    return { setup, latents };
  }

  function resetFromDefaults(d) {
    state.idealSetup = { ...d.ideal_setup };
    state.idealLatents = { ...d.ideal_latents };
    state.noisySetup = { ...d.noisy_setup };
    state.noisyLatents = { ...d.noisy_latents };
    state.thermal = initialThermal(d);
    state.size = d.size ?? 64;
    state.seed = d.seed ?? 15000;
    state.object = { source: "mnist", mnist_index: d.mnist_index ?? 5 };
    state.profiles = [...PROFILE_IDS];
    state.modes = { ideal: true, noisy: true };
    state.sourceMode = "noisy";
    state.preset = "moderate";
    state.enlarge = null;
    state.images = {};
    const applied = applyPreset(d, "moderate");
    for (const p of NOISE_KEYS) {
      if (p.target === "setup") state.noisySetup[p.key] = applied.setup[p.key];
      else state.noisyLatents[p.key] = applied.latents[p.key];
    }
    for (const g of NOISE_GROUPS) {
      for (const t of g.toggles || []) {
        if (applied.setup[t.key] !== undefined) state.noisySetup[t.key] = applied.setup[t.key];
      }
    }
  }

  function buildConfig() {
    return {
      profiles: [...state.profiles],
      modes: { ...state.modes },
      object: clone(state.object),
      size: state.size,
      seed: state.seed,
      thermal: Object.fromEntries(state.profiles.map((p) => [p, { ...(state.thermal[p] || {}) }])),
      ideal_setup: { ...state.idealSetup },
      ideal_latents: { ...state.idealLatents },
      noisy_setup: { ...state.noisySetup },
      noisy_latents: { ...state.noisyLatents },
    };
  }

  function panel(id, title, subtitle, bodyHtml, extras = {}) {
    const fixed = Boolean(extras.fixed);
    const open = fixed || !state.collapsed[id];
    const right = extras.right || "";
    const headAttrs = fixed
      ? ""
      : `data-toggle="${id}" role="button" tabindex="0" aria-expanded="${open ? "true" : "false"}" aria-controls="panel-body-${id}"`;
    return `
      <div class="card" data-panel="${id}">
        <div class="panel-head" ${headAttrs}>
          <div>
            <h2 id="panel-title-${id}">${escapeHtml(title)}</h2>
            <p class="panel-sub">${escapeHtml(subtitle)}</p>
          </div>
          <div style="display:flex;align-items:center;gap:0.4rem">
            ${right}
            ${fixed ? "" : `<span class="chevron ${open ? "open" : ""}" aria-hidden="true">▸</span>`}
          </div>
        </div>
        <div class="panel-body ${open ? "" : "collapsed"}" data-body="${id}" id="panel-body-${id}" role="region" aria-labelledby="panel-title-${id}">
          ${bodyHtml}
        </div>
      </div>`;
  }

  function sliderHtml(p, value, accentClass = "") {
    const v = num(value, p.min);
    const unit = p.unit ? ` ${p.unit}` : "";
    const labelId = `slider-label-${p.key}`;
    return `
      <div class="slider-row ${accentClass}" data-slider="${p.key}" data-target="${p.target || "setup"}">
        <div class="slider-top">
          <strong id="${labelId}">${escapeHtml(p.label)}</strong>
          <span class="val" data-val>${formatValue(v, p.step)}${escapeHtml(unit)}</span>
        </div>
        <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}"
          aria-labelledby="${labelId}"
          aria-label="${escapeHtml(p.label)}"
          data-key="${p.key}" data-step="${p.step}" data-unit="${escapeHtml(p.unit || "")}"
          data-integer="${p.integer ? "1" : "0"}" data-target="${p.target || "setup"}" />
        ${p.hint ? `<div class="hint">${escapeHtml(p.hint)}</div>` : ""}
      </div>`;
  }

  function numberFieldHtml(opts) {
    return `
      <label class="field">
        ${escapeHtml(opts.label)}${opts.unit ? ` (${escapeHtml(opts.unit)})` : ""}
        <input type="number" data-num="${opts.key}" value="${opts.value}"
          ${opts.min != null ? `min="${opts.min}"` : ""}
          ${opts.max != null ? `max="${opts.max}"` : ""}
          step="${opts.step ?? 1}" />
        ${opts.hint ? `<span class="hint">${escapeHtml(opts.hint)}</span>` : ""}
      </label>`;
  }

  function toggleHtml(key, label, checked, desc = "", colorClass = "") {
    return `
      <label class="toggle ${colorClass}">
        <span>
          ${escapeHtml(label)}
          ${desc ? `<div class="desc">${escapeHtml(desc)}</div>` : ""}
        </span>
        <input type="checkbox" data-toggle-key="${key}" ${checked ? "checked" : ""} />
      </label>`;
  }

  function renderSidebar() {
    const d = state.defaults;
    if (!d || !sidebar) return;

    // Keep the user at the same place after toggles rebuild the sidebar DOM.
    const prevScroll = sidebar.querySelector(".sidebar-scroll");
    const savedSidebarScroll = prevScroll ? prevScroll.scrollTop : 0;
    const mainEl = mainScrollEl();
    const savedMainScroll = mainEl ? mainEl.scrollTop : 0;
    const savedWindowScroll = window.scrollY || document.documentElement.scrollTop || 0;

    const arrangements = `
      <div style="display:flex;justify-content:flex-end;margin-bottom:0.35rem">
        <button type="button" class="btn tiny ghost" data-action="toggle-all-profiles">
          ${state.profiles.length === 4 ? "Clear" : "All"}
        </button>
      </div>
      <div class="arr-grid">
        ${PROFILE_IDS.map((id) => {
          const info = d.profiles[id];
          const selected = state.profiles.includes(id);
          const archOn = state.enlarge?.type === "architecture" && state.enlarge.id === id;
          const crystals = (info.crystals || []).join("/");
          const two = info.architecture === "two_crystal";
          const far = info.imaging_mode === "far_field";
          const coeff = info.object_coefficient || (two ? "T" : "R");
          const archName = two ? "Two-crystal" : "Single-crystal";
          const fieldName = far ? "Far field" : "Near field";
          const objKind = coeff === "T" ? "Transmissive T" : "Reflective R";
          const plane = (info.object_plane || "").replaceAll("_", " ");
          const title = [
            `Arrangement ${id}`,
            `${archName} · ${crystals}`,
            `${fieldName} · ${objKind}`,
            plane ? `Object plane: ${plane}` : "",
          ].filter(Boolean).join("\n");
          return `
            <div class="arr-card ${selected ? "selected" : ""}" data-profile="${id}" role="button" tabindex="0"
              aria-pressed="${selected ? "true" : "false"}" title="${escapeHtml(title)}">
              <div class="arr-card-top">
                <strong class="arr-id">${id}</strong>
                <span class="arr-nl">${escapeHtml(crystals || "NL")}</span>
              </div>
              <div class="arr-line">${escapeHtml(archName)}</div>
              <div class="arr-meta">
                <span class="arr-chip ${far ? "violet" : "sky"}">${escapeHtml(fieldName)}</span>
                <span class="arr-chip ${coeff === "T" ? "green" : "amber"}">${escapeHtml(objKind)}</span>
              </div>
              <div class="arr-card-foot">
                <span class="arr-plane">${escapeHtml(plane)}</span>
                <button type="button" class="arr-enlarge ${archOn ? "active" : ""}"
                  data-enlarge="architecture" data-arch="${id}"
                  title="Show full architecture in the simulation panel (Esc to close)">⛶</button>
              </div>
            </div>`;
        }).join("")}
      </div>`;

    const src = state.object.source;
    let objectDetails = "";
    if (src === "mnist") {
      objectDetails = `
        <div class="field-row">
          ${numberFieldHtml({
            key: "mnist_index",
            label: "MNIST index",
            value: state.object.mnist_index ?? 5,
            min: 0,
            max: (d.mnist_train_count || 100) - 1,
            step: 1,
            hint: "Index into the shipped MNIST subset. The digit label is not the index.",
          })}
          <div class="icon-row" style="padding-bottom:0.15rem">
            <button type="button" class="icon-btn" data-action="mnist-prev" title="Previous">‹</button>
            <button type="button" class="icon-btn" data-action="mnist-next" title="Next">›</button>
            <button type="button" class="icon-btn" data-action="mnist-rand" title="Random">⚄</button>
          </div>
        </div>`;
    } else if (src === "mask") {
      objectDetails = `
        <div class="mask-grid">
          ${Array.from({ length: 10 }, (_, i) => i + 1).map((id) => `
            <button type="button" data-mask-id="${id}" class="${state.object.mask_id === id ? "active" : ""}">${id}</button>
          `).join("")}
        </div>`;
    } else if (src === "rabbit") {
      objectDetails = `
        <div class="subcard" style="margin-top:0">
          <div class="muted">Built-in rabbit amplitude mask from <code>test-masks/rabbit-slm-mask.png</code>. Resized to the simulation grid with aspect ratio preserved and zero padding.</div>
        </div>`;
    } else {
      objectDetails = `
        <div class="upload-box">
          <input type="file" id="mask-upload-input" accept="image/png,image/jpeg,image/webp,image/bmp,.png,.jpg,.jpeg,.webp,.bmp" hidden />
          <button type="button" class="btn" data-action="pick-upload">
            ${state.object.png_base64 ? "Replace mask image" : "Upload SLM mask"}
          </button>
          ${state.object.file_name ? `<div class="file-name">${escapeHtml(state.object.file_name)}</div>` : ""}
          ${!state.object.png_base64 ? `<div class="muted">Choose a grayscale / binary amplitude mask (not an RGB photo).</div>` : ""}
          <div class="upload-error" id="upload-error"></div>
        </div>`;
    }

    const objectBody = `
      <div class="seg seg-4">
        <button type="button" data-object-source="mnist" class="${src === "mnist" ? "active" : ""}">MNIST</button>
        <button type="button" data-object-source="mask" class="${src === "mask" ? "active" : ""}">Catalog</button>
        <button type="button" data-object-source="rabbit" class="${src === "rabbit" ? "active" : ""}">Rabbit</button>
        <button type="button" data-object-source="upload" class="${src === "upload" ? "active" : ""}">Upload</button>
      </div>
      ${objectDetails}`;

    const sizeRange = d.size_range || [32, 128];
    const runBody = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem">
        ${numberFieldHtml({
          key: "size", label: "Grid size", unit: "px", value: state.size,
          min: sizeRange[0], max: sizeRange[1], step: 8,
          hint: `Square simulation grid, ${sizeRange[0]}–${sizeRange[1]} px.`,
        })}
        ${numberFieldHtml({
          key: "seed", label: "Seed", value: state.seed, min: 0, step: 1,
          hint: "Random seed for detector and drift noise; same seed → identical noisy image.",
        })}
      </div>
      <div class="subcard">
        ${toggleHtml("mode_ideal", "Ideal", state.modes.ideal, "All optical, drift and detector noise off")}
        ${toggleHtml("mode_noisy", "Noisy", state.modes.noisy, "Drift latents, optical blur and EMCCD noise on")}
      </div>`;

    const geometryBody = `
      ${GEOMETRY.map((p) => sliderHtml(p, state.idealSetup[p.key] ?? state.noisySetup[p.key], "")).join("")}
      <div class="subcard">
        <span class="badge sky">Near field · B, D</span>
        ${MAGNIFICATION.near_field.map((p) => sliderHtml(p, state.idealSetup[p.key] ?? state.noisySetup[p.key])).join("")}
      </div>
      <div class="subcard">
        <span class="badge violet">Far field · A</span>
        <div class="hint" style="margin:0.35rem 0 0.55rem">Relay magnification is applied for arrangement A. Arrangement C is far-field but does not use these relay controls in the current UI mapping.</div>
        ${MAGNIFICATION.far_field.map((p) => sliderHtml(p, state.idealSetup[p.key] ?? state.noisySetup[p.key])).join("")}
      </div>`;

    const sourceLatents = state.sourceMode === "ideal" ? state.idealLatents : state.noisyLatents;
    const sourceAccent = state.sourceMode === "ideal" ? "green" : "rose";
    const sourceBody = `
      <div class="seg">
        <button type="button" data-source-mode="ideal" class="${state.sourceMode === "ideal" ? "active" : ""}" aria-pressed="${state.sourceMode === "ideal"}">Ideal run</button>
        <button type="button" data-source-mode="noisy" class="${state.sourceMode === "noisy" ? "active" : ""}" aria-pressed="${state.sourceMode === "noisy"}">Noisy run</button>
      </div>
      <div class="hint" style="margin-top:-0.25rem;margin-bottom:0.55rem">
        Editing the <b>${state.sourceMode}</b> tab only. Ideal and noisy source parameters are stored separately.
      </div>
      ${SOURCE.map((p) => sliderHtml(p, sourceLatents[p.key], sourceAccent)).join("")}`;

    const thermalBody = state.profiles.length === 0
      ? `<p class="muted">Select an arrangement to edit its crystals.</p>`
      : state.profiles.map((id) => {
          const profile = d.profiles[id];
          const crystals = profile.crystals || [];
          return `
            <div class="subcard">
              <div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.35rem">
                <span class="badge violet">${id}</span>
                <span class="muted">${escapeHtml(profile.label)}</span>
              </div>
              ${crystals.map((crystal) => `
                <div class="thermal-grid">
                  <span class="crystal-tag">${escapeHtml(crystal)}</span>
                  <label class="field">T_pm0 (°C)
                    <input type="number" step="0.1" min="20" max="90" data-thermal="${id}" data-tkey="T_pm0_${crystal}_C"
                      title="Phase-matching temperature reference (server accepts crystal T in 20–90 °C)"
                      value="${num(state.thermal[id]?.[`T_pm0_${crystal}_C`])}" />
                  </label>
                  <label class="field">ΔT (°C)
                    <input type="number" step="0.05" min="-30" max="30" data-thermal="${id}" data-tkey="dT_${crystal}_C"
                      title="Offset from T_pm0; keep T_pm0+ΔT inside 20–90 °C"
                      value="${num(state.thermal[id]?.[`dT_${crystal}_C`])}" />
                  </label>
                </div>`).join("")}
            </div>`;
        }).join("");

    const noiseBody = `
      <div class="preset-row">
        ${["off", "mild", "moderate", "strong"].map((p) => `
          <button type="button" data-preset="${p}" class="${state.preset === p ? "active" : ""}">${p}</button>
        `).join("")}
      </div>
      ${state.preset === "custom" ? `<p class="muted" style="color:#fcd34d;margin:0 0 0.5rem">Custom values</p>` : ""}
      ${NOISE_GROUPS.map((group) => `
        <div class="subcard">
          <div style="margin-bottom:0.35rem">
            <div style="font-size:0.78rem;font-weight:600">${escapeHtml(group.title)}</div>
            <div class="muted">${escapeHtml(group.description)}</div>
          </div>
          ${group.numeric.map((p) => {
            const source = p.target === "setup" ? state.noisySetup : state.noisyLatents;
            return sliderHtml(p, source[p.key], "rose");
          }).join("")}
          ${(group.toggles || []).map((t) =>
            toggleHtml(t.key, t.label, Boolean(state.noisySetup[t.key]), t.desc || "")
          ).join("")}
        </div>
      `).join("")}
      ${toggleHtml(
        "blur_model",
        "SPDC correlation PSF (double blur)",
        state.noisySetup.blur_model === "double_blur",
        "Physical two-photon blur; washes out fine detail",
      )}`;

    const arrangementsPanel = panel(
      "arrangements",
      "Arrangements",
      "Architecture × imaging mode",
      arrangements,
      { fixed: true },
    );
    const scrollPanels = [
      panel("object", "SLM object", "Amplitude pattern written on the SLM", objectBody),
      panel("run", "Run settings", "Grid, RNG seed and output modes", runBody),
      panel("geometry", "Geometry", "Applied to both ideal and noisy runs", geometryBody),
      panel("source", "Source & exposure", "SPDC brightness and camera integration", sourceBody),
      panel(
        "thermal",
        "Crystal temperature",
        "T_crystal = T_pm0 + ΔT per crystal",
        thermalBody,
        {
          right: `<button type="button" class="btn tiny ghost" data-action="reset-thermal" title="Reset to profile defaults">↺</button>`,
        },
      ),
      panel("noise", "Noise model", "Noisy run only — ideal keeps everything at zero", noiseBody),
    ].join("");

    sidebar.innerHTML = `
      <div class="sidebar-pinned">${arrangementsPanel}</div>
      <div class="sidebar-scroll">${scrollPanels}</div>`;

    updatePresetMetric();
    bindSidebar();
    updateViewportCss();

    const restoreScroll = () => {
      const nextScroll = sidebar.querySelector(".sidebar-scroll");
      if (nextScroll && Number.isFinite(savedSidebarScroll)) {
        nextScroll.scrollTop = savedSidebarScroll;
      }
      if (mainEl && Number.isFinite(savedMainScroll)) {
        mainEl.scrollTop = savedMainScroll;
      }
      if (Number.isFinite(savedWindowScroll)) {
        window.scrollTo(0, savedWindowScroll);
      }
    };
    restoreScroll();
    window.requestAnimationFrame(() => {
      restoreScroll();
      window.setTimeout(restoreScroll, 0);
    });
  }

  function updatePresetMetric() {
    const el = $("m-preset");
    if (!el) return;
    if (!state.modes.noisy) {
      el.textContent = "n/a";
      return;
    }
    el.textContent = state.preset === "custom" ? "custom" : state.preset;
  }

  function parseSlider(input) {
    let v = Number(input.value);
    if (input.dataset.integer === "1") v = Math.round(v);
    return v;
  }

  function updateSliderLabel(input) {
    const row = input.closest("[data-slider]");
    if (!row) return;
    const valEl = row.querySelector("[data-val]");
    const step = Number(input.dataset.step) || 1;
    const unit = input.dataset.unit ? ` ${input.dataset.unit}` : "";
    if (valEl) valEl.textContent = `${formatValue(parseSlider(input), step)}${unit}`;
  }

  function setGeometry(key, value) {
    const patch = { [key]: value };
    // Twin A–D uses per-crystal pump waists; keep them aligned with the UI control.
    if (key === "pump_waist_um") {
      patch.pump_waist_NL1_um = value;
      patch.pump_waist_NL2_um = value;
    }
    state.idealSetup = { ...state.idealSetup, ...patch };
    state.noisySetup = { ...state.noisySetup, ...patch };
  }

  function setSourceLatent(key, value) {
    if (state.sourceMode === "ideal") {
      state.idealLatents = { ...state.idealLatents, [key]: value };
    } else {
      state.noisyLatents = { ...state.noisyLatents, [key]: value };
    }
  }

  function markCustomPreset() {
    if (state.preset !== "custom") {
      state.preset = "custom";
      updatePresetMetric();
      sidebar.querySelectorAll("[data-preset]").forEach((btn) => {
        btn.classList.toggle("active", false);
      });
    }
  }

  function onNoisePreset(preset) {
    const d = state.defaults;
    if (!d) return;
    const { setup, latents } = applyPreset(d, preset);
    for (const p of NOISE_KEYS) {
      if (p.target === "setup") state.noisySetup[p.key] = setup[p.key];
      else state.noisyLatents[p.key] = latents[p.key];
    }
    for (const g of NOISE_GROUPS) {
      for (const t of g.toggles || []) {
        if (setup[t.key] !== undefined) state.noisySetup[t.key] = setup[t.key];
      }
    }
    state.preset = preset;
    renderSidebar();
    updatePresetMetric();
  }

  function schedulePreview() {
    if (state.previewTimer) clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(() => {
      state.previewTimer = null;
      void refreshPreview();
    }, 450);
  }

  function bindSidebar() {
    const togglePanel = (head) => {
      const id = head.getAttribute("data-toggle");
      if (!id) return;
      state.collapsed[id] = !state.collapsed[id];
      const open = !state.collapsed[id];
      const body = sidebar.querySelector(`[data-body="${id}"]`);
      const chev = head.querySelector(".chevron");
      body?.classList.toggle("collapsed", !open);
      chev?.classList.toggle("open", open);
      head.setAttribute("aria-expanded", open ? "true" : "false");
    };
    sidebar.querySelectorAll("[data-toggle]").forEach((head) => {
      head.addEventListener("click", (e) => {
        if (e.target.closest("[data-action]")) return;
        togglePanel(head);
      });
      head.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (e.target.closest("[data-action]")) return;
          togglePanel(head);
        }
      });
    });

    sidebar.querySelectorAll("[data-profile]").forEach((card) => {
      const toggleProfile = () => {
        const id = card.getAttribute("data-profile");
        if (state.profiles.includes(id)) {
          state.profiles = state.profiles.filter((p) => p !== id);
        } else {
          state.profiles = PROFILE_IDS.filter((p) => p === id || state.profiles.includes(p));
        }
        renderSidebar();
      };
      card.addEventListener("click", (e) => {
        if (e.target.closest("[data-enlarge]")) return;
        e.preventDefault();
        toggleProfile();
      });
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleProfile();
        }
      });
    });

    sidebar.querySelectorAll('[data-enlarge="architecture"]').forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = btn.getAttribute("data-arch");
        setEnlarge({ type: "architecture", id });
        // Keep arrangements enlarge button active state in sync.
        sidebar.querySelectorAll('[data-enlarge="architecture"]').forEach((b) => {
          b.classList.toggle("active", state.enlarge?.type === "architecture" && state.enlarge.id === b.getAttribute("data-arch"));
        });
      });
    });

    sidebar.querySelector("[data-action='toggle-all-profiles']")?.addEventListener("click", (e) => {
      e.stopPropagation();
      state.profiles = state.profiles.length === 4 ? [] : [...PROFILE_IDS];
      renderSidebar();
    });

    sidebar.querySelector("[data-action='reset-thermal']")?.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!state.defaults) return;
      state.thermal = initialThermal(state.defaults);
      renderSidebar();
      log("info", "thermal reset to profile defaults");
    });

    sidebar.querySelectorAll("[data-object-source]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const source = btn.getAttribute("data-object-source");
        if (source === "mnist") {
          const idx = state.object.source === "mnist" ? state.object.mnist_index : (state.defaults?.mnist_index ?? 5);
          state.object = { source: "mnist", mnist_index: idx };
        } else if (source === "mask") {
          state.object = { source: "mask", mask_id: state.object.mask_id || 5 };
        } else if (source === "rabbit") {
          state.object = { source: "rabbit" };
        } else if (source === "upload") {
          if (state.object.source !== "upload") {
            state.object = { source: "upload", png_base64: "", file_name: undefined };
          }
        }
        renderSidebar();
        schedulePreview();
      });
    });

    sidebar.querySelectorAll("[data-mask-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.object = { source: "mask", mask_id: Number(btn.getAttribute("data-mask-id")) };
        renderSidebar();
        schedulePreview();
      });
    });

    const pickUpload = sidebar.querySelector("[data-action='pick-upload']");
    const fileInput = sidebar.querySelector("#mask-upload-input");
    pickUpload?.addEventListener("click", () => fileInput?.click());
    fileInput?.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      const errEl = sidebar.querySelector("#upload-error");
      if (!file.type.startsWith("image/")) {
        if (errEl) errEl.textContent = "please choose an image file (PNG preferred)";
        return;
      }
      const reader = new FileReader();
      reader.onerror = () => {
        if (errEl) errEl.textContent = "failed to read file";
      };
      reader.onload = () => {
        const result = String(reader.result || "");
        const comma = result.indexOf(",");
        const png_base64 = comma >= 0 ? result.slice(comma + 1) : result;
        if (!png_base64) {
          if (errEl) errEl.textContent = "empty file";
          return;
        }
        state.object = { source: "upload", png_base64, file_name: file.name || "upload.png" };
        renderSidebar();
        schedulePreview();
      };
      reader.readAsDataURL(file);
    });

    const clampMnist = (i) => {
      const max = (state.defaults?.mnist_train_count || 100) - 1;
      return Math.min(max, Math.max(0, Math.round(i)));
    };

    sidebar.querySelector("[data-action='mnist-prev']")?.addEventListener("click", () => {
      const idx = clampMnist((state.object.mnist_index ?? 0) - 1);
      state.object = { source: "mnist", mnist_index: idx };
      renderSidebar();
      schedulePreview();
    });
    sidebar.querySelector("[data-action='mnist-next']")?.addEventListener("click", () => {
      const idx = clampMnist((state.object.mnist_index ?? 0) + 1);
      state.object = { source: "mnist", mnist_index: idx };
      renderSidebar();
      schedulePreview();
    });
    sidebar.querySelector("[data-action='mnist-rand']")?.addEventListener("click", () => {
      const max = state.defaults?.mnist_train_count || 100;
      state.object = { source: "mnist", mnist_index: Math.floor(Math.random() * max) };
      renderSidebar();
      schedulePreview();
    });

    sidebar.querySelectorAll("[data-source-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.sourceMode = btn.getAttribute("data-source-mode");
        renderSidebar();
      });
    });

    sidebar.querySelectorAll("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => onNoisePreset(btn.getAttribute("data-preset")));
    });

    sidebar.querySelectorAll("input[type='range']").forEach((input) => {
      const apply = () => {
        const key = input.dataset.key;
        const value = parseSlider(input);
        updateSliderLabel(input);

        if (SOURCE.some((p) => p.key === key)) {
          setSourceLatent(key, value);
          return;
        }
        if (GEOMETRY.some((p) => p.key === key) ||
            MAGNIFICATION.near_field.some((p) => p.key === key) ||
            MAGNIFICATION.far_field.some((p) => p.key === key)) {
          setGeometry(key, value);
          return;
        }
        const noiseParam = NOISE_KEYS.find((p) => p.key === key);
        if (noiseParam) {
          if (noiseParam.target === "setup") {
            const next = { ...state.noisySetup, [key]: value };
            if (key === "delta_phi_align_span_rad" && value > 1e-6) {
              next.couple_align_to_wavefront = true;
            }
            state.noisySetup = next;
          } else {
            state.noisyLatents = { ...state.noisyLatents, [key]: value };
            // Physical A–D profiles ignore s_focus unless mechanical Δz is set.
            if (key === "s_focus") {
              state.noisySetup = { ...state.noisySetup, delta_z_um: value * 100 };
            }
          }
          markCustomPreset();
        }
      };
      input.addEventListener("input", apply);
      input.addEventListener("change", apply);
    });

    sidebar.querySelectorAll("[data-num]").forEach((input) => {
      input.addEventListener("change", () => {
        const key = input.getAttribute("data-num");
        let v = Number(input.value);
        if (!Number.isFinite(v)) return;
        if (key === "size") {
          const [lo, hi] = state.defaults.size_range || [32, 128];
          state.size = Math.min(hi, Math.max(lo, Math.round(v)));
          input.value = String(state.size);
          schedulePreview();
        } else if (key === "seed") {
          state.seed = Math.max(0, Math.round(v));
          input.value = String(state.seed);
        } else if (key === "mnist_index") {
          state.object = { source: "mnist", mnist_index: clampMnist(v) };
          input.value = String(state.object.mnist_index);
          schedulePreview();
        }
      });
    });

    sidebar.querySelectorAll("[data-toggle-key]").forEach((input) => {
      input.addEventListener("change", () => {
        const key = input.getAttribute("data-toggle-key");
        const checked = input.checked;
        if (key === "mode_ideal") state.modes = { ...state.modes, ideal: checked };
        else if (key === "mode_noisy") {
          state.modes = { ...state.modes, noisy: checked };
          updatePresetMetric();
        } else if (key === "blur_model") {
          state.noisySetup = { ...state.noisySetup, blur_model: checked ? "double_blur" : "single_blur" };
          markCustomPreset();
        } else if (key === "enable_lens_rings") {
          // Twin gates rings on enable_lens_rings AND (Δz or lens shift). Seed a
          // visible demo shift when the user turns the toggle on with all zeros.
          const next = { ...state.noisySetup, enable_lens_rings: checked };
          if (checked) {
            const dx = Math.abs(Number(next.lens_shift_x_um) || 0);
            const dy = Math.abs(Number(next.lens_shift_y_um) || 0);
            const dz = Math.abs(Number(next.delta_z_um) || 0);
            if (dx < 1e-9 && dy < 1e-9 && dz < 1e-9) {
              next.lens_shift_x_um = 20;
            }
          }
          state.noisySetup = next;
          markCustomPreset();
          renderSidebar();
        } else {
          state.noisySetup = { ...state.noisySetup, [key]: checked };
          markCustomPreset();
        }
      });
    });

    sidebar.querySelectorAll("[data-thermal]").forEach((input) => {
      input.addEventListener("change", () => {
        const id = input.getAttribute("data-thermal");
        const tkey = input.getAttribute("data-tkey");
        const v = Number(input.value);
        if (!Number.isFinite(v)) return;
        state.thermal = {
          ...state.thermal,
          [id]: { ...(state.thermal[id] || {}), [tkey]: v },
        };
      });
    });
  }

  function ensureTiles() {
    for (const mode of ["ideal", "noisy"]) {
      const grid = $(`${mode}-grid`);
      grid.innerHTML = PROFILE_IDS.map((id) => {
        const info = state.defaults.profiles[id];
        const enlarged =
          state.enlarge?.type === "result" &&
          state.enlarge.profile === id &&
          state.enlarge.mode === mode;
        return `
          <div class="card tile" id="tile-${id}-${mode}">
            <div class="image-head">
              <div class="min-w-0">
                <strong>${id} · ${escapeHtml(info.label)}</strong>
                <div class="muted">${escapeHtml((info.imaging_mode || "").replaceAll("_", " "))}</div>
              </div>
              <button type="button" class="icon-btn enlarge-btn ${enlarged ? "active" : ""}"
                data-enlarge="result" data-profile="${id}" data-mode="${mode}"
                title="Enlarge to fill the simulation panel (Esc to close)">⛶</button>
            </div>
            <div class="image-box" data-box aria-label="${id} ${mode} output">idle</div>
            <div class="stats" data-stats></div>
          </div>`;
      }).join("");
    }
    document.querySelectorAll('[data-enlarge="result"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        setEnlarge({
          type: "result",
          profile: btn.getAttribute("data-profile"),
          mode: btn.getAttribute("data-mode"),
        });
      });
    });
    // Restore any previously finished images into tiles after rebuild.
    for (const mode of ["ideal", "noisy"]) {
      for (const id of PROFILE_IDS) {
        const img = state.images[`${id}_${mode}`];
        if (img) setTile(id, mode, "done", { image: img, elapsed_s: img._elapsed_s });
      }
    }
  }

  function setTile(profile, mode, status, payload) {
    const tile = $(`tile-${profile}-${mode}`);
    if (!tile) return;
    const box = tile.querySelector("[data-box]");
    const stats = tile.querySelector("[data-stats]");
    if (status === "idle") {
      box.textContent = "idle";
      stats.textContent = "";
      return;
    }
    if (status === "running" || status === "queued") {
      box.textContent = status;
      stats.textContent = "";
      return;
    }
    if (status === "error") {
      box.textContent = payload?.message || "error";
      stats.textContent = "";
      return;
    }
    if (status === "done" && payload?.image) {
      const stored = { ...payload.image, _elapsed_s: payload.elapsed_s, _sha: payload.sha256 };
      state.images[`${profile}_${mode}`] = stored;
      setImage(box, payload.image, "—", `${profile} ${mode} camera output`);
      const img = payload.image;
      const elapsed = payload.elapsed_s != null ? `${Number(payload.elapsed_s).toFixed(2)} s` : "";
      const sha = payload.sha256 ? String(payload.sha256).slice(0, 10) : "";
      const unitHint = mode === "ideal" ? "arb. units" : "detector units";
      stats.innerHTML = `mean=${escapeHtml(String(img.mean?.toFixed?.(4) ?? img.mean))} <span class="muted">(${unitHint})</span>`
        + ` · min=${escapeHtml(String(img.min?.toFixed?.(4) ?? img.min))}`
        + ` · max=${escapeHtml(String(img.max?.toFixed?.(4) ?? img.max))}`
        + ` · ${escapeHtml(elapsed)}`
        + (sha ? ` · sha ${escapeHtml(sha)}…` : "");
      if (
        state.enlarge?.type === "result" &&
        state.enlarge.profile === profile &&
        state.enlarge.mode === mode
      ) {
        renderEnlarge();
      }
    }
  }

  async function loadDefaults() {
    const res = await apiFetch("/api/defaults");
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    state.defaults = data;
    resetFromDefaults(data);
    $("version-badge").textContent = `v${data.package_version}`;
    $("backend-pill").textContent = `backend · ${data.backend?.framework || "django"}`;
    renderSidebar();
    syncDiagProfileOptions();
    ensureTiles();
    renderDiagnostics();
    boot.classList.add("hidden");
    workspace.classList.remove("hidden");
    await refreshPreview();
  }

  async function refreshPreview() {
    if (!objectReady(state.object)) {
      $("preview-box").textContent = state.object.source === "upload" ? "upload a mask…" : "Waiting…";
      $("preview-title").textContent = "SLM object";
      $("preview-sub").textContent = "";
      delete state.images.preview;
      if (state.enlarge?.type === "preview") clearEnlarge();
      return;
    }
    const body = { object: state.object, size: state.size };
    $("preview-box").textContent = "loading…";
    try {
      const res = await apiFetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const ev = await res.json();
      if (!res.ok || ev.error) throw new Error(ev.error || ev.message || `HTTP ${res.status}`);
      state.images.preview = ev.image;
      setImage($("preview-box"), ev.image, "no image");
      if (ev.source === "mnist") {
        $("preview-title").textContent = `MNIST #${ev.mnist_index}`;
        $("preview-sub").textContent = `digit label ${ev.label} · ${state.size}×${state.size} px`;
      } else if (ev.source === "rabbit") {
        $("preview-title").textContent = "rabbit SLM mask";
        $("preview-sub").textContent = `${state.size}×${state.size} px amplitude mask`;
      } else if (ev.source === "upload") {
        $("preview-title").textContent = String(ev.label || "upload");
        $("preview-sub").textContent = `${state.size}×${state.size} px uploaded mask`;
      } else {
        $("preview-title").textContent = String(ev.label || "mask");
        $("preview-sub").textContent = `${state.size}×${state.size} px catalog mask`;
      }
      if (state.enlarge?.type === "preview") renderEnlarge();
    } catch (err) {
      $("preview-box").textContent = err.message;
      log("err", `preview: ${err.message}`);
    }
  }

  function archCaption(id) {
    const info = state.defaults?.profiles?.[id];
    if (!info) return id;
    const far = info.imaging_mode === "far_field";
    const coeff = info.object_coefficient || (info.architecture === "two_crystal" ? "T" : "R");
    return `${id} · ${far ? "FF" : "NF"} · ${coeff}`;
  }

  function fmtNum(v, digits = 4) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    if (Math.abs(n) >= 1000 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)) return n.toExponential(2);
    return n.toFixed(digits);
  }

  function prepareCanvas(canvas, opts = {}) {
    if (!canvas) return null;
    const parent = canvas.parentElement;
    const fallbackW = opts.cssW || Number(canvas.getAttribute("width")) || 640;
    const fallbackH = opts.cssH || Number(canvas.getAttribute("height")) || 260;
    const cssW = Math.max(
      280,
      Math.round(opts.cssW || canvas.clientWidth || parent?.clientWidth || fallbackW)
    );
    const cssH = Math.max(180, Math.round(opts.cssH || fallbackH));
    const dpr = Math.max(1, Math.min(3, opts.dpr || window.devicePixelRatio || 1));
    const bw = Math.max(1, Math.round(cssW * dpr));
    const bh = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    canvas.style.aspectRatio = `${cssW} / ${cssH}`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#0a1220";
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    return { ctx, w: cssW, h: cssH, dpr };
  }

  function drawAxes(ctx, pad, w, h) {
    ctx.strokeStyle = "rgba(148,163,184,0.45)";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, h - pad.b);
    ctx.lineTo(w - pad.r, h - pad.b);
    ctx.stroke();
  }

  function drawAxisLabels(ctx, pad, w, h, xLabel, yLabel) {
    ctx.save();
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "12px sans-serif";
    if (xLabel) {
      const tw = ctx.measureText(xLabel).width;
      ctx.fillText(xLabel, pad.l + (w - pad.l - pad.r - tw) / 2, h - 8);
    }
    if (yLabel) {
      ctx.translate(14, pad.t + (h - pad.t - pad.b) / 2);
      ctx.rotate(-Math.PI / 2);
      const tw = ctx.measureText(yLabel).width;
      ctx.fillText(yLabel, -tw / 2, 0);
    }
    ctx.restore();
  }

  function drawLineSeries(canvas, series, opts = {}) {
    const prepared = prepareCanvas(canvas, opts);
    if (!prepared) return;
    const { ctx, w, h } = prepared;
    if (opts.cacheKey) {
      state.chartCache[opts.cacheKey] = {
        kind: "line",
        series,
        opts: { ...opts, cacheKey: null },
        exportW: 1400,
        exportH: 560,
      };
    }
    const pad = { l: 58, r: 14, t: 22, b: 46 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const all = series.flatMap((s) => (s.y || []).filter((v) => Number.isFinite(v)));
    if (!all.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "13px sans-serif";
      ctx.fillText(opts.empty || "No data yet", pad.l, h / 2);
      drawAxisLabels(ctx, pad, w, h, opts.xLabel, opts.yLabel);
      return;
    }
    let ymin = opts.ymin != null ? opts.ymin : Math.min(...all);
    let ymax = opts.ymax != null ? opts.ymax : Math.max(...all);
    if (ymax <= ymin) {
      ymin -= 1;
      ymax += 1;
    }
    const xAll = series.flatMap((s, si) => {
      const y = s.y || [];
      return y.map((_, i) => (s.x && Number.isFinite(s.x[i]) ? Number(s.x[i]) : i));
    });
    let xmin = opts.xmin != null ? opts.xmin : Math.min(...xAll);
    let xmax = opts.xmax != null ? opts.xmax : Math.max(...xAll);
    if (xmax <= xmin) xmax = xmin + 1;
    drawAxes(ctx, pad, w, h);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.fillText(fmtNum(ymax, 3), 22, pad.t + 8);
    ctx.fillText(fmtNum(ymin, 3), 22, h - pad.b);
    ctx.fillText(fmtNum(xmin, 3), pad.l, h - pad.b + 14);
    const xmaxLabel = fmtNum(xmax, 3);
    ctx.fillText(xmaxLabel, w - pad.r - ctx.measureText(xmaxLabel).width, h - pad.b + 14);
    series.forEach((s) => {
      const y = s.y || [];
      if (y.length < 2) return;
      ctx.strokeStyle = s.color || "#38bdf8";
      ctx.lineWidth = s.width || 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      let started = false;
      y.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const xv = s.x && Number.isFinite(s.x[i]) ? Number(s.x[i]) : i;
        const x = pad.l + ((xv - xmin) / (xmax - xmin)) * plotW;
        const yy = pad.t + (1 - (v - ymin) / (ymax - ymin)) * plotH;
        if (!started) {
          ctx.moveTo(x, yy);
          started = true;
        } else ctx.lineTo(x, yy);
      });
      if (started) ctx.stroke();
      // Sample markers help hover targeting on sparse sweeps.
      y.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const xv = s.x && Number.isFinite(s.x[i]) ? Number(s.x[i]) : i;
        const x = pad.l + ((xv - xmin) / (xmax - xmin)) * plotW;
        const yy = pad.t + (1 - (v - ymin) / (ymax - ymin)) * plotH;
        ctx.fillStyle = s.color || "#38bdf8";
        ctx.beginPath();
        ctx.arc(x, yy, opts.hover ? 3.2 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      });
    });
    if (opts.hover?.px != null) {
      ctx.save();
      ctx.strokeStyle = "rgba(226, 232, 240, 0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(opts.hover.px, pad.t);
      ctx.lineTo(opts.hover.px, h - pad.b);
      ctx.stroke();
      ctx.setLineDash([]);
      (opts.hover.rows || []).forEach((r) => {
        if (!Number.isFinite(r.px) || !Number.isFinite(r.py)) return;
        ctx.fillStyle = r.color || "#fff";
        ctx.beginPath();
        ctx.arc(r.px, r.py, 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0a1220";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
      ctx.restore();
    }
    if (opts.legend) {
      let lx = pad.l + 4;
      opts.legend.forEach((item) => {
        ctx.fillStyle = item.color;
        ctx.fillRect(lx, 6, 11, 11);
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(item.label, lx + 16, 15);
        lx += ctx.measureText(item.label).width + 36;
      });
    }
    drawAxisLabels(ctx, pad, w, h, opts.xLabel, opts.yLabel);
  }

  function drawBars(canvas, groups, opts = {}) {
    const prepared = prepareCanvas(canvas, opts);
    if (!prepared) return;
    const { ctx, w, h } = prepared;
    if (opts.cacheKey) {
      state.chartCache[opts.cacheKey] = {
        kind: "bars",
        groups,
        opts: { ...opts, cacheKey: null },
        exportW: 1400,
        exportH: 560,
      };
    }
    const pad = { l: 58, r: 14, t: 28, b: 56 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const vals = groups.flatMap((g) => g.values.map((v) => v.value).filter((v) => Number.isFinite(v)));
    if (!vals.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "13px sans-serif";
      ctx.fillText(opts.empty || "Run simulations for A–D", pad.l, h / 2);
      drawAxisLabels(ctx, pad, w, h, opts.xLabel, opts.yLabel);
      return;
    }
    const ymax = Math.max(...vals, 1e-12);
    drawAxes(ctx, pad, w, h);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.fillText(fmtNum(ymax, 3), 22, pad.t + 8);
    ctx.fillText("0", 22, h - pad.b);
    const n = groups.length;
    const gap = 12;
    const cluster = Math.max(plotW / n - gap, 10);
    groups.forEach((g, gi) => {
      const x0 = pad.l + gi * (cluster + gap) + gap / 2;
      const bw = cluster / Math.max(g.values.length, 1);
      g.values.forEach((v, vi) => {
        if (!Number.isFinite(v.value)) return;
        const bh = (v.value / ymax) * plotH;
        const x = x0 + vi * bw;
        const y = pad.t + plotH - bh;
        ctx.fillStyle = v.color || "#38bdf8";
        ctx.fillRect(x + 1, y, Math.max(bw - 2, 2), bh);
      });
      ctx.fillStyle = "#cbd5e1";
      ctx.font = "11px sans-serif";
      ctx.fillText(g.label || "", x0, h - pad.b + 14);
    });
    if (opts.hover && Number.isFinite(opts.hover.px) && Number.isFinite(opts.hover.py)) {
      ctx.save();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(opts.hover.px, opts.hover.py, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    if (opts.legend) {
      let lx = pad.l;
      opts.legend.forEach((item) => {
        ctx.fillStyle = item.color;
        ctx.fillRect(lx, 6, 11, 11);
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(item.label, lx + 16, 15);
        lx += ctx.measureText(item.label).width + 36;
      });
    }
    drawAxisLabels(ctx, pad, w, h, opts.xLabel, opts.yLabel);
  }

  function drawHistogram(canvas, hists, opts = {}) {
    const prepared = prepareCanvas(canvas, opts);
    if (!prepared) return;
    const { ctx, w, h } = prepared;
    if (opts.cacheKey) {
      state.chartCache[opts.cacheKey] = {
        kind: "hist",
        hists,
        opts: { ...opts, cacheKey: null },
        exportW: 1400,
        exportH: 560,
      };
    }
    const pad = { l: 58, r: 14, t: 24, b: 46 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const usable = (hists || []).filter((hh) => hh?.counts?.length);
    if (!usable.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "13px sans-serif";
      ctx.fillText(opts.empty || "No histogram yet", pad.l, h / 2);
      drawAxisLabels(ctx, pad, w, h, opts.xLabel, opts.yLabel);
      return;
    }
    const maxC = Math.max(...usable.flatMap((hh) => hh.counts), 1);
    drawAxes(ctx, pad, w, h);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.fillText(fmtNum(maxC, 0), 22, pad.t + 8);
    ctx.fillText("0", 22, h - pad.b);
    // Intensity edges from first histogram when available.
    const edges = usable[0]?.edges;
    if (edges && edges.length >= 2) {
      ctx.fillText(fmtNum(edges[0], 3), pad.l, h - pad.b + 14);
      const last = fmtNum(edges[edges.length - 1], 3);
      ctx.fillText(last, w - pad.r - ctx.measureText(last).width, h - pad.b + 14);
    }
    usable.forEach((hh) => {
      const n = hh.counts.length;
      const bw = plotW / n;
      ctx.strokeStyle = hh.color || "#38bdf8";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      hh.counts.forEach((c, i) => {
        const x = pad.l + i * bw + bw * 0.5;
        const y = pad.t + plotH - (c / maxC) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    if (opts.hover?.px != null) {
      ctx.save();
      ctx.strokeStyle = "rgba(226, 232, 240, 0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(opts.hover.px, pad.t);
      ctx.lineTo(opts.hover.px, h - pad.b);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
    if (opts.legend) {
      let lx = pad.l + 4;
      opts.legend.forEach((item) => {
        ctx.fillStyle = item.color;
        ctx.fillRect(lx, 6, 11, 11);
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(item.label, lx + 16, 15);
        lx += ctx.measureText(item.label).width + 36;
      });
    }
    drawAxisLabels(ctx, pad, w, h, opts.xLabel, opts.yLabel);
  }

  function selectedDiagProfile() {
    return $("diag-profile")?.value || state.profiles[0] || "A";
  }

  function storeResultDiagnostics(ev) {
    if (!ev?.profile || !ev?.mode) return;
    if (["stage", "mechanism", "phase_step", "sweep"].includes(ev.mode)) return;
    state.diagnostics[`${ev.profile}_${ev.mode}`] = {
      diagnostics: ev.diagnostics || {},
      thermal: ev.thermal || [],
      architecture: ev.architecture || {},
      image: ev.image,
      elapsed_s: ev.elapsed_s,
    };
  }

  function renderVisibilityThermal(profile) {
    const visBox = $("diag-visibility");
    const thBox = $("diag-thermal");
    if (!visBox || !thBox) return;
    const ideal = state.diagnostics[`${profile}_ideal`];
    const noisy = state.diagnostics[`${profile}_noisy`];
    const bundle = noisy || ideal;
    if (!bundle) {
      visBox.innerHTML = `<p class="muted">No visibility data for ${escapeHtml(archCaption(profile))}.</p>`;
      thBox.innerHTML = "";
      return;
    }
    const vis = bundle.diagnostics?.visibility || {};
    const rows = [
      ["V₅₀₍₅₀₎", vis.V_50_50],
      ["N₁", vis.N_1],
      ["N₂", vis.N_2],
      ["|C₁₂|", vis.C_12_abs],
      ["n̄₁", vis.nbar_1],
      ["n̄₂", vis.nbar_2],
      ["Gain regime₁", vis.gain_regime_1],
      ["Gain regime₂", vis.gain_regime_2],
      ["Output port", bundle.diagnostics?.selected_output_port],
    ];
    visBox.innerHTML = `
      <table>
        <thead><tr><th colspan="2">${escapeHtml(archCaption(profile))} · ${noisy ? "noisy" : "ideal"}</th></tr></thead>
        <tbody>
          ${rows.map(([k, v]) => `<tr><th>${k}</th><td>${escapeHtml(v == null || v === "" ? "—" : (typeof v === "number" ? fmtNum(v) : String(v)))}</td></tr>`).join("")}
        </tbody>
      </table>
      ${vis.visibility_equation ? `<div class="muted" style="margin-top:0.35rem">${escapeHtml(String(vis.visibility_equation))}</div>` : ""}
    `;
    const thermal = bundle.thermal || [];
    if (!thermal.length) {
      thBox.innerHTML = `<p class="muted" style="margin-top:0.5rem">No thermal readout.</p>`;
      return;
    }
    thBox.innerHTML = `
      <table style="margin-top:0.55rem">
        <thead><tr><th>Crystal</th><th>T<sub>pm0</sub></th><th>ΔT</th><th>T<sub>crystal</sub></th></tr></thead>
        <tbody>
          ${thermal.map((r) => `
            <tr>
              <td>${escapeHtml(r.crystal || "—")}</td>
              <td>${escapeHtml(fmtNum(r.T_pm0_C, 2))} °C</td>
              <td>${escapeHtml(fmtNum(r.dT_C, 3))} °C</td>
              <td>${escapeHtml(fmtNum(r.T_crystal_C, 2))} °C</td>
            </tr>`).join("")}
        </tbody>
      </table>`;
  }

  function renderKpiStrip(profile) {
    const el = $("diag-kpi");
    if (!el) return;
    const ideal = state.diagnostics[`${profile}_ideal`];
    const noisy = state.diagnostics[`${profile}_noisy`];
    if (!ideal && !noisy) {
      el.classList.add("muted");
      el.textContent = "Run a simulation to populate diagnostics.";
      return;
    }
    el.classList.remove("muted");
    const parts = [`Arrangement ${archCaption(profile)}`];
    const pick = (bundle, label) => {
      if (!bundle) return;
      const d = bundle.diagnostics || {};
      const snr = d.snr;
      const v = d.visibility?.V_50_50;
      const rms = d.residual_rms;
      const cyx = d.centroid_yx;
      parts.push(
        `${label}: SNR=${fmtNum(snr, 2)} · V=${fmtNum(v, 4)}`
        + (rms != null ? ` · RMSΔ=${fmtNum(rms, 4)}` : "")
        + (cyx ? ` · centroid=(${fmtNum(cyx[0], 1)}, ${fmtNum(cyx[1], 1)})` : "")
      );
    };
    pick(ideal, "Ideal");
    pick(noisy, "Noisy");
    el.textContent = parts.join("  |  ");
  }

  function renderTransfer(profile) {
    const box = $("diag-transfer");
    if (!box) return;
    const bundle = state.diagnostics[`${profile}_noisy`] || state.diagnostics[`${profile}_ideal`];
    const t = bundle?.diagnostics?.transfer;
    if (!t) {
      box.classList.add("muted");
      box.textContent = "Available after a run.";
      return;
    }
    box.classList.remove("muted");
    const rows = [
      ["Object type", t.object_type],
      ["Object device", t.object_device],
      ["Object mode", t.object_mode],
      ["Interactions", t.interaction_count],
      ["T_eff min", t.intensity_transmission_min],
      ["T_eff max", t.intensity_transmission_max],
      ["T_eff mean", t.intensity_transmission_mean],
      ["φ_eff min (rad)", t.phase_rad_min],
      ["φ_eff max (rad)", t.phase_rad_max],
      ["φ_eff mean (rad)", t.phase_rad_mean],
    ];
    box.innerHTML = `
      <table>
        <thead><tr><th colspan="2">${escapeHtml(archCaption(profile))}</th></tr></thead>
        <tbody>
          ${rows.map(([k, v]) => `<tr><th>${k}</th><td>${escapeHtml(typeof v === "number" ? fmtNum(v) : (v == null ? "—" : String(v)))}</td></tr>`).join("")}
        </tbody>
      </table>
      ${t.transfer_equation ? `<div class="muted" style="margin-top:0.35rem">${escapeHtml(String(t.transfer_equation))}</div>` : ""}
    `;
  }

  function diagMapPayload(plot) {
    const profile = selectedDiagProfile();
    const noisy = state.diagnostics[`${profile}_noisy`];
    const ideal = state.diagnostics[`${profile}_ideal`];
    const d = (noisy || ideal)?.diagnostics || {};
    if (plot === "residual") return noisy?.diagnostics?.residual_to_ideal || null;
    if (plot === "diffport") return d.difference_port || null;
    if (plot === "teff") return d.T_eff_map || null;
    if (plot === "iplus") return d.plus_port || null;
    if (plot === "iminus") return d.minus_port || null;
    if (plot === "phase-map") return state.labPhase?.diagnostics?.phase_map || null;
    if (plot === "phase-mod") return state.labPhase?.diagnostics?.phase_modulation_map || null;
    return null;
  }

  function setDiagImage(boxId, payload, emptyText) {
    const box = $(boxId);
    if (!box) return;
    if (payload?.png) {
      box.classList.remove("muted");
      const note = payload.constant
        ? `<div class="diag-map-note">Constant field (min=max=${escapeHtml(fmtNum(payload.mean))})</div>`
        : "";
      box.innerHTML = `
        <img class="pixelated" src="data:image/png;base64,${payload.png}" alt="" />
        <div class="diag-map-stats">min=${escapeHtml(fmtNum(payload.min))} · max=${escapeHtml(fmtNum(payload.max))} · mean=${escapeHtml(fmtNum(payload.mean))}</div>
        ${note}`;
    } else {
      box.classList.add("muted");
      box.textContent = emptyText;
    }
  }

  function renderMapPanels(profile) {
    const noisy = state.diagnostics[`${profile}_noisy`];
    const ideal = state.diagnostics[`${profile}_ideal`];
    const d = (noisy || ideal)?.diagnostics || {};
    setDiagImage(
      "diag-residual",
      noisy?.diagnostics?.residual_to_ideal,
      ideal && !noisy
        ? "Run Noisy as well (same arrangement) to build Ideal − Noisy."
        : "Needs both Ideal and Noisy results for this arrangement."
    );
    setDiagImage("diag-diffport", d.difference_port, "Available after a run.");
    setDiagImage("diag-iplus", d.plus_port, "Available after a run.");
    setDiagImage("diag-iminus", d.minus_port, "Available after a run.");
    setDiagImage("diag-teff", d.T_eff_map, "Available after a run.");
    renderOperatingPoint(profile);
  }

  function michelsonContrast(values) {
    const f = (values || []).map(Number).filter((v) => Number.isFinite(v));
    if (f.length < 2) return NaN;
    const lo = Math.min(...f);
    const hi = Math.max(...f);
    const den = Math.abs(hi) + Math.abs(lo);
    if (den < 1e-15) return 0;
    return (hi - lo) / den;
  }

  function archStripHtml() {
    const modes = ["ideal", "noisy"].filter((m) =>
      PROFILE_IDS.some((id) => state.images[`${id}_${m}`]?.png)
    );
    if (!modes.length) return "";
    return modes.map((mode) => `
      <div class="diag-arch-row">
        <div class="diag-arch-row-label">${mode === "ideal" ? "Ideal" : "Noisy"}</div>
        <div class="diag-arch-thumbs">
          ${PROFILE_IDS.map((id) => {
            const img = state.images[`${id}_${mode}`];
            const diag = state.diagnostics[`${id}_${mode}`]?.diagnostics;
            const snr = diag?.snr;
            const v = diag?.visibility?.V_50_50;
            const enlarged =
              state.enlarge?.type === "diag" &&
              state.enlarge?.plot === "arch-thumb" &&
              state.enlarge?.profile === id &&
              state.enlarge?.mode === mode;
            return `
              <div class="diag-arch-thumb" title="${escapeHtml(archCaption(id))}">
                <div class="diag-arch-thumb-head">
                  <strong>${escapeHtml(archCaption(id))}</strong>
                  <button type="button" class="icon-btn enlarge-btn ${enlarged ? "active" : ""}"
                    data-enlarge="diag" data-plot="arch-thumb"
                    data-profile="${escapeHtml(id)}" data-mode="${escapeHtml(mode)}"
                    title="Enlarge this ${mode} frame (Esc to close)" ${img?.png ? "" : "disabled"}>⛶</button>
                </div>
                ${img?.png
                  ? `<img class="pixelated" src="${pngSrc(img)}" alt="${escapeHtml(id + " " + mode)}" />`
                  : `<div class="muted tiny">—</div>`}
                <div class="meta">SNR ${escapeHtml(fmtNum(snr, 2))} · V ${escapeHtml(fmtNum(v, 3))}</div>
              </div>`;
          }).join("")}
        </div>
      </div>`).join("");
  }

  function renderArchStrip() {
    const box = $("diag-arch-strip");
    if (!box) return;
    const html = archStripHtml();
    if (!html) {
      box.classList.add("muted");
      box.textContent = "Run A–D Ideal and/or Noisy to compare camera frames.";
      return;
    }
    box.classList.remove("muted");
    box.innerHTML = html;
  }

  function renderOperatingPoint(profile) {
    const box = $("diag-operating");
    if (!box) return;
    const bundle = state.diagnostics[`${profile}_noisy`] || state.diagnostics[`${profile}_ideal`];
    const op = bundle?.diagnostics?.operating_point;
    if (!op) {
      box.innerHTML = "";
      return;
    }
    const rows = [
      ["Detector mode", op.detector_mode],
      ["Selected port", op.selected_output_port],
      ["n̄₁", op.nbar_1],
      ["n̄₂", op.nbar_2],
      ["Gain r₁", op.gain_r_1],
      ["Gain r₂", op.gain_r_2],
      ["Regime₁", op.gain_regime_1],
      ["Regime₂", op.gain_regime_2],
      ["V₅₀₍₅₀₎", op.V_50_50],
    ];
    box.innerHTML = `
      <strong style="font-size:0.78rem">SPDC operating point · ${escapeHtml(archCaption(profile))}</strong>
      <table style="margin-top:0.35rem">
        <tbody>
          ${rows.map(([k, v]) => `<tr><th>${k}</th><td>${escapeHtml(typeof v === "number" ? fmtNum(v) : (v == null ? "—" : String(v)))}</td></tr>`).join("")}
        </tbody>
      </table>`;
  }

  function renderProfileCharts(profile) {
    const ideal = state.diagnostics[`${profile}_ideal`];
    const noisy = state.diagnostics[`${profile}_noisy`];
    const cut = $("diag-cut")?.value === "col" ? "col" : "row";
    const cutLabel = cut === "col" ? "mid-column" : "mid-row";
    const slm = ideal?.diagnostics?.slm_profiles || noisy?.diagnostics?.slm_profiles;
    const series = [];
    const legend = [];
    if (slm?.[cut]) {
      series.push({ y: slm[cut], color: "#94a3b8" });
      legend.push({ label: `SLM ${cut}`, color: "#94a3b8" });
    }
    if (ideal?.diagnostics?.profiles?.[cut]) {
      series.push({ y: ideal.diagnostics.profiles[cut], color: "#38bdf8" });
      legend.push({ label: "Ideal", color: "#38bdf8" });
    }
    if (noisy?.diagnostics?.profiles?.[cut]) {
      series.push({ y: noisy.diagnostics.profiles[cut], color: "#f97316" });
      legend.push({ label: "Noisy", color: "#f97316" });
    }
    drawLineSeries($("diag-profiles"), series, {
      legend,
      empty: `Run ideal/noisy to plot ${cutLabel} cuts`,
      cssH: 280,
      cacheKey: "profiles",
      xLabel: `Pixel index (${cutLabel})`,
      yLabel: "Intensity",
    });

    const hists = [];
    const hLeg = [];
    if (ideal?.diagnostics?.histogram) {
      hists.push({ ...ideal.diagnostics.histogram, color: "#38bdf8" });
      hLeg.push({ label: "Ideal", color: "#38bdf8" });
    }
    if (noisy?.diagnostics?.histogram) {
      hists.push({ ...noisy.diagnostics.histogram, color: "#f97316" });
      hLeg.push({ label: "Noisy", color: "#f97316" });
    }
    drawHistogram($("diag-hist"), hists, {
      legend: hLeg,
      empty: "Histograms appear after a run",
      cssH: 280,
      cacheKey: "hist",
      xLabel: "Intensity",
      yLabel: "Pixel count",
    });
  }

  function renderArchBars() {
    const groups = PROFILE_IDS.map((id) => {
      const ideal = state.diagnostics[`${id}_ideal`];
      const noisy = state.diagnostics[`${id}_noisy`];
      const vIdeal = ideal?.diagnostics?.visibility?.V_50_50;
      const vNoisy = noisy?.diagnostics?.visibility?.V_50_50;
      const meanIdeal = ideal?.image?.mean;
      const meanNoisy = noisy?.image?.mean;
      // Prefer visibility when present; fall back to mean intensity.
      const useVis = Number.isFinite(Number(vIdeal)) || Number.isFinite(Number(vNoisy));
      return {
        label: archCaption(id),
        values: useVis
          ? [
              { value: Number(vIdeal), color: "#38bdf8" },
              { value: Number(vNoisy), color: "#f97316" },
            ]
          : [
              { value: Number(meanIdeal), color: "#38bdf8" },
              { value: Number(meanNoisy), color: "#f97316" },
            ],
        metric: useVis ? "V" : "mean",
      };
    });
    const any = groups.some((g) => g.values.some((v) => Number.isFinite(v.value)));
    const useVis = any && groups.some((g) => g.metric === "V");
    drawBars($("diag-arch-bars"), any ? groups : [], {
      legend: [
        { label: "Ideal", color: "#38bdf8" },
        { label: "Noisy", color: "#f97316" },
      ],
      empty: "Run A–D to compare V or mean intensity",
      cssH: 280,
      cacheKey: "arch",
      xLabel: "Arrangement (NF/FF · T/R)",
      yLabel: useVis ? "Visibility V₅₀₍₅₀₎" : "Mean intensity",
    });
  }

  function renderStageStrip() {
    const strip = $("diag-stage-strip");
    const hint = $("diag-stage-hint");
    if (!strip) return;
    const profile = selectedDiagProfile();
    const stages = (state.stages || []).filter((s) => s.profile === profile);
    if (!stages.length) {
      strip.innerHTML = "";
      if (hint) {
        hint.textContent = "Click “Run stage ladder” for Ideal → Optics → Drift → Shot/Read → Detector → Full.";
      }
      drawBars($("diag-stage-kpi"), [], {
        empty: "Stage KPI chart after ladder run",
        cssH: 300,
        cacheKey: "stage-kpi",
        xLabel: "Noise stage",
        yLabel: "KPI value",
      });
      syncEnlargeButtons();
      return;
    }
    if (hint) {
      hint.textContent = `${archCaption(profile)} · ${stages.length} stages (image · Δ to ideal · KPIs)`;
    }
    strip.innerHTML = stages.map((s) => {
      const d = s.diagnostics || {};
      const snr = fmtNum(d.snr, 2);
      const v = fmtNum(d.visibility?.V_50_50, 3);
      const rms = d.residual_rms != null ? fmtNum(d.residual_rms, 3) : "—";
      const png = s.image?.png;
      const sid = escapeHtml(s.stage_id || "");
      const enlarged =
        state.enlarge?.type === "diag" &&
        state.enlarge?.plot === "stage-img" &&
        state.enlarge?.stageId === String(s.stage_id);
      return `
        <div class="diag-stage-card" title="${escapeHtml(s.stage_explain || "")}">
          <div class="diag-stage-card-head">
            <strong>${escapeHtml(s.stage_label || s.stage_id)}</strong>
            <button type="button" class="icon-btn enlarge-btn ${enlarged ? "active" : ""}"
              data-enlarge="diag" data-plot="stage-img" data-stage-id="${sid}"
              title="Enlarge this stage image (Esc to close)">⛶</button>
          </div>
          ${png ? `<img class="pixelated" src="data:image/png;base64,${png}" alt="${escapeHtml(s.stage_label || s.stage_id)}" />` : ""}
          <div class="meta">SNR ${escapeHtml(snr)} · V ${escapeHtml(v)} · RMSΔ ${escapeHtml(rms)}</div>
        </div>`;
    }).join("");

    const groups = stages.map((s) => ({
      label: (s.stage_id || "").replace(/_/g, " "),
      values: [
        { value: Number(s.diagnostics?.snr), color: "#38bdf8" },
        { value: Number(s.diagnostics?.visibility?.V_50_50), color: "#a78bfa" },
        {
          value: s.diagnostics?.residual_rms != null ? Number(s.diagnostics.residual_rms) : NaN,
          color: "#f97316",
        },
      ],
    }));
    drawBars($("diag-stage-kpi"), groups, {
      legend: [
        { label: "SNR", color: "#38bdf8" },
        { label: "V₅₀₍₅₀₎", color: "#a78bfa" },
        { label: "RMS residual", color: "#f97316" },
      ],
      cssH: 300,
      cacheKey: "stage-kpi",
      xLabel: "Noise stage",
      yLabel: "KPI value (SNR / V / RMSΔ)",
    });
    syncEnlargeButtons();
  }

  function renderLabPanels() {
    const strip = $("lab-mechanism-strip");
    const hint = $("lab-hint");
    const profile = selectedDiagProfile();
    const mechs = (state.mechanisms || []).filter((s) => s.profile === profile);
    if (strip) {
      if (!mechs.length) {
        strip.innerHTML = "";
      } else {
        strip.innerHTML = mechs.map((s) => {
          const d = s.diagnostics || {};
          const snr = fmtNum(d.snr, 2);
          const v = fmtNum(d.visibility?.V_50_50, 3);
          const rms = d.residual_rms != null ? fmtNum(d.residual_rms, 3) : "—";
          const png = s.image?.png;
          const sid = escapeHtml(s.stage_id || "");
          const enlarged =
            state.enlarge?.type === "diag" &&
            state.enlarge?.plot === "mech-img" &&
            state.enlarge?.stageId === String(s.stage_id);
          return `
            <div class="diag-stage-card" title="${escapeHtml(s.stage_explain || "")}">
              <div class="diag-stage-card-head">
                <strong>${escapeHtml(s.stage_label || s.stage_id)}</strong>
                <button type="button" class="icon-btn enlarge-btn ${enlarged ? "active" : ""}"
                  data-enlarge="diag" data-plot="mech-img" data-stage-id="${sid}"
                  title="Enlarge this mechanism image (Esc to close)">⛶</button>
              </div>
              ${png ? `<img class="pixelated" src="data:image/png;base64,${png}" alt="" />` : ""}
              <div class="meta">SNR ${escapeHtml(snr)} · V ${escapeHtml(v)} · RMSΔ ${escapeHtml(rms)}</div>
            </div>`;
        }).join("");
      }
    }
    const phase = state.labPhase;
    setDiagImage("lab-phase-map", phase?.diagnostics?.phase_map, "Run phase-step first.");
    setDiagImage("lab-phase-mod", phase?.diagnostics?.phase_modulation_map, "Run phase-step first.");

    const curve = state.sweepCurve?.curve || [];
    const sweepTable = $("lab-sweep-table");
    if (curve.length) {
      const xs = curve.map((p) => Number(p.value));
      const snr = curve.map((p) => Number(p.snr));
      const v50 = curve.map((p) => Number(p.V_50_50));
      const rms = curve.map((p) => Number(p.residual_rms));
      const mean = curve.map((p) => Number(p.mean));
      const rmsNorm = curve.map((p, i) => {
        const m = mean[i];
        const r = rms[i];
        if (!Number.isFinite(r) || !Number.isFinite(m) || Math.abs(m) < 1e-12) return NaN;
        return r / Math.abs(m);
      });
      const varies = (arr) => {
        const f = arr.filter((v) => Number.isFinite(v));
        if (f.length < 2) return false;
        const lo = Math.min(...f);
        const hi = Math.max(...f);
        return hi - lo > 1e-4 * (1 + Math.max(Math.abs(lo), Math.abs(hi)));
      };
      // Relative % change keeps unlike metrics comparable; skip flat series.
      // If the first point is ~0 (e.g. residual vs matched ref), scale by max |y|.
      const relPct = (arr) => {
        const base = arr.find((v) => Number.isFinite(v));
        if (!Number.isFinite(base) || Math.abs(base) < 1e-12) {
          const scale = Math.max(0, ...arr.map((v) => (Number.isFinite(v) ? Math.abs(v) : 0)));
          if (scale < 1e-12) return arr.map(() => NaN);
          return arr.map((v) => (Number.isFinite(v) ? (100 * v) / scale : NaN));
        }
        return arr.map((v) => (Number.isFinite(v) ? (100 * (v - base)) / Math.abs(base) : NaN));
      };
      const series = [];
      const legend = [];
      if (varies(snr)) {
        series.push({ x: xs, y: relPct(snr), color: "#38bdf8" });
        legend.push({ label: "SNR (% vs first)", color: "#38bdf8" });
      }
      if (varies(v50)) {
        series.push({ x: xs, y: relPct(v50), color: "#a78bfa" });
        legend.push({ label: "V (% vs first)", color: "#a78bfa" });
      }
      if (varies(rmsNorm)) {
        series.push({ x: xs, y: relPct(rmsNorm), color: "#f97316" });
        legend.push({ label: "RMSΔ/mean (% vs first)", color: "#f97316" });
      } else if (varies(rms)) {
        series.push({ x: xs, y: relPct(rms), color: "#f97316" });
        legend.push({ label: "RMSΔ (% vs first)", color: "#f97316" });
      }
      if (varies(mean)) {
        series.push({ x: xs, y: relPct(mean), color: "#34d399" });
        legend.push({ label: "mean (% vs first)", color: "#34d399" });
      }
      drawLineSeries($("lab-sweep"), series, {
        legend,
        empty: "No varying metrics for this sweep. Try s_focus, s_thermal, EM gain, or n̄.",
        cssH: 280,
        cacheKey: "sweep",
        xLabel: state.sweepCurve?.label || "Parameter",
        yLabel: "Relative change (%)",
      });
      if (sweepTable) {
        sweepTable.innerHTML = `
          <table>
            <thead><tr><th>${escapeHtml(state.sweepCurve?.label || "x")}</th><th>SNR</th><th>V</th><th>mean</th><th>RMSΔ</th><th>RMSΔ/mean</th></tr></thead>
            <tbody>
              ${curve.map((p, i) => `
                <tr>
                  <td>${escapeHtml(fmtNum(p.value, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.snr, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.V_50_50, 4))}</td>
                  <td>${escapeHtml(fmtNum(p.mean, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.residual_rms, 3))}</td>
                  <td>${escapeHtml(fmtNum(rmsNorm[i], 4))}</td>
                </tr>`).join("")}
            </tbody>
          </table>`;
      }
    } else {
      drawLineSeries($("lab-sweep"), [], {
        empty: "Run a parameter sweep (EM gain or n̄ recommended)",
        cssH: 280,
        cacheKey: "sweep",
        xLabel: "Parameter",
        yLabel: "Relative change (%)",
      });
      if (sweepTable) sweepTable.innerHTML = "";
    }

    const fringe = state.fringeCurve?.curve || [];
    const fringeTable = $("lab-fringe-table");
    const fringeSummary = $("lab-fringe-summary");
    if (fringe.length) {
      const xs = fringe.map((p) => Number(p.phi_rad));
      const iplus = fringe.map((p) => Number(p.mean_I_plus));
      const iminus = fringe.map((p) => Number(p.mean_I_minus));
      const idiff = fringe.map((p) => Number(p.mean_D));
      const series = [
        { x: xs, y: iplus, color: "#38bdf8" },
        { x: xs, y: iminus, color: "#a78bfa" },
        { x: xs, y: idiff, color: "#f97316" },
      ];
      drawLineSeries($("lab-fringe"), series, {
        legend: [
          { label: "⟨I₊⟩", color: "#38bdf8" },
          { label: "⟨I₋⟩", color: "#a78bfa" },
          { label: "⟨D⟩", color: "#f97316" },
        ],
        empty: "Run I(φ) fringe first.",
        cssH: 280,
        cacheKey: "fringe",
        xLabel: state.fringeCurve?.label || "φ (rad)",
        yLabel: "Mean intensity",
      });
      const vPlus = michelsonContrast(iplus);
      const vMinus = michelsonContrast(iminus);
      const vDiff = michelsonContrast(idiff);
      if (fringeSummary) {
        fringeSummary.classList.remove("muted");
        fringeSummary.textContent =
          `Michelson from I(φ): V₊=${fmtNum(vPlus, 4)} · V₋=${fmtNum(vMinus, 4)} · V_D=${fmtNum(vDiff, 4)}`
          + `  (${fringe.length} φ points · 0…2π)`;
      }
      if (fringeTable) {
        fringeTable.innerHTML = `
          <table>
            <thead><tr><th>φ (rad)</th><th>⟨I₊⟩</th><th>⟨I₋⟩</th><th>⟨D⟩</th><th>V</th></tr></thead>
            <tbody>
              ${fringe.map((p) => `
                <tr>
                  <td>${escapeHtml(fmtNum(p.phi_rad, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.mean_I_plus, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.mean_I_minus, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.mean_D, 3))}</td>
                  <td>${escapeHtml(fmtNum(p.V_50_50, 4))}</td>
                </tr>`).join("")}
            </tbody>
          </table>`;
      }
    } else {
      drawLineSeries($("lab-fringe"), [], {
        empty: "Run I(φ) fringe to see interferometric response vs object phase",
        cssH: 280,
        cacheKey: "fringe",
        xLabel: "φ (rad)",
        yLabel: "Mean intensity",
      });
      if (fringeSummary) {
        fringeSummary.classList.add("muted");
        fringeSummary.textContent = "Run I(φ) fringe for Michelson visibility from the curve.";
      }
      if (fringeTable) fringeTable.innerHTML = "";
    }

    if (hint && !mechs.length && !phase && !curve.length && !fringe.length) {
      hint.textContent = "Run a lab diagnostic to populate the matrix, phase maps, fringe curve, or sweep.";
    }
  }

  function renderDiagnostics() {
    const profile = selectedDiagProfile();
    renderKpiStrip(profile);
    renderVisibilityThermal(profile);
    renderTransfer(profile);
    renderProfileCharts(profile);
    renderMapPanels(profile);
    renderArchBars();
    renderArchStrip();
    renderStageStrip();
    renderLabPanels();
    if (state.enlarge?.type === "diag") renderEnlarge();
  }

  function syncDiagProfileOptions() {
    const sel = $("diag-profile");
    if (!sel || !state.defaults?.profiles) return;
    const current = sel.value;
    sel.innerHTML = PROFILE_IDS.map((id) => {
      const info = state.defaults.profiles[id];
      const far = info.imaging_mode === "far_field";
      const coeff = info.object_coefficient || (info.architecture === "two_crystal" ? "T" : "R");
      return `<option value="${id}">${id} · ${far ? "FF" : "NF"} · ${coeff}</option>`;
    }).join("");
    sel.value = PROFILE_IDS.includes(current) ? current : (state.profiles[0] || "A");
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "start":
        $("m-progress").textContent = `0 / ${ev.total}`;
        $("output-dir").textContent = ev.output_dir || "";
        $("output-dir").classList.toggle("hidden", !ev.output_dir);
        $("run-badge").textContent = "running";
        if (ev.diagnostic === "stage_ladder") {
          state.stages = [];
          log("info", `stage ladder: ${(ev.profiles || []).join(", ")}`);
        } else if (ev.diagnostic === "mechanism_matrix") {
          state.mechanisms = [];
          log("info", `mechanism matrix: ${(ev.profiles || []).join(", ")}`);
        } else if (ev.diagnostic === "phase_step") {
          state.labPhase = null;
          log("info", `phase-step: ${(ev.profiles || []).join(", ")}`);
        } else if (ev.diagnostic === "sweep") {
          state.sweepCurve = { param: ev.sweep?.param, label: ev.sweep?.label, curve: [] };
          log("info", `sweep ${ev.sweep?.label || ""}: ${(ev.profiles || []).join(", ")}`);
        } else if (ev.diagnostic === "fringe_curve") {
          state.fringeCurve = { label: "Object phase φ (rad)", curve: [] };
          log("info", `I(φ) fringe: ${(ev.profiles || []).join(", ")}`);
        } else {
          log("info", `run started: ${(ev.profiles || []).join(", ")}`);
        }
        break;
      case "object":
        state.images.preview = ev.image;
        setImage($("preview-box"), ev.image);
        log("info", `SLM object ready (${ev.source})`);
        if (state.enlarge?.type === "preview") renderEnlarge();
        break;
      case "progress":
        if (["stage", "mechanism", "phase_step", "sweep", "fringe"].includes(ev.mode)) {
          if (ev.stage_label) log("info", `${ev.mode} ${ev.profile}/${ev.stage_label}…`);
        } else {
          setTile(ev.profile, ev.mode, "running");
        }
        break;
      case "result":
        if (ev.mode === "stage") {
          state.stages.push({
            profile: ev.profile,
            stage_id: ev.stage_id,
            stage_label: ev.stage_label,
            stage_family: ev.stage_family,
            stage_explain: ev.stage_explain,
            image: ev.image,
            diagnostics: ev.diagnostics || {},
            thermal: ev.thermal || [],
            architecture: ev.architecture || {},
            elapsed_s: ev.elapsed_s,
          });
          renderStageStrip();
          log("ok", `${ev.profile} stage ${ev.stage_id} in ${Number(ev.elapsed_s).toFixed(2)} s`);
        } else if (ev.mode === "mechanism") {
          state.mechanisms.push({
            profile: ev.profile,
            stage_id: ev.stage_id,
            stage_label: ev.stage_label,
            stage_family: ev.stage_family,
            stage_explain: ev.stage_explain,
            image: ev.image,
            diagnostics: ev.diagnostics || {},
            elapsed_s: ev.elapsed_s,
          });
          renderLabPanels();
          log("ok", `${ev.profile} mechanism ${ev.stage_id} in ${Number(ev.elapsed_s).toFixed(2)} s`);
        } else if (ev.mode === "phase_step") {
          state.labPhase = {
            profile: ev.profile,
            image: ev.image,
            diagnostics: ev.diagnostics || {},
            elapsed_s: ev.elapsed_s,
          };
          renderLabPanels();
          log("ok", `${ev.profile} phase-step in ${Number(ev.elapsed_s).toFixed(2)} s`);
        } else if (ev.mode === "sweep") {
          if (ev.sweep_point && state.sweepCurve) {
            state.sweepCurve.curve.push(ev.sweep_point);
          }
          renderLabPanels();
          log("ok", `${ev.profile} sweep ${ev.stage_id} in ${Number(ev.elapsed_s).toFixed(2)} s`);
        } else if (ev.mode === "fringe") {
          if (ev.fringe_point && state.fringeCurve) {
            state.fringeCurve.curve.push(ev.fringe_point);
          }
          renderLabPanels();
          log("ok", `${ev.profile} fringe ${ev.stage_id} in ${Number(ev.elapsed_s).toFixed(2)} s`);
        } else {
          setTile(ev.profile, ev.mode, "done", ev);
          storeResultDiagnostics(ev);
          renderDiagnostics();
          log("ok", `${ev.profile} ${ev.mode} done in ${Number(ev.elapsed_s).toFixed(2)} s`);
        }
        {
          const parts = $("m-progress").textContent.split("/");
          const done = Number(parts[0]?.trim()) || 0;
          const total = Number(parts[1]?.trim()) || 0;
          const next = done + 1;
          $("m-progress").textContent = `${next} / ${total || next}`;
          $("progress-fill").style.width = total ? `${Math.round((next / total) * 100)}%` : "100%";
        }
        break;
      case "sweep_curve":
        state.sweepCurve = {
          param: ev.param,
          label: ev.label,
          curve: ev.curve || [],
        };
        renderLabPanels();
        break;
      case "fringe_curve":
        state.fringeCurve = {
          label: ev.label || "Object phase φ (rad)",
          curve: ev.curve || [],
          summary: ev.summary || null,
        };
        renderLabPanels();
        break;
      case "error":
        if (ev.profile && ev.mode && !["stage", "mechanism", "phase_step", "sweep", "fringe"].includes(ev.mode)) {
          setTile(ev.profile, ev.mode, "error", ev);
        }
        log("err", ev.message || "error");
        break;
      case "done":
        $("m-elapsed").textContent = `${Number(ev.elapsed_s).toFixed(1)} s`;
        $("run-badge").textContent = "complete";
        $("output-dir").textContent = ev.output_dir || "";
        $("output-dir").classList.toggle("hidden", !ev.output_dir);
        renderDiagnostics();
        log("ok", `finished in ${Number(ev.elapsed_s).toFixed(2)} s`);
        break;
      default:
        break;
    }
  }

  async function streamNdjson(url, config) {
    const res = await apiFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
      signal: state.abort.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        handleEvent(JSON.parse(line));
      }
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
  }

  async function runDiagnosticsStream(diagnostic, extra = {}) {
    if (state.running) return;
    const config = { ...buildConfig(), diagnostic, ...extra };
    const focus = selectedDiagProfile();
    config.profiles = [focus];
    if (!objectReady(config.object)) {
      log("err", "upload an SLM mask image before running");
      return;
    }
    state.running = true;
    state.abort = new AbortController();
    $("btn-run").classList.add("hidden");
    $("btn-stop").classList.remove("hidden");
    ["btn-stage-ladder", "btn-mechanism-matrix", "btn-phase-step", "btn-fringe-curve", "btn-sweep"].forEach((id) => {
      $(id)?.setAttribute("disabled", "true");
    });
    $("progress-fill").style.width = "0%";
    $("m-profiles").textContent = `${focus} · ${diagnostic}`;
    $("m-elapsed").textContent = "…";
    $("run-badge").textContent = "running";
    const hint = diagnostic === "stage_ladder" ? $("diag-stage-hint") : $("lab-hint");
    if (hint) hint.textContent = `Running ${diagnostic} for ${archCaption(focus)}…`;
    try {
      await streamNdjson("/api/diagnostics", config);
    } catch (err) {
      if (err.name === "AbortError") {
        log("err", `${diagnostic} cancelled`);
        $("run-badge").textContent = "idle";
      } else {
        log("err", err.message || String(err));
        $("run-badge").textContent = "error";
      }
    } finally {
      state.running = false;
      state.abort = null;
      $("btn-run").classList.remove("hidden");
      $("btn-stop").classList.add("hidden");
      ["btn-stage-ladder", "btn-mechanism-matrix", "btn-phase-step", "btn-fringe-curve", "btn-sweep"].forEach((id) => {
        $(id)?.removeAttribute("disabled");
      });
      const badge = $("run-badge");
      if (badge?.textContent === "running") badge.textContent = "idle";
      if (badge?.textContent === "complete") {
        window.setTimeout(() => {
          if ($("run-badge")?.textContent === "complete") $("run-badge").textContent = "idle";
        }, 2000);
      }
      renderDiagnostics();
    }
  }

  async function runStageLadder() {
    await runDiagnosticsStream("stage_ladder");
  }

  async function runSimulation() {
    if (state.running) return;
    const config = buildConfig();
    if (!config.profiles.length || (!config.modes.ideal && !config.modes.noisy)) {
      log("err", "Select at least one arrangement and one mode.");
      return;
    }
    if (!objectReady(config.object)) {
      log("err", "upload an SLM mask image before running");
      return;
    }
    state.running = true;
    state.abort = new AbortController();
    clearEnlarge();
    $("btn-run").classList.add("hidden");
    $("btn-stop").classList.remove("hidden");
    $("progress-fill").style.width = "0%";
    $("m-profiles").textContent = config.profiles.join(" · ");
    $("m-elapsed").textContent = "…";
    $("run-badge").textContent = "running";
    for (const p of PROFILE_IDS) {
      for (const mode of ["ideal", "noisy"]) {
        const active = config.profiles.includes(p) && config.modes[mode];
        if (!active) {
          delete state.images[`${p}_${mode}`];
          delete state.diagnostics[`${p}_${mode}`];
        }
        setTile(p, mode, active ? "queued" : "idle");
      }
    }
    try {
      await streamNdjson("/api/run", config);
    } catch (err) {
      if (err.name === "AbortError") {
        log("err", "run cancelled");
        $("run-badge").textContent = "idle";
      } else {
        log("err", err.message || String(err));
        $("run-badge").textContent = "error";
      }
    } finally {
      state.running = false;
      state.abort = null;
      $("btn-run").classList.remove("hidden");
      $("btn-stop").classList.add("hidden");
      const badge = $("run-badge");
      if (badge?.textContent === "running") badge.textContent = "idle";
      if (badge?.textContent === "complete") {
        window.setTimeout(() => {
          if ($("run-badge")?.textContent === "complete") $("run-badge").textContent = "idle";
        }, 2000);
      }
    }
  }

  function downloadConfig() {
    const blob = new Blob([JSON.stringify(buildConfig(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "qiup_run_config.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function loadConfigFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onerror = () => log("err", "failed to read config file");
    reader.onload = () => {
      try {
        const cfg = JSON.parse(String(reader.result || ""));
        if (!cfg || typeof cfg !== "object") throw new Error("config must be a JSON object");
        if (Array.isArray(cfg.profiles)) {
          state.profiles = PROFILE_IDS.filter((id) => cfg.profiles.includes(id));
        }
        if (cfg.modes && typeof cfg.modes === "object") {
          state.modes = {
            ideal: Boolean(cfg.modes.ideal),
            noisy: Boolean(cfg.modes.noisy),
          };
        }
        if (cfg.object && typeof cfg.object === "object") state.object = cfg.object;
        if (Number.isFinite(cfg.size)) state.size = cfg.size;
        if (Number.isFinite(cfg.seed)) state.seed = cfg.seed;
        if (cfg.ideal_setup) state.idealSetup = { ...state.idealSetup, ...cfg.ideal_setup };
        if (cfg.noisy_setup) state.noisySetup = { ...state.noisySetup, ...cfg.noisy_setup };
        if (cfg.ideal_latents) state.idealLatents = { ...state.idealLatents, ...cfg.ideal_latents };
        if (cfg.noisy_latents) state.noisyLatents = { ...state.noisyLatents, ...cfg.noisy_latents };
        if (cfg.thermal && typeof cfg.thermal === "object") {
          state.thermal = { ...state.thermal, ...cfg.thermal };
        }
        state.preset = "custom";
        renderSidebar();
        ensureTiles();
        schedulePreview();
        log("ok", `config loaded: ${file.name}`);
      } catch (err) {
        log("err", `config load failed: ${err.message || err}`);
      }
    };
    reader.readAsText(file);
  }

  function on(id, event, handler) {
    const el = $(id);
    if (!el) {
      console.warn(`QIUP UI: missing #${id}`);
      return;
    }
    el.addEventListener(event, handler);
  }

  function setTeamModal(open) {
    const modal = $("team-modal");
    if (!modal) return;
    modal.classList.toggle("hidden", !open);
  }

  function bindChrome() {
    on("btn-team", "click", () => setTeamModal(true));
    document.querySelectorAll("[data-team-close]").forEach((el) => {
      el.addEventListener("click", () => setTeamModal(false));
    });
    on("btn-run", "click", () => void runSimulation());
    on("btn-stop", "click", () => state.abort?.abort());
    on("btn-stage-ladder", "click", () => void runStageLadder());
    on("btn-mechanism-matrix", "click", () => void runDiagnosticsStream("mechanism_matrix"));
    on("btn-phase-step", "click", () => void runDiagnosticsStream("phase_step"));
    on("btn-fringe-curve", "click", () => void runDiagnosticsStream("fringe_curve", {
      fringe_points: 9,
    }));
    on("btn-sweep", "click", () => void runDiagnosticsStream("sweep", {
      sweep_param: $("sweep-param")?.value || "mean_photons_per_mode",
      sweep_points: 7,
    }));
    on("diag-profile", "change", () => renderDiagnostics());
    on("diag-cut", "change", () => renderDiagnostics());
    document.addEventListener("click", (e) => {
      const btn = e.target?.closest?.('[data-enlarge="diag"]');
      if (!btn || btn.disabled) return;
      e.preventDefault();
      const plot = btn.getAttribute("data-plot");
      const stageId = btn.getAttribute("data-stage-id");
      const profile = btn.getAttribute("data-profile");
      const mode = btn.getAttribute("data-mode");
      setEnlarge({
        type: "diag",
        plot,
        ...(stageId != null ? { stageId } : {}),
        ...(profile != null ? { profile } : {}),
        ...(mode != null ? { mode } : {}),
      });
    });
    on("btn-config", "click", downloadConfig);
    on("btn-config-load", "click", () => $("config-upload-input")?.click());
    on("config-upload-input", "change", (e) => {
      const file = e.target.files && e.target.files[0];
      loadConfigFile(file);
      e.target.value = "";
    });
    on("btn-reset", "click", () => {
      if (!state.defaults) return;
      resetFromDefaults(state.defaults);
      renderEnlarge();
      renderSidebar();
      ensureTiles();
      schedulePreview();
      log("info", "parameters reset");
    });
    on("btn-enlarge-close", "click", () => clearEnlarge());
    document.querySelector('[data-enlarge="preview"]')?.addEventListener("click", () => {
      setEnlarge({ type: "preview" });
    });
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void runSimulation();
      }
      if (e.key === "Escape") {
        if ($("team-modal") && !$("team-modal").classList.contains("hidden")) {
          setTeamModal(false);
          return;
        }
        clearEnlarge();
      }
    });
    let resizeTimer = null;
    window.addEventListener("resize", () => {
      updateViewportCss();
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        if (state.defaults) renderDiagnostics();
      }, 120);
    });
    updateViewportCss();
  }

  try {
    bindChrome();
    loadDefaults().catch((err) => {
      if (boot) boot.classList.add("hidden");
      if (errorBox) {
        errorBox.classList.remove("hidden");
        errorBox.textContent = err.message || String(err);
      }
      console.error(err);
    });
  } catch (err) {
    if (boot) boot.classList.add("hidden");
    if (errorBox) {
      errorBox.classList.remove("hidden");
      errorBox.textContent = err.message || String(err);
    }
    console.error(err);
  }
})();
