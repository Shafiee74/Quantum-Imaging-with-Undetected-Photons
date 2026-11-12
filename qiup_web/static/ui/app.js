(() => {
  const PROFILE_IDS = ["A", "B", "C", "D"];
  const PRESET_SCALE = { off: 0, mild: 0.5, moderate: 1, strong: 2 };

  const GEOMETRY = [
    { key: "pump_waist_um", label: "Pump waist w₀", unit: "µm", min: 100, max: 350, step: 5, target: "setup",
      hint: "Gaussian pump envelope. Larger waist → flatter illumination of the SLM object." },
    { key: "pixel_um", label: "Camera pixel", unit: "µm", min: 6.5, max: 24, step: 0.5, target: "setup",
      hint: "Detector pixel pitch (spec range 6.5–24 µm)." },
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
    { key: "num_frames", label: "Frames averaged", min: 1, max: 64, step: 1, target: "latents", integer: true },
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
        { key: "s_thermal", label: "Thermal", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "s_focus", label: "Focus", min: 0, max: 1, step: 0.01, target: "latents" },
        { key: "s_slm", label: "SLM", min: 0, max: 1, step: 0.01, target: "latents" },
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
        { key: "delta_phi_align_span_rad", label: "Align phase span", unit: "rad", min: 0, max: 1, step: 0.01, target: "setup" },
        { key: "image_rotation_deg", label: "Image rotation", unit: "°", min: -5, max: 5, step: 0.1, target: "setup" },
        { key: "slm_misreg_x_px", label: "SLM misreg. x", unit: "px", min: -4, max: 4, step: 0.1, target: "setup" },
        { key: "slm_misreg_y_px", label: "SLM misreg. y", unit: "px", min: -4, max: 4, step: 0.1, target: "setup" },
      ],
      toggles: [
        { key: "enable_lens_rings", label: "Lens rings", target: "setup" },
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
        { key: "accidental_e_at_s1", label: "Accidentals", unit: "e⁻", min: 0, max: 0.05, step: 0.0005, target: "setup" },
      ],
    },
  ];

  const NOISE_KEYS = NOISE_GROUPS.flatMap((g) => g.numeric);

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
    collapsed: {},
    abort: null,
    running: false,
    previewTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const boot = $("boot");
  const errorBox = $("error");
  const workspace = $("workspace");
  const logEl = $("log");
  const sidebar = $("sidebar");

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

  function setImage(box, image, fallback) {
    if (!image?.png) {
      box.textContent = fallback || "—";
      return;
    }
    box.innerHTML = `<img alt="result" src="${pngSrc(image)}" />`;
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
    const open = !state.collapsed[id];
    const right = extras.right || "";
    return `
      <div class="card" data-panel="${id}">
        <div class="panel-head" data-toggle="${id}">
          <div>
            <h2>${escapeHtml(title)}</h2>
            <p class="panel-sub">${escapeHtml(subtitle)}</p>
          </div>
          <div style="display:flex;align-items:center;gap:0.4rem">
            ${right}
            <span class="chevron ${open ? "open" : ""}">▸</span>
          </div>
        </div>
        <div class="panel-body ${open ? "" : "collapsed"}" data-body="${id}">
          ${bodyHtml}
        </div>
      </div>`;
  }

  function sliderHtml(p, value, accentClass = "") {
    const v = num(value, p.min);
    const unit = p.unit ? ` ${p.unit}` : "";
    return `
      <div class="slider-row ${accentClass}" data-slider="${p.key}" data-target="${p.target || "setup"}">
        <div class="slider-top">
          <strong>${escapeHtml(p.label)}</strong>
          <span class="val" data-val>${formatValue(v, p.step)}${escapeHtml(unit)}</span>
        </div>
        <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}"
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

    const arrangements = `
      <div style="display:flex;justify-content:flex-end;margin-bottom:0.45rem">
        <button type="button" class="btn tiny ghost" data-action="toggle-all-profiles">
          ${state.profiles.length === 4 ? "Clear" : "All"}
        </button>
      </div>
      <div class="arr-grid">
        ${PROFILE_IDS.map((id) => {
          const info = d.profiles[id];
          const selected = state.profiles.includes(id);
          return `
            <button type="button" class="arr-card ${selected ? "selected" : ""}" data-profile="${id}">
              <strong>${id}</strong>
              <span>${escapeHtml(info.label)}</span>
              <span>${escapeHtml((info.imaging_mode || "").replaceAll("_", " "))}</span>
            </button>`;
        }).join("")}
      </div>`;

    const objectBody = `
      <div class="seg">
        <button type="button" data-object-source="mnist" class="${state.object.source === "mnist" ? "active" : ""}">MNIST digit</button>
        <button type="button" data-object-source="mask" class="${state.object.source === "mask" ? "active" : ""}">Catalog mask</button>
      </div>
      ${state.object.source === "mnist" ? `
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
        </div>` : `
        <div class="mask-grid">
          ${Array.from({ length: 10 }, (_, i) => i + 1).map((id) => `
            <button type="button" data-mask-id="${id}" class="${state.object.mask_id === id ? "active" : ""}">${id}</button>
          `).join("")}
        </div>`}`;

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
        <span class="badge violet">Far field · A, C</span>
        ${MAGNIFICATION.far_field.map((p) => sliderHtml(p, state.idealSetup[p.key] ?? state.noisySetup[p.key])).join("")}
      </div>`;

    const sourceLatents = state.sourceMode === "ideal" ? state.idealLatents : state.noisyLatents;
    const sourceAccent = state.sourceMode === "ideal" ? "green" : "rose";
    const sourceBody = `
      <div class="seg">
        <button type="button" data-source-mode="ideal" class="${state.sourceMode === "ideal" ? "active" : ""}">Ideal run</button>
        <button type="button" data-source-mode="noisy" class="${state.sourceMode === "noisy" ? "active" : ""}">Noisy run</button>
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
                    <input type="number" step="0.1" data-thermal="${id}" data-tkey="T_pm0_${crystal}_C"
                      value="${num(state.thermal[id]?.[`T_pm0_${crystal}_C`])}" />
                  </label>
                  <label class="field">ΔT (°C)
                    <input type="number" step="0.05" data-thermal="${id}" data-tkey="dT_${crystal}_C"
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
            toggleHtml(t.key, t.label, Boolean(state.noisySetup[t.key]))
          ).join("")}
        </div>
      `).join("")}
      ${toggleHtml(
        "blur_model",
        "SPDC correlation PSF (double blur)",
        state.noisySetup.blur_model === "double_blur",
        "Physical two-photon blur; washes out fine detail",
      )}`;

    sidebar.innerHTML = [
      panel("arrangements", "Arrangements", "Architecture × imaging mode", arrangements),
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

    $("m-preset").textContent = state.preset;
    bindSidebar();
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
    state.idealSetup = { ...state.idealSetup, [key]: value };
    state.noisySetup = { ...state.noisySetup, [key]: value };
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
      $("m-preset").textContent = "custom";
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
  }

  function schedulePreview() {
    if (state.previewTimer) clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(() => {
      state.previewTimer = null;
      void refreshPreview();
    }, 450);
  }

  function bindSidebar() {
    sidebar.querySelectorAll("[data-toggle]").forEach((head) => {
      head.addEventListener("click", (e) => {
        if (e.target.closest("[data-action]")) return;
        const id = head.getAttribute("data-toggle");
        state.collapsed[id] = !state.collapsed[id];
        const body = sidebar.querySelector(`[data-body="${id}"]`);
        const chev = head.querySelector(".chevron");
        body?.classList.toggle("collapsed", !!state.collapsed[id]);
        chev?.classList.toggle("open", !state.collapsed[id]);
      });
    });

    sidebar.querySelectorAll("[data-profile]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-profile");
        if (state.profiles.includes(id)) {
          state.profiles = state.profiles.filter((p) => p !== id);
        } else {
          state.profiles = PROFILE_IDS.filter((p) => p === id || state.profiles.includes(p));
        }
        renderSidebar();
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
        } else {
          state.object = { source: "mask", mask_id: state.object.mask_id || 5 };
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
          if (noiseParam.target === "setup") state.noisySetup = { ...state.noisySetup, [key]: value };
          else state.noisyLatents = { ...state.noisyLatents, [key]: value };
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
        else if (key === "mode_noisy") state.modes = { ...state.modes, noisy: checked };
        else if (key === "blur_model") {
          state.noisySetup = { ...state.noisySetup, blur_model: checked ? "double_blur" : "single_blur" };
          markCustomPreset();
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
        return `
          <div class="card tile" id="tile-${id}-${mode}">
            <div class="image-head">
              <strong>${id} · ${escapeHtml(info.label)}</strong>
              <span class="muted">${escapeHtml((info.imaging_mode || "").replaceAll("_", " "))}</span>
            </div>
            <div class="image-box" data-box>idle</div>
            <div class="stats" data-stats></div>
          </div>`;
      }).join("");
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
      setImage(box, payload.image);
      const img = payload.image;
      const elapsed = payload.elapsed_s != null ? `${Number(payload.elapsed_s).toFixed(2)} s` : "";
      stats.textContent = `mean=${img.mean?.toFixed?.(4) ?? img.mean} · ${elapsed}`;
    }
  }

  async function loadDefaults() {
    const res = await fetch("/api/defaults");
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    state.defaults = data;
    resetFromDefaults(data);
    $("version-badge").textContent = `v${data.package_version}`;
    $("backend-pill").textContent = `backend · ${data.backend?.framework || "django"}`;
    if (data.calibration_note) $("calib-line").textContent = data.calibration_note;
    renderSidebar();
    ensureTiles();
    boot.classList.add("hidden");
    workspace.classList.remove("hidden");
    await refreshPreview();
  }

  async function refreshPreview() {
    const body = { object: state.object, size: state.size };
    $("preview-box").textContent = "loading…";
    try {
      const res = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const ev = await res.json();
      if (!res.ok || ev.error) throw new Error(ev.error || ev.message || `HTTP ${res.status}`);
      setImage($("preview-box"), ev.image, "no image");
      if (ev.source === "mnist") {
        $("preview-title").textContent = `MNIST #${ev.mnist_index}`;
        $("preview-sub").textContent = `digit label ${ev.label}`;
      } else {
        $("preview-title").textContent = String(ev.label || "mask");
        $("preview-sub").textContent = "catalog mask";
      }
    } catch (err) {
      $("preview-box").textContent = err.message;
      log("err", `preview: ${err.message}`);
    }
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "start":
        $("m-progress").textContent = `0 / ${ev.total}`;
        $("output-dir").textContent = ev.output_dir || "";
        $("output-dir").classList.toggle("hidden", !ev.output_dir);
        $("run-badge").textContent = "running";
        log("info", `run started: ${(ev.profiles || []).join(", ")}`);
        break;
      case "object":
        setImage($("preview-box"), ev.image);
        log("info", `SLM object ready (${ev.source})`);
        break;
      case "progress":
        setTile(ev.profile, ev.mode, "running");
        break;
      case "result":
        setTile(ev.profile, ev.mode, "done", ev);
        {
          const parts = $("m-progress").textContent.split("/");
          const done = Number(parts[0]?.trim()) || 0;
          const total = Number(parts[1]?.trim()) || 0;
          const next = done + 1;
          $("m-progress").textContent = `${next} / ${total || next}`;
          $("progress-fill").style.width = total ? `${Math.round((next / total) * 100)}%` : "100%";
        }
        log("ok", `${ev.profile} ${ev.mode} done in ${Number(ev.elapsed_s).toFixed(2)} s`);
        break;
      case "error":
        if (ev.profile && ev.mode) setTile(ev.profile, ev.mode, "error", ev);
        log("err", ev.message || "error");
        break;
      case "done":
        $("m-elapsed").textContent = `${Number(ev.elapsed_s).toFixed(1)} s`;
        $("run-badge").textContent = "complete";
        $("output-dir").textContent = ev.output_dir || "";
        $("output-dir").classList.toggle("hidden", !ev.output_dir);
        log("ok", `finished in ${Number(ev.elapsed_s).toFixed(2)} s`);
        break;
      default:
        break;
    }
  }

  async function runSimulation() {
    if (state.running) return;
    const config = buildConfig();
    if (!config.profiles.length || (!config.modes.ideal && !config.modes.noisy)) {
      log("err", "Select at least one arrangement and one mode.");
      return;
    }
    state.running = true;
    state.abort = new AbortController();
    $("btn-run").classList.add("hidden");
    $("btn-stop").classList.remove("hidden");
    $("progress-fill").style.width = "0%";
    $("m-profiles").textContent = config.profiles.join(" · ");
    $("m-elapsed").textContent = "…";
    $("run-badge").textContent = "running";
    for (const p of PROFILE_IDS) {
      for (const mode of ["ideal", "noisy"]) {
        const active = config.profiles.includes(p) && config.modes[mode];
        setTile(p, mode, active ? "queued" : "idle");
      }
    }
    try {
      const res = await fetch("/api/run", {
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
    } catch (err) {
      if (err.name === "AbortError") log("err", "run cancelled");
      else log("err", err.message || String(err));
    } finally {
      state.running = false;
      state.abort = null;
      $("btn-run").classList.remove("hidden");
      $("btn-stop").classList.add("hidden");
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

  function on(id, event, handler) {
    const el = $(id);
    if (!el) {
      console.warn(`QIUP UI: missing #${id}`);
      return;
    }
    el.addEventListener(event, handler);
  }

  function bindChrome() {
    on("btn-run", "click", () => void runSimulation());
    on("btn-stop", "click", () => state.abort?.abort());
    on("btn-config", "click", downloadConfig);
    on("btn-reset", "click", () => {
      if (!state.defaults) return;
      resetFromDefaults(state.defaults);
      renderSidebar();
      ensureTiles();
      schedulePreview();
      log("info", "parameters reset");
    });
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void runSimulation();
      }
    });
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
