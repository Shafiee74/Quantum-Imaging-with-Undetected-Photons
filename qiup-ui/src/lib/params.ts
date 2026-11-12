export type Target = "setup" | "latents";

export interface NumericParam {
  key: string;
  label: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  target: Target;
  hint?: string;
  integer?: boolean;
}

export interface BoolParam {
  key: string;
  label: string;
  target: Target;
  hint?: string;
}

export interface ParamGroup {
  id: string;
  title: string;
  description: string;
  numeric: NumericParam[];
  toggles?: BoolParam[];
}

export const GEOMETRY: NumericParam[] = [
  { key: "pump_waist_um", label: "Pump waist w₀", unit: "µm", min: 100, max: 350, step: 5, target: "setup",
    hint: "Gaussian pump envelope. Larger waist → flatter illumination of the SLM object." },
  { key: "pixel_um", label: "Camera pixel", unit: "µm", min: 6.5, max: 24, step: 0.5, target: "setup",
    hint: "Detector pixel pitch (spec range 6.5–24 µm)." },
  { key: "object_map_pixel_um", label: "SLM map pixel", unit: "µm", min: 2, max: 24, step: 0.5, target: "setup",
    hint: "Physical size of one object-map pixel on the SLM." },
];

export const MAGNIFICATION: Record<"near_field" | "far_field", NumericParam[]> = {
  near_field: [
    { key: "near_field_idler_magnification", label: "Idler magnification", min: 0.25, max: 4, step: 0.05, target: "setup" },
    { key: "near_field_signal_magnification", label: "Signal magnification", min: 0.25, max: 4, step: 0.05, target: "setup" },
  ],
  far_field: [
    { key: "far_field_idler_relay_magnification", label: "Idler relay mag.", min: 0.25, max: 4, step: 0.05, target: "setup" },
    { key: "far_field_signal_relay_magnification", label: "Signal relay mag.", min: 0.25, max: 4, step: 0.05, target: "setup" },
  ],
};

export const SOURCE: NumericParam[] = [
  { key: "mean_photons_per_mode", label: "Mean photons / mode n̄", min: 0.1, max: 20, step: 0.1, target: "latents",
    hint: "SPDC brightness. Higher n̄ raises signal above the detector floor." },
  { key: "exposure_s", label: "Exposure", unit: "s", min: 0.005, max: 1, step: 0.005, target: "latents" },
  { key: "em_gain", label: "EM gain", min: 1, max: 1000, step: 1, target: "latents", integer: true,
    hint: "EMCCD multiplication gain (≥ 1)." },
  { key: "num_frames", label: "Frames averaged", min: 1, max: 64, step: 1, target: "latents", integer: true },
];

export const NOISE_GROUPS: ParamGroup[] = [
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

export const NOISE_KEYS = NOISE_GROUPS.flatMap((g) => g.numeric);

export const MNIST_LABEL_HINT = "MNIST train index into the shipped subset. The digit label is not the index.";

export function formatValue(value: number, step: number): string {
  if (!Number.isFinite(value)) return "—";
  const decimals = step >= 1 ? 0 : Math.min(4, Math.max(0, Math.ceil(-Math.log10(step))));
  return value.toFixed(decimals);
}

export function formatStat(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(2);
  return value.toFixed(abs >= 100 ? 1 : 3);
}
