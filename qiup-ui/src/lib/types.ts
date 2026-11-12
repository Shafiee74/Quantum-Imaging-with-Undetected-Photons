export type ProfileId = "A" | "B" | "C" | "D";
export type Mode = "ideal" | "noisy";

export interface ProfileInfo {
  id: ProfileId;
  slug: string;
  label: string;
  architecture: "two_crystal" | "single_crystal_double_pass";
  imaging_mode: "near_field" | "far_field";
  crystals: string[];
  object_coefficient: "T" | "R";
  object_type: string;
  object_plane: string;
  thermal: Record<string, number>;
  lambda_nm: { pump: number; signal: number; idler: number };
}

export type ParamValue = number | boolean | string;
export type ParamDict = Record<string, ParamValue>;

export interface Defaults {
  package_version: string;
  release_classification?: string;
  calibration_status?: string;
  profiles: Record<ProfileId, ProfileInfo>;
  ideal_setup: ParamDict;
  ideal_latents: ParamDict;
  noisy_setup: ParamDict;
  noisy_latents: ParamDict;
  size: number;
  seed: number;
  mnist_index: number;
  size_range: [number, number];
  mnist_train_count: number;
}

export interface ImagePayload {
  png: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  shape: number[];
}

export interface ThermalRow {
  crystal: string;
  T_pm0_C: number | null;
  T_pm_C: number | null;
  T_crystal_C: number | null;
  dT_C: number | null;
}

export type ObjectSpec =
  | { source: "mnist"; mnist_index: number }
  | { source: "mask"; mask_id: number };

export interface RunConfig {
  profiles: ProfileId[];
  modes: { ideal: boolean; noisy: boolean };
  object: ObjectSpec;
  size: number;
  seed: number;
  thermal: Partial<Record<ProfileId, Record<string, number>>>;
  ideal_setup: ParamDict;
  ideal_latents: ParamDict;
  noisy_setup: ParamDict;
  noisy_latents: ParamDict;
}

export interface ObjectEvent {
  type: "object";
  source: "mnist" | "mask";
  mnist_index?: number;
  mask_id?: number;
  label: number | string;
  origin?: string;
  image: ImagePayload;
}

export interface ResultEvent {
  type: "result";
  profile: ProfileId;
  mode: Mode;
  image: ImagePayload;
  elapsed_s: number;
  sha256?: string | null;
  thermal: ThermalRow[];
}

export type RunEvent =
  | { type: "start"; total: number; profiles: ProfileId[]; modes: Mode[]; output_dir: string }
  | ObjectEvent
  | { type: "progress"; profile: ProfileId; mode: Mode }
  | ResultEvent
  | { type: "error"; profile?: ProfileId; mode?: Mode; message: string }
  | { type: "done"; elapsed_s: number; output_dir: string };
