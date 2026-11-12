"""Generate QIUP A–D images using MNIST train sample index 5 as the SLM object.

Ideal: every listed disturbance is off so the camera sees the SLM intensity.
Noisy: mild remaining noise so the digit stays recognizable.
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from latents import LatentState
from setup_params import resolve_setup_selection
from simulate import simulate_forward

# This file lives in examples/; MNIST and package imports use the repo root.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "outputs" / "mnist_index5_four_arrangements"
DATA = ROOT / "data" / "MNIST" / "raw"
PROFILES = (
    ("A", "two_crystal_far_field", "NL1/NL2, T, far field"),
    ("B", "single_crystal_near_field", "NL, R, near field"),
    ("C", "single_crystal_far_field", "NL, R, far field"),
    ("D", "two_crystal_near_field", "NL1/NL2, T, near field"),
)
SIZE = 64
SEED = 15000
MNIST_INDEX = 5

# Shared lab geometry so the pump covers the SLM (otherwise only a Gaussian
# spot remains). These are not stochastic noise; they set the ideal transfer.
GEOMETRY = {
    "use_legacy_beam_match": False,
    "pump_waist_um": 350.0,
    "pixel_um": 6.5,
    "object_map_pixel_um": 6.5,
    "near_field_idler_magnification": 1.0,
    "near_field_signal_magnification": 1.0,
    "far_field_idler_relay_magnification": 1.0,
    "far_field_signal_relay_magnification": 1.0,
}

# Ideal: all disturbance sources off.
IDEAL_SETUP = {
    **GEOMETRY,
    "blur_model": "single_blur",
    "sigma_blur_extra_px": 0.0,
    "use_derived_diffraction": False,
    "sigma_diffraction_px": 0.0,
    "use_derived_aberration": False,
    "sigma_aberration_px": 0.0,
    "wavefront_rms_waves": 0.0,
    "delta_z_um": 0.0,
    "delta_M_rel": 0.0,
    "image_rotation_deg": 0.0,
    "slm_misreg_x_px": 0.0,
    "slm_misreg_y_px": 0.0,
    "enable_lens_rings": False,
    "enable_etalon_fringes": False,
    "couple_align_to_wavefront": False,
    "delta_phi_align_span_rad": 0.0,
    "prnu_sigma": 0.0,
    "read_noise0_e": 0.0,
    "read_noise_span_e": 0.0,
    "cic0": 0.0,
    "dark0": 0.0,
    "hot_frac_at_s1": 0.0,
    "accidental_e_at_s1": 0.0,
    "dT_NL1_C": 0.0,
    "dT_NL2_C": 0.0,
    "pump_waist_NL1_um": 350.0,
    "pump_waist_NL2_um": 350.0,
}
IDEAL_LATENTS = {
    "s_pump": 0.0,
    "s_gain": 0.0,
    "s_align": 0.0,
    "s_thermal": 0.0,
    "s_det": 0.0,
    "s_cool": 0.0,
    "s_focus": 0.0,
    "s_slm": 0.0,
    "delta_phi_rms": 0.0,
    "mean_photons_per_mode": 1.0,
    "num_frames": 1,
    "exposure_s": 0.02,
    "em_gain": 1.0,
}

# Noisy: same geometry; mild remaining noise so the digit stays visible.
NOISY_SETUP = {
    **GEOMETRY,
    "blur_model": "single_blur",
    "sigma_blur_extra_px": 0.35,
    "use_derived_diffraction": False,
    "sigma_diffraction_px": 0.25,
    "use_derived_aberration": False,
    "sigma_aberration_px": 0.20,
    "wavefront_rms_waves": 0.0,
    "delta_z_um": 0.0,
    "delta_M_rel": 0.0,
    "image_rotation_deg": 0.0,
    "slm_misreg_x_px": 0.0,
    "slm_misreg_y_px": 0.0,
    "enable_lens_rings": False,
    "enable_etalon_fringes": False,
    "couple_align_to_wavefront": False,
    "delta_phi_align_span_rad": 0.08,
    "prnu_sigma": 0.012,
    "read_noise0_e": 4.0,
    "read_noise_span_e": 8.0,
    "cic0": 0.002,
    "dark0": 0.0008,
    "hot_frac_at_s1": 0.0004,
    "accidental_e_at_s1": 0.004,
    "dT_NL1_C": 0.0,
    "dT_NL2_C": 0.0,
    "pump_waist_NL1_um": 350.0,
    "pump_waist_NL2_um": 350.0,
}
NOISY_LATENTS = {
    "s_pump": 0.04,
    "s_gain": 0.04,
    "s_align": 0.06,
    "s_thermal": 0.04,
    "s_det": 0.10,
    "s_cool": 0.08,
    "s_focus": 0.04,
    "s_slm": 0.05,
    "delta_phi_rms": 0.04,
    "mean_photons_per_mode": 2.0,
    "num_frames": 1,
    "exposure_s": 0.05,
    "em_gain": 20.0,
}


def _allowed_overrides(setup, raw: dict) -> dict:
    allowed = set(setup.to_dict())
    out = {key: value for key, value in raw.items() if key in allowed}
    two = str(getattr(setup, "spdc_architecture", "")).startswith("two")
    if two:
        if "pump_waist_NL1_um" in raw:
            out["pump_waist_NL1_um"] = raw["pump_waist_NL1_um"]
        if "pump_waist_NL2_um" in raw:
            out["pump_waist_NL2_um"] = raw["pump_waist_NL2_um"]
    else:
        out.pop("pump_waist_NL2_um", None)
        out.pop("far_field_idler_relay_magnification", None)
        out.pop("far_field_signal_relay_magnification", None)
        if str(getattr(setup, "imaging_mode", "")) == "far_field":
            out.pop("near_field_idler_magnification", None)
            out.pop("near_field_signal_magnification", None)
        else:
            out.pop("far_field_idler_relay_magnification", None)
            out.pop("far_field_signal_relay_magnification", None)
    if str(getattr(setup, "imaging_mode", "")) == "far_field":
        out.pop("near_field_idler_magnification", None)
        out.pop("near_field_signal_magnification", None)
    else:
        out.pop("far_field_idler_relay_magnification", None)
        out.pop("far_field_signal_relay_magnification", None)
    return out


def _save_png(path: Path, image: np.ndarray, title: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    im = ax.imshow(np.asarray(image, dtype=np.float64), cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_array(path: Path, image: np.ndarray) -> None:
    np.save(path, np.asarray(image, dtype=np.float32))


def _open_mnist(path: Path):
    """Open an MNIST idx file; supports plain or .gz."""
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def mnist_train_count() -> int:
    """Number of shipped train samples (header of local idx labels file)."""
    labels_plain = DATA / "train-labels-idx1-ubyte"
    labels_gz = DATA / "train-labels-idx1-ubyte.gz"
    path = labels_plain if labels_plain.is_file() else labels_gz
    if not path.is_file():
        raise FileNotFoundError(
            f"MNIST labels missing at {labels_plain}. "
            "Ship data/MNIST/raw or rebuild with tools/build_mnist_subset.py"
        )
    with _open_mnist(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
    if magic != 2049:
        raise RuntimeError(f"unexpected MNIST label magic {magic}")
    return int(count)


def _ensure_mnist_files() -> tuple[Path, Path]:
    """Resolve train images/labels under data/MNIST/raw (repo-root canonical store)."""
    images_plain = DATA / "train-images-idx3-ubyte"
    labels_plain = DATA / "train-labels-idx1-ubyte"
    if images_plain.is_file() and labels_plain.is_file() and images_plain.stat().st_size > 1000:
        return images_plain, labels_plain
    images_gz = DATA / "train-images-idx3-ubyte.gz"
    labels_gz = DATA / "train-labels-idx1-ubyte.gz"
    if images_gz.is_file() and labels_gz.is_file() and images_gz.stat().st_size > 1000:
        return images_gz, labels_gz
    raise FileNotFoundError(
        f"MNIST train idx files missing under {DATA}. "
        "Expected the shipped subset (train-*-idx*-ubyte). "
        "Rebuild with: python tools/build_mnist_subset.py --n 100"
    )


def load_mnist_sample(index: int) -> tuple[np.ndarray, int, str]:
    try:
        from torchvision import datasets

        ds = datasets.MNIST(root=str(ROOT / "data"), train=True, download=False)
        image = np.array(ds[index][0], dtype=np.float32)
        if image.max() <= 1.0 + 1e-6:
            image = image * 255.0
        return image, int(ds[index][1]), "mnist_local_torchvision"
    except Exception:
        pass

    images_path, labels_path = _ensure_mnist_files()
    with _open_mnist(images_path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise RuntimeError(f"unexpected MNIST image magic {magic}")
        if not 0 <= index < count:
            raise IndexError(f"mnist_index={index} outside 0..{count - 1}")
        handle.seek(16 + index * rows * cols)
        image = np.frombuffer(handle.read(rows * cols), dtype=np.uint8).reshape(rows, cols)
    with _open_mnist(labels_path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise RuntimeError(f"unexpected MNIST label magic {magic}")
        handle.seek(8 + index)
        label = int.from_bytes(handle.read(1), "big")
    return np.asarray(image, dtype=np.float32), label, "mnist_idx_files"


def mnist_reference(index: int, size: int) -> tuple[dict, np.ndarray, int, str]:
    raw, label, source = load_mnist_sample(index)
    native = raw.astype(np.float32) / 255.0
    scale = float(size) / float(raw.shape[0])
    T = np.clip(zoom(native, scale, order=1), 0.0, 1.0).astype(np.float32)
    phi = np.zeros_like(T, dtype=np.float32)
    ref = {
        "M": T.copy(),
        "T": T,
        "phi": phi,
        "O_i": np.sqrt(T).astype(np.float32),
        "D": np.zeros_like(T, dtype=np.float32),
        "target_size": size,
        "label": str(label),
        "mnist_index": index,
        "object_source": source,
    }
    return ref, raw, label, source


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref, raw, digit_label, source = mnist_reference(MNIST_INDEX, SIZE)
    slm = np.asarray(ref["T"], dtype=np.float64)
    _save_png(OUT / "mnist_index5_native.png", raw / 255.0, f"MNIST train[{MNIST_INDEX}] label={digit_label}", 0.0, 1.0)
    _save_png(OUT / "mnist_index5_slm_object.png", slm, f"SLM object T  |  index {MNIST_INDEX}, label {digit_label}", 0.0, 1.0)
    _save_array(OUT / "mnist_index5_native.npy", raw)
    _save_array(OUT / "mnist_index5_slm_object.npy", slm)
    print(f"MNIST train[{MNIST_INDEX}] label={digit_label} source={source}")

    summary: dict[str, object] = {
        "mnist_index": MNIST_INDEX,
        "mnist_label": digit_label,
        "object_source": source,
        "object_encoding": "intensity T = MNIST/255, phase = 0",
        "grid": SIZE,
        "seed": SEED,
        "ideal_latents": IDEAL_LATENTS,
        "noisy_latents": NOISY_LATENTS,
        "ideal_setup": IDEAL_SETUP,
        "noisy_setup": NOISY_SETUP,
        "ideal_detector_noise": False,
        "noisy_detector_noise": True,
        "profiles": {},
    }
    images: dict[str, dict[str, np.ndarray]] = {}

    for selector, slug, label in PROFILES:
        base = resolve_setup_selection(selector)
        setup_ideal = base.overridden(**_allowed_overrides(base, IDEAL_SETUP))
        setup_noisy = base.overridden(**_allowed_overrides(base, NOISY_SETUP))
        ideal = simulate_forward(
            ref,
            LatentState(setup=setup_ideal, **IDEAL_LATENTS),
            seed=SEED,
            detector_noise=False,
        )
        noisy = simulate_forward(
            ref,
            LatentState(setup=setup_noisy, **NOISY_LATENTS),
            seed=SEED,
            detector_noise=True,
        )
        ideal_img = np.asarray(ideal.ideal_signal_image, dtype=np.float64)
        noisy_img = np.asarray(noisy.detector_signal_image, dtype=np.float64)
        images[selector] = {"ideal": ideal_img, "noisy": noisy_img}

        prefix = f"{selector}_{slug}"
        _save_png(OUT / f"{prefix}_ideal_signal.png", ideal_img, f"{selector} ideal  |  {label}")
        _save_png(OUT / f"{prefix}_noisy_detector.png", noisy_img, f"{selector} noisy  |  {label}")
        _save_array(OUT / f"{prefix}_ideal_signal.npy", ideal_img)
        _save_array(OUT / f"{prefix}_noisy_detector.npy", noisy_img)
        summary["profiles"][selector] = {
            "slug": slug,
            "label": label,
            "ideal_min": float(np.min(ideal_img)),
            "ideal_max": float(np.max(ideal_img)),
            "noisy_min": float(np.min(noisy_img)),
            "noisy_max": float(np.max(noisy_img)),
            "ideal_sha256": ideal.meta.get("ideal_signal_image_sha256"),
            "noisy_sha256": noisy.meta.get("detector_signal_image_sha256"),
        }
        print(f"{selector}: ideal max={np.max(ideal_img):.4g}  noisy max={np.max(noisy_img):.4g}")

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.2), constrained_layout=True)
    for row, (selector, slug, label) in enumerate(PROFILES):
        panels = (
            (slm, f"{selector} SLM T", 0.0, 1.0),
            (images[selector]["ideal"], f"{selector} ideal", None, None),
            (images[selector]["noisy"], f"{selector} noisy", None, None),
        )
        for col, (data, title, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[row, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(
        f"QIUP MNIST train[{MNIST_INDEX}] label={digit_label} — all-noise-off vs mild noise",
        fontsize=12,
    )
    fig.savefig(OUT / "comparison_A_B_C_D_ideal_vs_noisy.png", dpi=170)
    plt.close(fig)

    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
