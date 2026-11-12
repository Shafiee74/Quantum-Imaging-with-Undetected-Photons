"""Generate mask-5 images for QIUP arrangements A–D, ideal and noisy."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from repo_paths import ensure_import_paths
ensure_import_paths()
from experiment_runner import default_experiment_reference
from latents import LatentState
from setup_params import object_mask_intensity, resolve_setup_selection
from simulate import simulate_forward

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "examples" / "outputs" / "mask5_four_arrangements"
PROFILES = (
    ("A", "two_crystal_far_field", "NL1/NL2, T, far field"),
    ("B", "single_crystal_near_field", "NL, R, near field"),
    ("C", "single_crystal_far_field", "NL, R, far field"),
    ("D", "two_crystal_near_field", "NL1/NL2, T, near field"),
)
SIZE = 48
SEED = 15000
MASK_ID = 5


def _zero_instability() -> dict[str, float]:
    return {
        "s_pump": 0.0,
        "s_gain": 0.0,
        "s_align": 0.0,
        "s_thermal": 0.0,
        "s_det": 0.0,
        "s_cool": 0.0,
        "s_focus": 0.0,
        "s_slm": 0.0,
        "delta_phi_rms": 0.0,
    }


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = default_experiment_reference(SIZE)
    mask = object_mask_intensity(MASK_ID, (SIZE, SIZE))
    _save_png(OUT / "mask_5_catalog.png", mask, "Catalog mask 5 (top half = 1)")
    _save_array(OUT / "mask_5_catalog.npy", mask)

    summary: dict[str, object] = {
        "mask_id": MASK_ID,
        "mask_definition": "yy < ny/2  (top half open, bottom half blocked)",
        "grid": SIZE,
        "seed": SEED,
        "nbar": 0.1,
        "profiles": {},
    }
    images: dict[str, dict[str, np.ndarray]] = {}

    for selector, slug, label in PROFILES:
        setup = resolve_setup_selection(selector).overridden(
            object_mode="mask",
            object_mask_id=MASK_ID,
            object_device="binary_mask",
        )
        common = dict(
            setup=setup,
            mean_photons_per_mode=0.1,
            num_frames=1,
            exposure_s=0.02,
            em_gain=100.0,
        )
        ideal_state = LatentState(**common, **_zero_instability())
        noisy_state = LatentState(**common)
        ideal = simulate_forward(ref, ideal_state, seed=SEED, detector_noise=False)
        noisy = simulate_forward(ref, noisy_state, seed=SEED, detector_noise=True)

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
            "object_type": setup.object_type,
            "object_device": setup.object_device,
            "object_plane": setup.object_plane,
            "ideal_min": float(np.min(ideal_img)),
            "ideal_max": float(np.max(ideal_img)),
            "noisy_min": float(np.min(noisy_img)),
            "noisy_max": float(np.max(noisy_img)),
            "ideal_sha256": ideal.meta.get("ideal_signal_image_sha256"),
            "noisy_sha256": noisy.meta.get("detector_signal_image_sha256"),
            "detector_noise_ideal": bool(ideal.meta.get("detector_noise")),
            "detector_noise_noisy": bool(noisy.meta.get("detector_noise")),
        }
        print(f"{selector}: ideal max={np.max(ideal_img):.4g}  noisy max={np.max(noisy_img):.4g}")

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.2), constrained_layout=True)
    for row, (selector, slug, label) in enumerate(PROFILES):
        panels = (
            (mask, f"{selector} mask 5", 0.0, 1.0),
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
    fig.suptitle("QIUP mask 5 — four arrangements, ideal vs detector noise", fontsize=12)
    fig.savefig(OUT / "comparison_A_B_C_D_ideal_vs_noisy.png", dpi=170)
    plt.close(fig)

    (OUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
