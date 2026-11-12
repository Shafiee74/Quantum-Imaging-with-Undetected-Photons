"""Grid-level specifications with fixed physical domain."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from setup_params import resolve_setup_selection, resolved_spatial_mapping
from verification.convergence.definitions import (
    ADDITIONAL_LEVEL_FACTORS,
    GRID_LEVEL_FACTORS,
    N_MID_DEFAULT,
)


@dataclass(frozen=True)
class GridLevel:
    level_id: str
    profile: str
    n: int
    pixel_um: float
    domain_um: float
    factor: float
    stock_object_pixel_um: float
    is_numerical_reference: bool

    @property
    def physical_fov_sampling_window_um(self) -> float:
        """Array sampling window (= domain). Not the pump 1/e² FoV."""
        return float(self.domain_um)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["physical_sampling_window_um"] = self.physical_fov_sampling_window_um
        d["note"] = (
            "domain = N*pixel_um held fixed across levels; "
            "pump 1/e^2 FoV is independent (Task 09)"
        )
        return d


def stock_object_pixel_um(profile: str) -> float:
    setup = resolve_setup_selection(profile)
    sm = resolved_spatial_mapping(setup)
    return float(sm["object_plane_pixel_um"])


def domain_um_for_profile(profile: str, *, n_mid: int = N_MID_DEFAULT) -> float:
    return float(n_mid) * stock_object_pixel_um(profile)


def build_grid_levels(
    profile: str,
    *,
    n_mid: int = N_MID_DEFAULT,
    include_additional: bool = True,
) -> list[GridLevel]:
    stock = stock_object_pixel_um(profile)
    domain = float(n_mid) * stock
    levels: list[GridLevel] = []
    factors = list(GRID_LEVEL_FACTORS)
    if include_additional:
        factors = factors + list(ADDITIONAL_LEVEL_FACTORS)
    # Identify finest primary level (max factor among GRID_LEVEL_FACTORS)
    max_primary = max(f for _, f in GRID_LEVEL_FACTORS)
    for level_id, factor in factors:
        n = int(round(n_mid * float(factor)))
        if n < 8:
            raise ValueError(f"grid n too small: {n}")
        # Exact domain preservation: dx = domain / n
        dx = domain / float(n)
        is_ref = (
            level_id.startswith("grid_level_")
            and abs(float(factor) - max_primary) < 1e-15
        )
        levels.append(
            GridLevel(
                level_id=level_id,
                profile=profile,
                n=n,
                pixel_um=float(dx),
                domain_um=float(domain),
                factor=float(factor),
                stock_object_pixel_um=stock,
                is_numerical_reference=bool(is_ref),
            )
        )
    return levels


def build_explicit_n_ladder(
    profile: str,
    ns: tuple[int, ...] | list[int],
    *,
    n_mid: int = N_MID_DEFAULT,
    numerical_reference_n: int | None = None,
) -> list[GridLevel]:
    """Fixed-FoV ladder at explicit grid sizes ``ns``.

    Domain ``L = n_mid * stock_object_pixel_um`` is held constant; each level
    uses ``pixel_um = L / N``. The finest listed ``N`` (or
    ``numerical_reference_n``) is marked as the numerical reference — used only
    for vs-finest diagnostics, not as a substitute for successive-grid checks.
    """
    stock = stock_object_pixel_um(profile)
    domain = float(n_mid) * stock
    n_list = [int(n) for n in ns]
    if any(n < 8 for n in n_list):
        raise ValueError(f"grid n too small in {n_list}")
    ref_n = int(numerical_reference_n) if numerical_reference_n is not None else max(n_list)
    levels: list[GridLevel] = []
    for n in n_list:
        dx = domain / float(n)
        levels.append(
            GridLevel(
                level_id=f"N{n}",
                profile=profile,
                n=int(n),
                pixel_um=float(dx),
                domain_um=float(domain),
                factor=float(n) / float(n_mid),
                stock_object_pixel_um=stock,
                is_numerical_reference=bool(n == ref_n),
            )
        )
    return levels


def setup_for_level(profile: str, level: GridLevel):
    """Override object-map pitch only — preserve all other physics."""
    base = resolve_setup_selection(profile)
    return base.overridden(object_map_pixel_um=float(level.pixel_um))
