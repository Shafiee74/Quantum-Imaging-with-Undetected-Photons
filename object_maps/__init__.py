"""Physical object-map loaders for QIUP simulator input."""

from object_maps.c_lab_y1_rabbit_physical import load_c_lab_y1_object_map
from object_maps.rabbit_loader import (
    load_object_map,
    resample_object_map_to_grid,
)

__all__ = ["load_c_lab_y1_object_map", "load_object_map", "resample_object_map_to_grid"]
