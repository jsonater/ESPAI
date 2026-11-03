"""Create dimensioned concept geometry for review; this is not image reconstruction."""

from __future__ import annotations

import math

import trimesh
from shapely.geometry import Polygon


MIN_DIMENSION_MM = 10.0
MAX_DIMENSION_MM = 2000.0


def create_stepped_cornice(width_mm: float, height_mm: float, depth_mm: float) -> trimesh.Trimesh:
    """Extrude a symmetric stepped profile using supplied millimeter dimensions."""
    dimensions = (width_mm, height_mm, depth_mm)
    if not all(math.isfinite(value) for value in dimensions):
        raise ValueError("All dimensions must be finite numbers")
    if any(value < MIN_DIMENSION_MM or value > MAX_DIMENSION_MM for value in dimensions):
        raise ValueError(f"Each dimension must be between {MIN_DIMENSION_MM:g} and {MAX_DIMENSION_MM:g} mm")

    profile = Polygon([
        (0, 0),
        (width_mm, 0),
        (width_mm, height_mm * 0.24),
        (width_mm * 0.85, height_mm * 0.24),
        (width_mm * 0.85, height_mm * 0.68),
        (width_mm * 0.72, height_mm * 0.68),
        (width_mm * 0.72, height_mm),
        (width_mm * 0.28, height_mm),
        (width_mm * 0.28, height_mm * 0.68),
        (width_mm * 0.15, height_mm * 0.68),
        (width_mm * 0.15, height_mm * 0.24),
        (0, height_mm * 0.24),
    ])
    if not profile.is_valid or profile.area <= 0:
        raise ValueError("Dimensions did not produce a valid stepped profile")

    mesh = trimesh.creation.extrude_polygon(profile, height=depth_mm, engine="earcut")
    mesh.apply_translation((-width_mm / 2, -height_mm / 2, 0))
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError("Generated concept mesh is not a watertight volume")
    return mesh