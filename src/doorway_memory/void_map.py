"""Negative space characterization.

The void is everything outside known shapes. void_map analyzes the
gaps between shapes — identifying bounded void regions, measuring
void density, and characterizing what the system does NOT know.

This is the inverse of the shape library: instead of asking "what do
I know?" it asks "what is the shape of my ignorance?"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .shape import Shape, Dimension

# Sampling resolution for void analysis (points per dimension)
VOID_SAMPLE_RESOLUTION = 10

# Minimum gap size to be considered a meaningful void region
MIN_VOID_GAP = 0.1


@dataclass
class VoidRegion:
    """A bounded region of unknown territory between shapes."""
    dimensions: Dict[str, Dimension]
    bounded_by: List[str] = field(default_factory=list)
    metadata: Optional[Dict] = None

    def volume(self) -> float:
        if not self.dimensions:
            return 0.0
        ranges = np.array([d.max_value - d.min_value for d in self.dimensions.values()])
        return float(np.prod(ranges))

    def contains(self, point: Dict[str, float]) -> bool:
        for name, dim in self.dimensions.items():
            if name in point:
                if point[name] < dim.min_value or point[name] > dim.max_value:
                    return False
        return True


def find_void_regions_1d(
    shapes: List[Shape],
    dimension: str,
    bounds: Tuple[float, float],
    min_gap: float = MIN_VOID_GAP,
) -> List[VoidRegion]:
    """
    Find void gaps along a single dimension within given bounds.

    Scans the dimension range and identifies intervals not covered
    by any shape.
    """
    intervals = []
    for shape in shapes:
        if dimension in shape.dimensions:
            dim = shape.dimensions[dimension]
            lo = max(dim.min_value, bounds[0])
            hi = min(dim.max_value, bounds[1])
            if lo < hi:
                intervals.append((lo, hi))

    if not intervals:
        if bounds[1] - bounds[0] >= min_gap:
            return [VoidRegion(
                dimensions={dimension: Dimension(dimension, bounds[0], bounds[1])},
            )]
        return []

    intervals.sort()

    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    voids = []
    if merged[0][0] - bounds[0] >= min_gap:
        voids.append(VoidRegion(
            dimensions={dimension: Dimension(dimension, bounds[0], merged[0][0])},
        ))

    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_end - gap_start >= min_gap:
            voids.append(VoidRegion(
                dimensions={dimension: Dimension(dimension, gap_start, gap_end)},
            ))

    if bounds[1] - merged[-1][1] >= min_gap:
        voids.append(VoidRegion(
            dimensions={dimension: Dimension(dimension, merged[-1][1], bounds[1])},
        ))

    return voids


def void_density(
    shapes: List[Shape],
    bounds: Dict[str, Tuple[float, float]],
    resolution: int = VOID_SAMPLE_RESOLUTION,
) -> float:
    """
    Estimate the fraction of the bounded space that is void.

    Samples points on a grid and checks what fraction are not
    contained by any shape. Returns a value between 0.0 (all known)
    and 1.0 (all void).
    """
    if not bounds:
        return 1.0

    dim_names = sorted(bounds.keys())
    grids = []
    for name in dim_names:
        lo, hi = bounds[name]
        grids.append(np.linspace(lo, hi, resolution))

    mesh = np.meshgrid(*grids, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    total = len(points)
    void_count = 0

    for row in points:
        point = {dim_names[i]: float(row[i]) for i in range(len(dim_names))}
        in_any = any(s.contains(point) for s in shapes)
        if not in_any:
            void_count += 1

    return void_count / total if total > 0 else 1.0


def nearest_void(
    point: Dict[str, float],
    shapes: List[Shape],
) -> Optional[Dict[str, float]]:
    """
    Find the direction from a point toward the nearest void.

    For a point inside known territory, returns a direction vector
    pointing toward the nearest shape boundary (and thus toward void).
    Returns None if the point is already in the void.
    """
    from .intersect import find_containing_shapes

    containing = find_containing_shapes(point, shapes)
    if not containing:
        return None  # Already in void

    min_dist = float('inf')
    best_direction = None

    for shape in containing:
        for name, dim in shape.dimensions.items():
            if name not in point:
                continue
            val = point[name]
            dist_to_min = val - dim.min_value
            dist_to_max = dim.max_value - val

            if dist_to_min < min_dist:
                min_dist = dist_to_min
                best_direction = {name: -1.0}
            if dist_to_max < min_dist:
                min_dist = dist_to_max
                best_direction = {name: 1.0}

    return best_direction


def void_boundary_points(
    shapes: List[Shape],
    dimension: str,
    bounds: Tuple[float, float],
) -> List[float]:
    """
    Find all boundary points along a dimension where territory meets void.

    Returns sorted list of boundary values (both shape mins and maxes)
    that fall within the given bounds.
    """
    boundary_pts = set()
    for shape in shapes:
        if dimension in shape.dimensions:
            dim = shape.dimensions[dimension]
            if bounds[0] <= dim.min_value <= bounds[1]:
                boundary_pts.add(dim.min_value)
            if bounds[0] <= dim.max_value <= bounds[1]:
                boundary_pts.add(dim.max_value)
    return sorted(boundary_pts)
