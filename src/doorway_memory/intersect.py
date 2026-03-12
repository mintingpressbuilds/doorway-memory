"""Geometric query operations for doorway-memory."""

from typing import List, Dict, Tuple
from .shape import Shape


def point_in_shape(point: Dict[str, float], shape: Shape) -> bool:
    """Wrapper for shape.contains()."""
    return shape.contains(point)


def find_containing_shapes(
    point: Dict[str, float],
    shapes: List[Shape]
) -> List[Shape]:
    """Find all shapes that contain this point."""
    return [s for s in shapes if s.contains(point)]


def find_nearest_shapes(
    point: Dict[str, float],
    shapes: List[Shape],
    limit: int = 5
) -> List[Tuple[Shape, float]]:
    """
    Find shapes nearest to this point, even if outside.

    Returns list of (shape, distance) tuples sorted by distance.
    Positive = inside. Negative = outside.
    """
    distances = []
    for shape in shapes:
        dist = shape.distance_to_boundary(point)
        distances.append((shape, dist))

    # Sort: containing shapes first (by distance desc), then outside shapes (by distance desc / closest first)
    distances.sort(key=lambda x: (-x[1] if x[1] > 0 else float('inf') + abs(x[1])))
    return distances[:limit]


def find_void(point: Dict[str, float], shapes: List[Shape]) -> bool:
    """True if no shape contains this point."""
    return len(find_containing_shapes(point, shapes)) == 0
