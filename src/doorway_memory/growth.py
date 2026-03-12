"""Shape expansion from near-misses.

When a point falls just outside a shape's boundary, growth creates a new
expanded shape that encompasses the near-miss point. The original shape
is preserved (shapes are immutable); the new shape carries a parent_id
linking back to its origin.
"""

from typing import Dict, Optional
import numpy as np

from .shape import Shape, Dimension

# A point within this distance of a shape boundary is a "near miss"
NEAR_MISS_THRESHOLD = 2.0

# How much extra margin to add beyond the near-miss point
GROWTH_MARGIN = 0.5

# Confidence penalty applied to expanded shapes (multiplied)
GROWTH_CONFIDENCE_DECAY = 0.9

# Maximum number of times a shape lineage can grow
MAX_GROWTH_DEPTH = 10


def detect_near_miss(
    point: Dict[str, float], shape: Shape, threshold: float = NEAR_MISS_THRESHOLD
) -> bool:
    """
    Check if a point is a near miss — outside the shape but close.

    Returns True if point is outside and within threshold distance.
    """
    dist = shape.distance_to_boundary(point)
    return dist < 0 and abs(dist) <= threshold


def expand_shape(
    shape: Shape,
    point: Dict[str, float],
    margin: float = GROWTH_MARGIN,
    confidence_decay: float = GROWTH_CONFIDENCE_DECAY,
) -> Shape:
    """
    Create a new shape with boundaries expanded to include the point.

    The new shape's boundaries are the union of the original shape and
    the point, plus an optional margin. Confidence is reduced by the
    decay factor. The new shape's parent_id links to the original.

    Args:
        shape: The original shape to expand from.
        point: The near-miss point to encompass.
        margin: Extra space beyond the point to add.
        confidence_decay: Multiplier applied to the parent's confidence.

    Returns:
        A new Shape with expanded boundaries.
    """
    new_dims = {}
    for name, dim in shape.dimensions.items():
        if name in point:
            value = point[name]
            new_min = min(dim.min_value, value - margin)
            new_max = max(dim.max_value, value + margin)
        else:
            new_min = dim.min_value
            new_max = dim.max_value
        new_dims[name] = Dimension(name=name, min_value=new_min, max_value=new_max)

    return Shape(
        dimensions=new_dims,
        metadata={**(shape.metadata or {}), "grown_from": shape.id},
        confidence=shape.confidence * confidence_decay,
        parent_id=shape.id,
    )


def growth_depth(shape: Shape) -> int:
    """Count how many growth generations are recorded in metadata."""
    depth = 0
    meta = shape.metadata or {}
    while "grown_from" in meta:
        depth += 1
        break  # We only track immediate parent in metadata
    return depth


def can_grow(shape: Shape, max_depth: int = MAX_GROWTH_DEPTH) -> bool:
    """Check if a shape is allowed to grow further."""
    if shape.confidence <= 0.0:
        return False
    return growth_depth(shape) < max_depth


def try_grow(
    shape: Shape,
    point: Dict[str, float],
    threshold: float = NEAR_MISS_THRESHOLD,
    margin: float = GROWTH_MARGIN,
) -> Optional[Shape]:
    """
    Attempt to grow a shape toward a near-miss point.

    Returns a new expanded Shape if the point is a near miss and
    the shape is allowed to grow, otherwise None.
    """
    if not can_grow(shape):
        return None
    if not detect_near_miss(point, shape, threshold):
        return None
    return expand_shape(shape, point, margin)
