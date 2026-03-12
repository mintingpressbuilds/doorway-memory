"""Pairwise cross-domain intersection.

Finds where two shapes overlap in their shared dimensions and computes
the intersection region. Useful for detecting when knowledge domains
share common territory.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .shape import Shape, Dimension

# Minimum overlap volume to consider meaningful
MIN_OVERLAP_VOLUME = 1e-10


def find_overlap(shape_a: Shape, shape_b: Shape) -> Optional[Shape]:
    """
    Compute the intersection region of two shapes on shared dimensions.

    Returns a new Shape representing the overlap, or None if
    the shapes do not overlap or share no dimensions.
    """
    shared = set(shape_a.dimensions.keys()) & set(shape_b.dimensions.keys())
    if not shared:
        return None

    overlap_dims = {}
    for name in shared:
        da = shape_a.dimensions[name]
        db = shape_b.dimensions[name]
        new_min = max(da.min_value, db.min_value)
        new_max = min(da.max_value, db.max_value)
        if new_min > new_max:
            return None  # No overlap on this dimension
        overlap_dims[name] = Dimension(name=name, min_value=new_min, max_value=new_max)

    overlap_shape = Shape(
        dimensions=overlap_dims,
        metadata={
            "overlap_of": [shape_a.id, shape_b.id],
        },
        confidence=min(shape_a.confidence, shape_b.confidence),
    )
    return overlap_shape


def overlap_volume(shape_a: Shape, shape_b: Shape) -> float:
    """
    Calculate the volume of the overlap region between two shapes.

    Returns 0.0 if there is no overlap.
    """
    overlap = find_overlap(shape_a, shape_b)
    if overlap is None:
        return 0.0
    return overlap.volume()


def overlap_ratio(shape_a: Shape, shape_b: Shape) -> float:
    """
    Ratio of overlap volume to the smaller shape's volume.

    Returns a value between 0.0 (no overlap) and 1.0 (complete containment).
    """
    vol = overlap_volume(shape_a, shape_b)
    if vol == 0.0:
        return 0.0
    smaller = min(shape_a.volume(), shape_b.volume())
    if smaller == 0.0:
        return 0.0
    return vol / smaller


def pairwise_overlaps(
    shapes: List[Shape], min_volume: float = MIN_OVERLAP_VOLUME
) -> List[Tuple[Shape, Shape, Shape]]:
    """
    Find all pairwise overlaps among a list of shapes.

    Returns a list of (shape_a, shape_b, overlap_shape) tuples
    for every pair that has a meaningful overlap.
    """
    results = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            overlap = find_overlap(shapes[i], shapes[j])
            if overlap is not None and overlap.volume() >= min_volume:
                results.append((shapes[i], shapes[j], overlap))
    return results
