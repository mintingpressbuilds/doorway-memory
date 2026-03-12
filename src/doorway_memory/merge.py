"""Shape fusion when boundaries meet.

When two shapes overlap significantly, they can be merged into a single
shape that covers the union of their territories. The merged shape is
new (shapes are immutable); both originals are preserved with the
merged shape carrying metadata linking back to its parents.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .shape import Shape, Dimension
from .overlap import overlap_ratio

# Minimum overlap ratio to consider merging two shapes
MERGE_OVERLAP_THRESHOLD = 0.5

# Confidence of merged shape: weighted average of parents
# (alternative: min of parents — configurable via merge function)
MERGE_CONFIDENCE_MODE = "weighted"


def merge_shapes(shape_a: Shape, shape_b: Shape) -> Shape:
    """
    Fuse two shapes into one covering their combined territory.

    The merged shape's boundaries are the union (min of mins, max of maxes)
    on all dimensions present in either shape. Confidence is the
    weighted average by volume.

    Returns a new Shape with parent metadata.
    """
    all_dims = set(shape_a.dimensions.keys()) | set(shape_b.dimensions.keys())

    merged_dims = {}
    for name in all_dims:
        da = shape_a.dimensions.get(name)
        db = shape_b.dimensions.get(name)
        if da and db:
            new_min = min(da.min_value, db.min_value)
            new_max = max(da.max_value, db.max_value)
        elif da:
            new_min = da.min_value
            new_max = da.max_value
        else:
            new_min = db.min_value
            new_max = db.max_value
        merged_dims[name] = Dimension(name=name, min_value=new_min, max_value=new_max)

    vol_a = shape_a.volume()
    vol_b = shape_b.volume()
    total_vol = vol_a + vol_b
    if total_vol > 0:
        confidence = (shape_a.confidence * vol_a + shape_b.confidence * vol_b) / total_vol
    else:
        confidence = min(shape_a.confidence, shape_b.confidence)

    return Shape(
        dimensions=merged_dims,
        metadata={
            "merged_from": [shape_a.id, shape_b.id],
        },
        confidence=confidence,
        hit_count=shape_a.hit_count + shape_b.hit_count,
    )


def should_merge(
    shape_a: Shape,
    shape_b: Shape,
    threshold: float = MERGE_OVERLAP_THRESHOLD,
) -> bool:
    """
    Check if two shapes overlap enough to warrant merging.

    Returns True if the overlap ratio (relative to the smaller shape)
    exceeds the threshold.
    """
    return overlap_ratio(shape_a, shape_b) >= threshold


def find_merge_candidates(
    shapes: List[Shape],
    threshold: float = MERGE_OVERLAP_THRESHOLD,
) -> List[Tuple[Shape, Shape]]:
    """
    Find all pairs of shapes that are candidates for merging.

    Returns list of (shape_a, shape_b) tuples.
    """
    candidates = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if should_merge(shapes[i], shapes[j], threshold):
                candidates.append((shapes[i], shapes[j]))
    return candidates


def merge_all(
    shapes: List[Shape],
    threshold: float = MERGE_OVERLAP_THRESHOLD,
) -> List[Shape]:
    """
    Greedily merge all overlapping shape pairs until no more merges are possible.

    Returns a new list of shapes with merges applied. Original shapes
    that were merged are replaced by their merged result.
    """
    remaining = list(shapes)
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(remaining):
            j = i + 1
            while j < len(remaining):
                if should_merge(remaining[i], remaining[j], threshold):
                    merged = merge_shapes(remaining[i], remaining[j])
                    remaining[i] = merged
                    remaining.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return remaining
