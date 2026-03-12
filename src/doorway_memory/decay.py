"""Knowledge fading and archive.

Shapes that are not accessed or confirmed over time lose confidence
and shrink. When a shape's volume or confidence drops below thresholds
it is archived — moved out of active queries but preserved for history.

Shapes are immutable: decay produces a new, smaller shape with a
parent_id linking to the original.
"""

from typing import Dict, Optional
import numpy as np

from .shape import Shape, Dimension

# Default shrink factor per decay step (boundaries shrink by this fraction)
DECAY_FACTOR = 0.1

# Confidence reduction per decay step (multiplied)
DECAY_CONFIDENCE_FACTOR = 0.8

# Shapes below this volume are candidates for archiving
MIN_VOLUME_THRESHOLD = 1e-6

# Shapes below this confidence are candidates for archiving
MIN_CONFIDENCE_THRESHOLD = 0.05


def decay_shape(
    shape: Shape,
    factor: float = DECAY_FACTOR,
    confidence_factor: float = DECAY_CONFIDENCE_FACTOR,
) -> Shape:
    """
    Shrink a shape's boundaries inward, reducing its territory.

    Each dimension's range contracts by `factor` from both sides.
    For a dimension [min, max] with range R, the new boundaries are:
        [min + R*factor/2, max - R*factor/2]

    Confidence is multiplied by confidence_factor.

    Returns a new Shape (shapes are immutable).
    """
    new_dims = {}
    for name, dim in shape.dimensions.items():
        span = dim.max_value - dim.min_value
        shrink = span * factor / 2.0
        new_min = dim.min_value + shrink
        new_max = dim.max_value - shrink
        if new_min > new_max:
            mid = (dim.min_value + dim.max_value) / 2.0
            new_min = mid
            new_max = mid
        new_dims[name] = Dimension(name=name, min_value=new_min, max_value=new_max)

    return Shape(
        dimensions=new_dims,
        metadata={**(shape.metadata or {}), "decayed_from": shape.id},
        confidence=shape.confidence * confidence_factor,
        parent_id=shape.id,
        hit_count=shape.hit_count,
    )


def should_archive(
    shape: Shape,
    min_volume: float = MIN_VOLUME_THRESHOLD,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
) -> bool:
    """
    Check if a shape has decayed enough to be archived.

    A shape should be archived if its volume is below the threshold
    OR its confidence is below the threshold.
    """
    if shape.confidence < min_confidence:
        return True
    if shape.volume() < min_volume:
        return True
    return False


def archive_shape(shape: Shape) -> Shape:
    """
    Mark a shape as archived by adding archive metadata.

    Returns a new Shape with archive flag in metadata.
    """
    meta = {**(shape.metadata or {}), "archived": True}
    return Shape(
        dimensions=shape.dimensions,
        metadata=meta,
        id=shape.id,
        anchor_id=shape.anchor_id,
        confidence=shape.confidence,
        hit_count=shape.hit_count,
        parent_id=shape.parent_id,
    )


def apply_decay_steps(
    shape: Shape,
    steps: int,
    factor: float = DECAY_FACTOR,
    confidence_factor: float = DECAY_CONFIDENCE_FACTOR,
) -> Shape:
    """
    Apply multiple decay steps to a shape.

    Returns the final decayed shape after all steps.
    """
    current = shape
    for _ in range(steps):
        current = decay_shape(current, factor=factor, confidence_factor=confidence_factor)
    return current
