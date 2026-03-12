"""Tests for decay.py — knowledge fading and archive."""

import math

from doorway_memory.shape import Dimension, Shape
from doorway_memory.decay import (
    decay_shape,
    should_archive,
    archive_shape,
    apply_decay_steps,
    DECAY_FACTOR,
    DECAY_CONFIDENCE_FACTOR,
    MIN_VOLUME_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
)


def _make_shape(xmin=0.0, xmax=10.0, confidence=1.0):
    return Shape(
        dimensions={"x": Dimension("x", xmin, xmax)},
        confidence=confidence,
    )


def test_decay_shape_shrinks():
    shape = _make_shape(0.0, 10.0)
    decayed = decay_shape(shape)
    assert decayed.dimensions["x"].min_value > 0.0
    assert decayed.dimensions["x"].max_value < 10.0


def test_decay_shape_correct_amount():
    shape = _make_shape(0.0, 10.0)
    decayed = decay_shape(shape, factor=0.2)
    # Range is 10, shrink = 10*0.2/2 = 1.0 from each side
    assert decayed.dimensions["x"].min_value == 1.0
    assert decayed.dimensions["x"].max_value == 9.0


def test_decay_shape_preserves_original():
    shape = _make_shape(0.0, 10.0)
    decay_shape(shape)
    assert shape.dimensions["x"].min_value == 0.0
    assert shape.dimensions["x"].max_value == 10.0


def test_decay_shape_new_id():
    shape = _make_shape()
    decayed = decay_shape(shape)
    assert decayed.id != shape.id


def test_decay_shape_parent_id():
    shape = _make_shape()
    decayed = decay_shape(shape)
    assert decayed.parent_id == shape.id


def test_decay_shape_confidence():
    shape = _make_shape(confidence=1.0)
    decayed = decay_shape(shape)
    assert decayed.confidence == DECAY_CONFIDENCE_FACTOR


def test_decay_shape_metadata():
    shape = _make_shape()
    decayed = decay_shape(shape)
    assert decayed.metadata["decayed_from"] == shape.id


def test_decay_shape_collapses_to_point():
    """If factor is 1.0, the shape collapses to a point (zero volume)."""
    shape = _make_shape(0.0, 10.0)
    decayed = decay_shape(shape, factor=1.0)
    assert decayed.dimensions["x"].min_value == decayed.dimensions["x"].max_value
    assert decayed.volume() == 0.0


def test_decay_shape_multidimensional():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 20.0),
    })
    decayed = decay_shape(shape, factor=0.2)
    # x: range 10, shrink 1.0 each side → [1, 9]
    assert decayed.dimensions["x"].min_value == 1.0
    assert decayed.dimensions["x"].max_value == 9.0
    # y: range 20, shrink 2.0 each side → [2, 18]
    assert decayed.dimensions["y"].min_value == 2.0
    assert decayed.dimensions["y"].max_value == 18.0


def test_decay_shape_preserves_hit_count():
    shape = _make_shape()
    shape.record_hit()
    shape.record_hit()
    decayed = decay_shape(shape)
    assert decayed.hit_count == 2


def test_should_archive_low_confidence():
    shape = _make_shape(confidence=0.01)
    assert should_archive(shape) is True


def test_should_archive_healthy():
    shape = _make_shape(confidence=1.0)
    assert should_archive(shape) is False


def test_should_archive_zero_volume():
    shape = Shape(
        dimensions={"x": Dimension("x", 5.0, 5.0)},  # zero-width
        confidence=1.0,
    )
    assert should_archive(shape) is True


def test_should_archive_custom_thresholds():
    shape = _make_shape(confidence=0.3)
    # Default threshold is 0.05, so this is healthy
    assert should_archive(shape) is False
    # But with a higher threshold it should archive
    assert should_archive(shape, min_confidence=0.5) is True


def test_archive_shape_sets_flag():
    shape = _make_shape()
    archived = archive_shape(shape)
    assert archived.metadata["archived"] is True


def test_archive_shape_preserves_id():
    shape = _make_shape()
    archived = archive_shape(shape)
    assert archived.id == shape.id


def test_archive_shape_preserves_confidence():
    shape = _make_shape(confidence=0.3)
    archived = archive_shape(shape)
    assert archived.confidence == 0.3


def test_apply_decay_steps():
    shape = _make_shape(0.0, 100.0, confidence=1.0)
    decayed = apply_decay_steps(shape, steps=3, factor=0.2, confidence_factor=0.5)
    # Confidence: 1.0 * 0.5^3 = 0.125
    assert abs(decayed.confidence - 0.125) < 1e-10


def test_apply_decay_steps_zero():
    shape = _make_shape(0.0, 10.0)
    decayed = apply_decay_steps(shape, steps=0)
    # No decay applied — same shape returned
    assert decayed.dimensions["x"].min_value == 0.0
    assert decayed.dimensions["x"].max_value == 10.0


def test_decay_eventually_archives():
    """Repeated decay should eventually make a shape archivable."""
    shape = _make_shape(0.0, 10.0, confidence=1.0)
    decayed = apply_decay_steps(shape, steps=50)
    assert should_archive(decayed) is True
