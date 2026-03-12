"""Tests for growth.py — shape expansion from near-misses."""

from doorway_memory.shape import Dimension, Shape
from doorway_memory.growth import (
    detect_near_miss,
    expand_shape,
    can_grow,
    try_grow,
    growth_depth,
    NEAR_MISS_THRESHOLD,
    GROWTH_MARGIN,
    GROWTH_CONFIDENCE_DECAY,
)


def _make_shape(xmin=0.0, xmax=10.0):
    return Shape(dimensions={"x": Dimension("x", xmin, xmax)})


def test_detect_near_miss_true():
    shape = _make_shape()
    # Point at 11.0 is 1.0 outside — within default threshold of 2.0
    assert detect_near_miss({"x": 11.0}, shape) is True


def test_detect_near_miss_too_far():
    shape = _make_shape()
    # Point at 15.0 is 5.0 outside — beyond threshold
    assert detect_near_miss({"x": 15.0}, shape) is False


def test_detect_near_miss_inside():
    shape = _make_shape()
    # Point inside is not a near miss
    assert detect_near_miss({"x": 5.0}, shape) is False


def test_detect_near_miss_on_boundary():
    shape = _make_shape()
    # Exactly on boundary — inside, not a near miss
    assert detect_near_miss({"x": 10.0}, shape) is False


def test_detect_near_miss_custom_threshold():
    shape = _make_shape()
    assert detect_near_miss({"x": 11.0}, shape, threshold=0.5) is False
    assert detect_near_miss({"x": 10.3}, shape, threshold=0.5) is True


def test_expand_shape_basic():
    shape = _make_shape()
    expanded = expand_shape(shape, {"x": 12.0})

    # New shape should contain the original point and the near-miss
    assert expanded.contains({"x": 5.0}) is True
    assert expanded.contains({"x": 12.0}) is True
    # And margin beyond
    assert expanded.contains({"x": 12.4}) is True


def test_expand_shape_preserves_original():
    shape = _make_shape()
    expand_shape(shape, {"x": 12.0})
    # Original is unchanged
    assert shape.contains({"x": 12.0}) is False
    assert shape.dimensions["x"].max_value == 10.0


def test_expand_shape_new_id():
    shape = _make_shape()
    expanded = expand_shape(shape, {"x": 12.0})
    assert expanded.id != shape.id


def test_expand_shape_parent_id():
    shape = _make_shape()
    expanded = expand_shape(shape, {"x": 12.0})
    assert expanded.parent_id == shape.id


def test_expand_shape_confidence_decay():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}, confidence=1.0)
    expanded = expand_shape(shape, {"x": 12.0})
    assert expanded.confidence == GROWTH_CONFIDENCE_DECAY


def test_expand_shape_metadata():
    shape = _make_shape()
    expanded = expand_shape(shape, {"x": 12.0})
    assert expanded.metadata["grown_from"] == shape.id


def test_expand_shape_below_min():
    shape = _make_shape()
    expanded = expand_shape(shape, {"x": -2.0})
    assert expanded.contains({"x": -2.0}) is True
    assert expanded.dimensions["x"].min_value == -2.0 - GROWTH_MARGIN


def test_expand_shape_multidimensional():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    expanded = expand_shape(shape, {"x": 12.0, "y": 5.0})
    # x expanded, y unchanged (point was inside y bounds)
    assert expanded.dimensions["x"].max_value > 10.0
    assert expanded.dimensions["y"].min_value == 0.0
    assert expanded.dimensions["y"].max_value == 10.0


def test_expand_shape_partial_point():
    """Point missing a dimension — that dimension stays unchanged."""
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    expanded = expand_shape(shape, {"x": 12.0})
    assert expanded.dimensions["y"].min_value == 0.0
    assert expanded.dimensions["y"].max_value == 10.0


def test_can_grow_normal():
    shape = _make_shape()
    assert can_grow(shape) is True


def test_can_grow_zero_confidence():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}, confidence=0.0)
    assert can_grow(shape) is False


def test_growth_depth_no_parent():
    shape = _make_shape()
    assert growth_depth(shape) == 0


def test_growth_depth_with_parent():
    shape = _make_shape()
    expanded = expand_shape(shape, {"x": 12.0})
    assert growth_depth(expanded) == 1


def test_try_grow_success():
    shape = _make_shape()
    result = try_grow(shape, {"x": 11.0})
    assert result is not None
    assert result.contains({"x": 11.0}) is True


def test_try_grow_too_far():
    shape = _make_shape()
    result = try_grow(shape, {"x": 20.0})
    assert result is None


def test_try_grow_inside():
    shape = _make_shape()
    result = try_grow(shape, {"x": 5.0})
    assert result is None


def test_try_grow_zero_confidence():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}, confidence=0.0)
    result = try_grow(shape, {"x": 11.0})
    assert result is None
