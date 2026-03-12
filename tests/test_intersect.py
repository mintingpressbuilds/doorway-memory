"""Tests for intersect.py — geometric query operations."""

from doorway_memory.shape import Dimension, Shape
from doorway_memory.intersect import (
    point_in_shape, find_containing_shapes, find_nearest_shapes, find_void
)


def _make_shape(name, xmin, xmax):
    return Shape(dimensions={"x": Dimension("x", xmin, xmax)}, metadata={"name": name})


def test_point_in_shape():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert point_in_shape({"x": 5.0}, shape) is True
    assert point_in_shape({"x": 15.0}, shape) is False


def test_find_containing_shapes():
    s1 = _make_shape("s1", 0.0, 10.0)
    s2 = _make_shape("s2", 5.0, 15.0)
    s3 = _make_shape("s3", 20.0, 30.0)

    # Point inside both s1 and s2
    result = find_containing_shapes({"x": 7.0}, [s1, s2, s3])
    assert len(result) == 2

    # Point inside only s1
    result = find_containing_shapes({"x": 3.0}, [s1, s2, s3])
    assert len(result) == 1
    assert result[0].metadata["name"] == "s1"

    # Point inside none
    result = find_containing_shapes({"x": 18.0}, [s1, s2, s3])
    assert len(result) == 0


def test_find_containing_shapes_empty():
    result = find_containing_shapes({"x": 5.0}, [])
    assert result == []


def test_find_nearest_shapes():
    s1 = _make_shape("s1", 0.0, 10.0)
    s2 = _make_shape("s2", 5.0, 15.0)

    result = find_nearest_shapes({"x": 7.0}, [s1, s2])
    # Both contain the point, so both should have positive distances
    assert len(result) == 2
    assert all(dist > 0 for _, dist in result)


def test_find_nearest_shapes_limit():
    shapes = [_make_shape(f"s{i}", float(i * 10), float(i * 10 + 5)) for i in range(10)]
    result = find_nearest_shapes({"x": 2.5}, shapes, limit=3)
    assert len(result) == 3


def test_find_nearest_shapes_outside():
    s1 = _make_shape("s1", 0.0, 10.0)
    # Point outside
    result = find_nearest_shapes({"x": 12.0}, [s1])
    assert len(result) == 1
    assert result[0][1] < 0  # negative distance = outside


def test_find_void_true():
    s1 = _make_shape("s1", 0.0, 10.0)
    assert find_void({"x": 15.0}, [s1]) is True


def test_find_void_false():
    s1 = _make_shape("s1", 0.0, 10.0)
    assert find_void({"x": 5.0}, [s1]) is False


def test_find_void_empty_library():
    assert find_void({"x": 5.0}, []) is True
