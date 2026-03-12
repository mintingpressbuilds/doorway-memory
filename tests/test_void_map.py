"""Tests for void_map.py — negative space characterization."""

from doorway_memory.shape import Dimension, Shape
from doorway_memory.void_map import (
    VoidRegion,
    find_void_regions_1d,
    void_density,
    nearest_void,
    void_boundary_points,
)


def _make_shape(xmin, xmax, name=None):
    return Shape(
        dimensions={"x": Dimension("x", xmin, xmax)},
        metadata={"name": name} if name else None,
    )


# --- VoidRegion ---

def test_void_region_volume():
    vr = VoidRegion(dimensions={"x": Dimension("x", 5.0, 10.0)})
    assert vr.volume() == 5.0


def test_void_region_volume_2d():
    vr = VoidRegion(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 5.0),
    })
    assert vr.volume() == 50.0


def test_void_region_volume_empty():
    vr = VoidRegion(dimensions={})
    assert vr.volume() == 0.0


def test_void_region_contains():
    vr = VoidRegion(dimensions={"x": Dimension("x", 5.0, 10.0)})
    assert vr.contains({"x": 7.0}) is True
    assert vr.contains({"x": 3.0}) is False


# --- find_void_regions_1d ---

def test_find_void_regions_single_gap():
    s1 = _make_shape(0.0, 5.0)
    s2 = _make_shape(8.0, 10.0)
    voids = find_void_regions_1d([s1, s2], "x", (0.0, 10.0))
    assert len(voids) == 1
    assert voids[0].dimensions["x"].min_value == 5.0
    assert voids[0].dimensions["x"].max_value == 8.0


def test_find_void_regions_no_gaps():
    s1 = _make_shape(0.0, 6.0)
    s2 = _make_shape(4.0, 10.0)
    voids = find_void_regions_1d([s1, s2], "x", (0.0, 10.0))
    assert len(voids) == 0


def test_find_void_regions_gap_at_start():
    s1 = _make_shape(5.0, 10.0)
    voids = find_void_regions_1d([s1], "x", (0.0, 10.0))
    assert len(voids) == 1
    assert voids[0].dimensions["x"].min_value == 0.0
    assert voids[0].dimensions["x"].max_value == 5.0


def test_find_void_regions_gap_at_end():
    s1 = _make_shape(0.0, 5.0)
    voids = find_void_regions_1d([s1], "x", (0.0, 10.0))
    assert len(voids) == 1
    assert voids[0].dimensions["x"].min_value == 5.0
    assert voids[0].dimensions["x"].max_value == 10.0


def test_find_void_regions_both_ends():
    s1 = _make_shape(3.0, 7.0)
    voids = find_void_regions_1d([s1], "x", (0.0, 10.0))
    assert len(voids) == 2


def test_find_void_regions_multiple_gaps():
    s1 = _make_shape(2.0, 4.0)
    s2 = _make_shape(6.0, 8.0)
    voids = find_void_regions_1d([s1, s2], "x", (0.0, 10.0))
    # Gaps: [0,2], [4,6], [8,10]
    assert len(voids) == 3


def test_find_void_regions_no_shapes():
    voids = find_void_regions_1d([], "x", (0.0, 10.0))
    assert len(voids) == 1
    assert voids[0].dimensions["x"].min_value == 0.0
    assert voids[0].dimensions["x"].max_value == 10.0


def test_find_void_regions_min_gap():
    s1 = _make_shape(0.0, 5.0)
    s2 = _make_shape(5.01, 10.0)
    # Gap is 0.01 — below min_gap of 0.1
    voids = find_void_regions_1d([s1, s2], "x", (0.0, 10.0), min_gap=0.1)
    assert len(voids) == 0


def test_find_void_regions_wrong_dimension():
    s1 = Shape(dimensions={"y": Dimension("y", 0.0, 10.0)})
    voids = find_void_regions_1d([s1], "x", (0.0, 10.0))
    # Shape has no x dimension, so entire x range is void
    assert len(voids) == 1


# --- void_density ---

def test_void_density_all_void():
    density = void_density([], {"x": (0.0, 10.0)}, resolution=10)
    assert density == 1.0


def test_void_density_all_covered():
    s = _make_shape(0.0, 10.0)
    density = void_density([s], {"x": (0.0, 10.0)}, resolution=10)
    assert density == 0.0


def test_void_density_partial():
    s = _make_shape(0.0, 5.0)
    density = void_density([s], {"x": (0.0, 10.0)}, resolution=10)
    # Roughly half should be void (depends on grid alignment)
    assert 0.3 < density < 0.7


def test_void_density_empty_bounds():
    density = void_density([], {})
    assert density == 1.0


# --- nearest_void ---

def test_nearest_void_inside():
    s = _make_shape(0.0, 10.0)
    direction = nearest_void({"x": 3.0}, [s])
    # Closer to x=0 edge (dist=3) than x=10 (dist=7)
    assert direction is not None
    assert direction["x"] == -1.0


def test_nearest_void_near_max():
    s = _make_shape(0.0, 10.0)
    direction = nearest_void({"x": 8.0}, [s])
    assert direction["x"] == 1.0


def test_nearest_void_already_void():
    s = _make_shape(0.0, 10.0)
    result = nearest_void({"x": 15.0}, [s])
    assert result is None


def test_nearest_void_empty():
    result = nearest_void({"x": 5.0}, [])
    assert result is None


def test_nearest_void_2d():
    s = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    # At (1, 5): closest boundary is x=0 at distance 1
    direction = nearest_void({"x": 1.0, "y": 5.0}, [s])
    assert direction is not None
    assert "x" in direction
    assert direction["x"] == -1.0


# --- void_boundary_points ---

def test_void_boundary_points_basic():
    s1 = _make_shape(2.0, 5.0)
    s2 = _make_shape(7.0, 9.0)
    pts = void_boundary_points([s1, s2], "x", (0.0, 10.0))
    assert pts == [2.0, 5.0, 7.0, 9.0]


def test_void_boundary_points_within_bounds():
    s1 = _make_shape(0.0, 15.0)
    pts = void_boundary_points([s1], "x", (3.0, 10.0))
    # 0.0 and 15.0 are outside bounds, so no points
    assert pts == []


def test_void_boundary_points_empty():
    pts = void_boundary_points([], "x", (0.0, 10.0))
    assert pts == []


def test_void_boundary_points_wrong_dim():
    s1 = Shape(dimensions={"y": Dimension("y", 0.0, 10.0)})
    pts = void_boundary_points([s1], "x", (0.0, 10.0))
    assert pts == []
