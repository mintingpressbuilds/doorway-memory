"""Tests for overlap.py — pairwise cross-domain intersection."""

from doorway_memory.shape import Dimension, Shape
from doorway_memory.overlap import (
    find_overlap,
    overlap_volume,
    overlap_ratio,
    pairwise_overlaps,
)


def _make_shape(name, xmin, xmax):
    return Shape(
        dimensions={"x": Dimension("x", xmin, xmax)},
        metadata={"name": name},
    )


def test_find_overlap_basic():
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 5.0, 15.0)
    overlap = find_overlap(s1, s2)
    assert overlap is not None
    assert overlap.dimensions["x"].min_value == 5.0
    assert overlap.dimensions["x"].max_value == 10.0


def test_find_overlap_no_intersection():
    s1 = _make_shape("a", 0.0, 5.0)
    s2 = _make_shape("b", 10.0, 15.0)
    assert find_overlap(s1, s2) is None


def test_find_overlap_touching():
    """Shapes that touch at a single point have zero-volume overlap."""
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 10.0, 20.0)
    overlap = find_overlap(s1, s2)
    assert overlap is not None
    assert overlap.volume() == 0.0


def test_find_overlap_full_containment():
    s1 = _make_shape("a", 0.0, 20.0)
    s2 = _make_shape("b", 5.0, 10.0)
    overlap = find_overlap(s1, s2)
    assert overlap is not None
    assert overlap.dimensions["x"].min_value == 5.0
    assert overlap.dimensions["x"].max_value == 10.0


def test_find_overlap_no_shared_dims():
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    s2 = Shape(dimensions={"y": Dimension("y", 0.0, 10.0)})
    assert find_overlap(s1, s2) is None


def test_find_overlap_multidimensional():
    s1 = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    s2 = Shape(dimensions={
        "x": Dimension("x", 5.0, 15.0),
        "y": Dimension("y", 5.0, 15.0),
    })
    overlap = find_overlap(s1, s2)
    assert overlap is not None
    assert overlap.dimensions["x"].min_value == 5.0
    assert overlap.dimensions["x"].max_value == 10.0
    assert overlap.dimensions["y"].min_value == 5.0
    assert overlap.dimensions["y"].max_value == 10.0


def test_find_overlap_partial_shared_dims():
    """Only shared dimensions are included in the overlap shape."""
    s1 = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    s2 = Shape(dimensions={
        "x": Dimension("x", 5.0, 15.0),
        "z": Dimension("z", 0.0, 10.0),
    })
    overlap = find_overlap(s1, s2)
    assert overlap is not None
    assert "x" in overlap.dimensions
    assert "y" not in overlap.dimensions
    assert "z" not in overlap.dimensions


def test_find_overlap_confidence():
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}, confidence=0.8)
    s2 = Shape(dimensions={"x": Dimension("x", 5.0, 15.0)}, confidence=0.6)
    overlap = find_overlap(s1, s2)
    assert overlap.confidence == 0.6  # min of the two


def test_find_overlap_metadata():
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 5.0, 15.0)
    overlap = find_overlap(s1, s2)
    assert s1.id in overlap.metadata["overlap_of"]
    assert s2.id in overlap.metadata["overlap_of"]


def test_overlap_volume_basic():
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 5.0, 15.0)
    assert overlap_volume(s1, s2) == 5.0


def test_overlap_volume_none():
    s1 = _make_shape("a", 0.0, 5.0)
    s2 = _make_shape("b", 10.0, 15.0)
    assert overlap_volume(s1, s2) == 0.0


def test_overlap_volume_2d():
    s1 = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    s2 = Shape(dimensions={
        "x": Dimension("x", 5.0, 15.0),
        "y": Dimension("y", 5.0, 15.0),
    })
    assert overlap_volume(s1, s2) == 25.0  # 5 * 5


def test_overlap_ratio_full():
    s1 = _make_shape("a", 0.0, 20.0)
    s2 = _make_shape("b", 5.0, 10.0)
    assert overlap_ratio(s1, s2) == 1.0  # s2 fully inside s1


def test_overlap_ratio_partial():
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 5.0, 15.0)
    # overlap = 5, smaller shape = 10, ratio = 0.5
    assert overlap_ratio(s1, s2) == 0.5


def test_overlap_ratio_none():
    s1 = _make_shape("a", 0.0, 5.0)
    s2 = _make_shape("b", 10.0, 15.0)
    assert overlap_ratio(s1, s2) == 0.0


def test_pairwise_overlaps_basic():
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 5.0, 15.0)
    s3 = _make_shape("c", 20.0, 30.0)
    results = pairwise_overlaps([s1, s2, s3])
    # Only s1-s2 overlap
    assert len(results) == 1
    assert results[0][2].dimensions["x"].min_value == 5.0


def test_pairwise_overlaps_empty():
    assert pairwise_overlaps([]) == []


def test_pairwise_overlaps_single():
    s1 = _make_shape("a", 0.0, 10.0)
    assert pairwise_overlaps([s1]) == []


def test_pairwise_overlaps_all_overlap():
    s1 = _make_shape("a", 0.0, 10.0)
    s2 = _make_shape("b", 3.0, 13.0)
    s3 = _make_shape("c", 6.0, 16.0)
    results = pairwise_overlaps([s1, s2, s3])
    # s1-s2, s1-s3, s2-s3 all overlap
    assert len(results) == 3
