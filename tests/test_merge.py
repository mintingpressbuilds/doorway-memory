"""Tests for merge.py — shape fusion when boundaries meet."""

from doorway_memory.shape import Dimension, Shape
from doorway_memory.merge import (
    merge_shapes,
    should_merge,
    find_merge_candidates,
    merge_all,
    MERGE_OVERLAP_THRESHOLD,
)


def _make_shape(xmin, xmax, confidence=1.0, name=None):
    return Shape(
        dimensions={"x": Dimension("x", xmin, xmax)},
        confidence=confidence,
        metadata={"name": name} if name else None,
    )


def test_merge_shapes_union():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    merged = merge_shapes(s1, s2)
    assert merged.dimensions["x"].min_value == 0.0
    assert merged.dimensions["x"].max_value == 15.0


def test_merge_shapes_new_id():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    merged = merge_shapes(s1, s2)
    assert merged.id != s1.id
    assert merged.id != s2.id


def test_merge_shapes_metadata():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    merged = merge_shapes(s1, s2)
    assert s1.id in merged.metadata["merged_from"]
    assert s2.id in merged.metadata["merged_from"]


def test_merge_shapes_confidence_weighted():
    s1 = _make_shape(0.0, 10.0, confidence=1.0)  # vol=10
    s2 = _make_shape(5.0, 15.0, confidence=0.5)  # vol=10
    merged = merge_shapes(s1, s2)
    # Weighted avg: (1.0*10 + 0.5*10) / 20 = 0.75
    assert abs(merged.confidence - 0.75) < 1e-10


def test_merge_shapes_confidence_unequal_volume():
    s1 = _make_shape(0.0, 20.0, confidence=1.0)  # vol=20
    s2 = _make_shape(15.0, 20.0, confidence=0.0)  # vol=5
    merged = merge_shapes(s1, s2)
    # Weighted avg: (1.0*20 + 0.0*5) / 25 = 0.8
    assert abs(merged.confidence - 0.8) < 1e-10


def test_merge_shapes_hit_count():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    s1.record_hit()
    s1.record_hit()
    s2.record_hit()
    merged = merge_shapes(s1, s2)
    assert merged.hit_count == 3


def test_merge_shapes_disjoint_dims():
    """Shapes with non-overlapping dimension sets get union of all dims."""
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    s2 = Shape(dimensions={"y": Dimension("y", 0.0, 5.0)})
    merged = merge_shapes(s1, s2)
    assert "x" in merged.dimensions
    assert "y" in merged.dimensions
    assert merged.dimensions["x"].min_value == 0.0
    assert merged.dimensions["y"].max_value == 5.0


def test_merge_shapes_multidimensional():
    s1 = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    s2 = Shape(dimensions={
        "x": Dimension("x", 5.0, 15.0),
        "y": Dimension("y", 5.0, 15.0),
    })
    merged = merge_shapes(s1, s2)
    assert merged.dimensions["x"].min_value == 0.0
    assert merged.dimensions["x"].max_value == 15.0
    assert merged.dimensions["y"].min_value == 0.0
    assert merged.dimensions["y"].max_value == 15.0


def test_merge_shapes_containment():
    """Merged shape contains both originals' territory."""
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(8.0, 20.0)
    merged = merge_shapes(s1, s2)
    assert merged.contains({"x": 1.0}) is True
    assert merged.contains({"x": 15.0}) is True
    assert merged.contains({"x": 25.0}) is False


def test_should_merge_high_overlap():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(2.0, 12.0)
    # Overlap is [2,10]=8, smaller shape is 10, ratio=0.8
    assert should_merge(s1, s2) is True


def test_should_merge_low_overlap():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(9.0, 20.0)
    # Overlap is [9,10]=1, smaller shape is 10, ratio=0.1
    assert should_merge(s1, s2) is False


def test_should_merge_no_overlap():
    s1 = _make_shape(0.0, 5.0)
    s2 = _make_shape(10.0, 15.0)
    assert should_merge(s1, s2) is False


def test_should_merge_full_containment():
    s1 = _make_shape(0.0, 20.0)
    s2 = _make_shape(5.0, 10.0)
    # ratio=1.0 — s2 fully inside s1
    assert should_merge(s1, s2) is True


def test_should_merge_custom_threshold():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(7.0, 17.0)
    # Overlap=[7,10]=3, smaller=10, ratio=0.3
    assert should_merge(s1, s2, threshold=0.2) is True
    assert should_merge(s1, s2, threshold=0.5) is False


def test_find_merge_candidates_basic():
    s1 = _make_shape(0.0, 10.0, name="a")
    s2 = _make_shape(2.0, 12.0, name="b")
    s3 = _make_shape(50.0, 60.0, name="c")
    candidates = find_merge_candidates([s1, s2, s3])
    assert len(candidates) == 1


def test_find_merge_candidates_none():
    s1 = _make_shape(0.0, 5.0)
    s2 = _make_shape(10.0, 15.0)
    s3 = _make_shape(20.0, 25.0)
    assert find_merge_candidates([s1, s2, s3]) == []


def test_find_merge_candidates_empty():
    assert find_merge_candidates([]) == []


def test_merge_all_basic():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(2.0, 12.0)
    s3 = _make_shape(50.0, 60.0)
    result = merge_all([s1, s2, s3])
    assert len(result) == 2  # s1+s2 merged, s3 standalone


def test_merge_all_chain():
    """Three shapes that overlap pairwise should all merge into one."""
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(4.0, 14.0)
    s3 = _make_shape(8.0, 18.0)
    result = merge_all([s1, s2, s3])
    assert len(result) == 1
    assert result[0].dimensions["x"].min_value == 0.0
    assert result[0].dimensions["x"].max_value == 18.0


def test_merge_all_no_merges():
    s1 = _make_shape(0.0, 5.0)
    s2 = _make_shape(10.0, 15.0)
    result = merge_all([s1, s2])
    assert len(result) == 2


def test_merge_all_preserves_originals():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(2.0, 12.0)
    merge_all([s1, s2])
    # Originals unchanged
    assert s1.dimensions["x"].max_value == 10.0
    assert s2.dimensions["x"].min_value == 2.0


def test_merge_all_empty():
    assert merge_all([]) == []


def test_merge_all_single():
    s1 = _make_shape(0.0, 10.0)
    result = merge_all([s1])
    assert len(result) == 1
