"""Tests for shape.py — geometric primitives."""

from doorway_memory.shape import Dimension, Shape, extract_point


def test_contains_inside():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0)
    })
    assert shape.contains({"x": 5.0, "y": 5.0}) is True


def test_contains_outside():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0)
    })
    assert shape.contains({"x": 15.0, "y": 5.0}) is False


def test_contains_boundary():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.contains({"x": 0.0}) is True
    assert shape.contains({"x": 10.0}) is True


def test_contains_partial_dimensions():
    """Dimensions in shape but not in point are ignored."""
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0)
    })
    assert shape.contains({"x": 5.0}) is True


def test_contains_extra_dimensions():
    """Dimensions in point but not in shape are ignored."""
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.contains({"x": 5.0, "z": 999.0}) is True


def test_distance_inside():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.distance_to_boundary({"x": 5.0}) == 5.0
    assert shape.distance_to_boundary({"x": 2.0}) == 2.0


def test_distance_outside():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.distance_to_boundary({"x": 12.0}) == -2.0


def test_distance_on_boundary():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.distance_to_boundary({"x": 0.0}) == 0.0
    assert shape.distance_to_boundary({"x": 10.0}) == 0.0


def test_distance_multidimensional():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0)
    })
    # At (5,5), min distance to any edge is 5.0
    assert shape.distance_to_boundary({"x": 5.0, "y": 5.0}) == 5.0
    # At (1,5), min distance is 1.0 (to x=0 edge)
    assert shape.distance_to_boundary({"x": 1.0, "y": 5.0}) == 1.0


def test_shape_id_deterministic():
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    s2 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert s1.id == s2.id


def test_shape_id_differs_for_different_shapes():
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    s2 = Shape(dimensions={"x": Dimension("x", 0.0, 20.0)})
    assert s1.id != s2.id


def test_serialization_roundtrip():
    shape = Shape(
        dimensions={"x": Dimension("x", 0.0, 10.0), "y": Dimension("y", -5.0, 5.0)},
        metadata={"label": "test"}
    )
    data = shape.to_dict()
    restored = Shape.from_dict(data)
    assert restored.id == shape.id
    assert restored.metadata == shape.metadata
    assert restored.contains({"x": 5.0, "y": 0.0}) is True
    assert restored.contains({"x": 15.0}) is False


def test_extract_point_basic():
    point = extract_point({
        "structure": "causal",
        "elements": ["a", "b", "c"],
        "constraints": ["c1"],
        "implication": "forward"
    })
    assert point["structure_type"] == 1.0
    assert point["element_count"] == 3.0
    assert point["constraint_count"] == 1.0
    assert point["implication_direction"] == 1.0
    assert point["complexity_score"] == 3.0  # 3 elements * 1 constraint


def test_extract_point_minimal():
    point = extract_point({})
    assert point["complexity_score"] == 0.0


def test_extract_point_unknown_structure():
    point = extract_point({"structure": "unknown_type"})
    assert point["structure_type"] == 0.0


# --- Confidence / Phase 2 additions ---

def test_confidence_default():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.confidence == 1.0


def test_confidence_custom():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}, confidence=0.5)
    assert shape.confidence == 0.5


def test_hit_count_default():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.hit_count == 0


def test_record_hit():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    shape.record_hit()
    shape.record_hit()
    assert shape.hit_count == 2


def test_parent_id():
    parent = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    child = Shape(dimensions={"x": Dimension("x", 0.0, 12.0)}, parent_id=parent.id)
    assert child.parent_id == parent.id


def test_volume_1d():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.volume() == 10.0


def test_volume_2d():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 5.0)
    })
    assert shape.volume() == 50.0


def test_volume_empty():
    shape = Shape(dimensions={})
    assert shape.volume() == 0.0


def test_confidence_serialization_roundtrip():
    shape = Shape(
        dimensions={"x": Dimension("x", 0.0, 10.0)},
        confidence=0.75,
        hit_count=3,
        parent_id="abc123"
    )
    data = shape.to_dict()
    restored = Shape.from_dict(data)
    assert restored.confidence == 0.75
    assert restored.hit_count == 3
    assert restored.parent_id == "abc123"
