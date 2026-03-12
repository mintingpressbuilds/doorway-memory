"""Tests for memory.py — high-level geometric memory API."""

from doorway_memory.shape import Dimension, Shape
from doorway_memory.memory import Memory


def test_store_and_recall():
    mem = Memory(anchor=False)
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert len(mem.recall({"x": 5.0})) == 1


def test_void():
    mem = Memory(anchor=False)
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert mem.is_void({"x": 5.0}) is False
    assert mem.is_void({"x": 15.0}) is True


def test_is_known():
    mem = Memory(anchor=False)
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert mem.is_known({"x": 5.0}) is True
    assert mem.is_known({"x": 15.0}) is False


def test_store_with_metadata():
    mem = Memory(anchor=False)
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    shape_id = mem.store(shape, metadata={"source": "test"})
    retrieved = mem.library.get(shape_id)
    assert retrieved.metadata["source"] == "test"


def test_recall_raw_mode():
    mem = Memory(anchor=False)
    # Store a shape covering the extracted point dimensions
    shape = Shape(dimensions={
        "structure_type": Dimension("structure_type", 0.5, 1.5),
        "element_count": Dimension("element_count", 2.5, 3.5),
        "constraint_count": Dimension("constraint_count", 0.5, 1.5),
        "implication_direction": Dimension("implication_direction", 0.5, 1.5),
        "complexity_score": Dimension("complexity_score", 2.5, 3.5),
    })
    mem.store(shape)

    raw_input = {
        "structure": "causal",
        "elements": ["a", "b", "c"],
        "constraints": ["c1"],
        "implication": "forward"
    }
    result = mem.recall(raw_input, raw=True)
    assert len(result) == 1


def test_count():
    mem = Memory(anchor=False)
    assert mem.count() == 0
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert mem.count() == 1


def test_multiple_namespaces():
    mem1 = Memory(namespace="ns1", anchor=False)
    mem2 = Memory(namespace="ns2", anchor=False)
    mem1.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert mem1.count() == 1
    assert mem2.count() == 0


def test_multiple_shapes_recall():
    mem = Memory(anchor=False)
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    mem.store(Shape(dimensions={"x": Dimension("x", 5.0, 15.0)}))

    assert len(mem.recall({"x": 7.0})) == 2
    assert len(mem.recall({"x": 3.0})) == 1
    assert len(mem.recall({"x": 20.0})) == 0


def test_verify_without_xycore():
    mem = Memory(anchor=False)
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    shape_id = mem.store(shape)
    try:
        mem.verify(shape_id)
        assert False, "Should have raised ImportError"
    except ImportError:
        pass


def test_replay_without_xycore():
    mem = Memory(anchor=False)
    try:
        list(mem.replay())
        assert False, "Should have raised ImportError"
    except ImportError:
        pass
