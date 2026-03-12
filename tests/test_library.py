"""Tests for library.py — storage and retrieval."""

import json
import tempfile
from pathlib import Path

from doorway_memory.shape import Dimension, Shape
from doorway_memory.library import Library


def test_add_and_get():
    lib = Library()
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    shape_id = lib.add(shape)
    assert lib.get(shape_id).id == shape.id


def test_query_containment():
    lib = Library()
    shape1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    shape2 = Shape(dimensions={"x": Dimension("x", 5.0, 15.0)})
    lib.add(shape1)
    lib.add(shape2)

    assert len(lib.query({"x": 7.0})) == 2   # inside both
    assert len(lib.query({"x": 3.0})) == 1   # only shape1
    assert len(lib.query({"x": 20.0})) == 0  # void


def test_void_detection():
    lib = Library()
    lib.add(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert lib.is_void({"x": 5.0}) is False
    assert lib.is_void({"x": 15.0}) is True


def test_remove():
    lib = Library()
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    shape_id = lib.add(shape)
    assert lib.count() == 1
    assert lib.remove(shape_id) is True
    assert lib.count() == 0
    assert lib.get(shape_id) is None


def test_remove_nonexistent():
    lib = Library()
    assert lib.remove("nonexistent") is False


def test_all_iterator():
    lib = Library()
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    s2 = Shape(dimensions={"x": Dimension("x", 20.0, 30.0)})
    lib.add(s1)
    lib.add(s2)
    all_shapes = list(lib.all())
    assert len(all_shapes) == 2


def test_count():
    lib = Library()
    assert lib.count() == 0
    lib.add(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert lib.count() == 1


def test_file_backend_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "shapes.json")

        # Create and save
        lib1 = Library(backend="file", path=path)
        shape = Shape(
            dimensions={"x": Dimension("x", 0.0, 10.0)},
            metadata={"label": "test"}
        )
        lib1.add(shape)
        assert lib1.count() == 1

        # Reload from file
        lib2 = Library(backend="file", path=path)
        assert lib2.count() == 1
        restored = lib2.get(shape.id)
        assert restored is not None
        assert restored.metadata == {"label": "test"}
        assert restored.contains({"x": 5.0}) is True


def test_file_backend_nonexistent_path():
    """Loading from a nonexistent file should create an empty library."""
    lib = Library(backend="file", path="/tmp/nonexistent_doorway_test.json")
    assert lib.count() == 0
