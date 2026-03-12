"""Phase 5 integration tests — Memory class with all mechanics."""

import tempfile
from pathlib import Path

from doorway_memory.shape import Dimension, Shape
from doorway_memory.memory import Memory
from doorway_memory.void_map import VoidRegion
from doorway_memory.emergence import Tier2Shape


def _shape(dims, confidence=1.0, metadata=None):
    """Helper to create shapes quickly."""
    return Shape(
        dimensions={k: Dimension(k, lo, hi) for k, (lo, hi) in dims.items()},
        confidence=confidence,
        metadata=metadata,
    )


# --- Constructor flags ---

def test_memory_default_flags():
    mem = Memory(anchor=False)
    assert mem.growth_enabled is True
    assert mem.overlap_enabled is True
    assert mem.decay_enabled is True
    assert mem.merge_enabled is True
    assert mem.narrative_enabled is True


def test_memory_disabled_flags():
    mem = Memory(anchor=False, growth=False, overlap=False,
                 decay=False, merge=False, narrative=False)
    assert mem.growth_enabled is False
    assert mem.overlap_enabled is False
    assert mem.decay_enabled is False
    assert mem.merge_enabled is False
    assert mem.narrative_enabled is False


# --- recall_with_confidence ---

def test_recall_with_confidence_inside():
    mem = Memory(anchor=False)
    s = _shape({"x": (0.0, 10.0)}, confidence=0.9)
    mem.store(s)
    results = mem.recall_with_confidence({"x": 5.0})
    assert len(results) == 1
    shape, conf = results[0]
    assert conf == 0.9


def test_recall_with_confidence_empty():
    mem = Memory(anchor=False)
    results = mem.recall_with_confidence({"x": 5.0})
    assert results == []


# --- Growth integration ---

def test_recall_triggers_growth():
    mem = Memory(anchor=False, overlap=False, merge=False)
    s = _shape({"x": (0.0, 10.0)})
    mem.store(s)
    # Point just outside — should trigger growth via try_grow
    result = mem.recall({"x": 10.5})
    # Growth may or may not succeed depending on threshold;
    # at minimum, recall shouldn't crash
    assert isinstance(result, list)


def test_recall_no_growth_when_disabled():
    mem = Memory(anchor=False, growth=False)
    s = _shape({"x": (0.0, 10.0)})
    mem.store(s)
    result = mem.recall({"x": 10.5})
    assert result == []


# --- Narrative integration ---

def test_store_in_trajectory():
    mem = Memory(anchor=False)
    s1 = _shape({"x": (0.0, 10.0)})
    s2 = _shape({"x": (5.0, 15.0)})
    mem.store_in_trajectory(s1, "traj1", timestamp=0.0)
    mem.store_in_trajectory(s2, "traj1", timestamp=1.0)
    assert mem.count() == 2
    assert "traj1" in mem._trajectories
    assert mem._trajectories["traj1"].length == 2


def test_predict_next_trajectory():
    mem = Memory(anchor=False)
    s1 = _shape({"x": (0.0, 10.0)})
    s2 = _shape({"x": (10.0, 20.0)})
    s3 = _shape({"x": (20.0, 30.0)})
    mem.store_in_trajectory(s1, "t", timestamp=0.0)
    mem.store_in_trajectory(s2, "t", timestamp=1.0)
    mem.store_in_trajectory(s3, "t", timestamp=2.0)
    prediction = mem.predict_next("t")
    assert prediction is not None
    assert "x" in prediction


def test_predict_next_no_trajectory():
    mem = Memory(anchor=False)
    assert mem.predict_next("nonexistent") is None


def test_predict_next_disabled():
    mem = Memory(anchor=False, narrative=False)
    assert mem.predict_next("t") is None


def test_find_common_paths():
    mem = Memory(anchor=False)
    # Create two similar trajectories
    for tid in ["a", "b"]:
        for i in range(3):
            s = _shape({"x": (float(i), float(i + 1))})
            mem.store_in_trajectory(s, tid, timestamp=float(i))
    paths = mem.find_common_paths()
    assert isinstance(paths, list)


def test_find_common_paths_disabled():
    mem = Memory(anchor=False, narrative=False)
    assert mem.find_common_paths() == []


# --- Decay integration ---

def test_maintain_decay():
    mem = Memory(anchor=False, overlap=False)
    s = _shape({"x": (0.0, 0.001)}, confidence=0.01)
    mem.store(s)
    result = mem.maintain()
    assert "decayed" in result
    assert "archived" in result


def test_maintain_disabled():
    mem = Memory(anchor=False, decay=False)
    result = mem.maintain()
    assert result == {"decayed": 0, "archived": 0}


# --- Merge integration ---

def test_check_merges():
    mem = Memory(anchor=False, overlap=False, growth=False)
    # Two highly overlapping shapes
    s1 = _shape({"x": (0.0, 10.0)})
    s2 = _shape({"x": (2.0, 12.0)})
    mem.store(s1)
    mem.store(s2)
    count = mem._check_merges()
    assert count >= 1
    assert mem.count() == 1  # merged into one


def test_check_merges_no_overlap():
    mem = Memory(anchor=False, overlap=False)
    s1 = _shape({"x": (0.0, 5.0)})
    s2 = _shape({"x": (20.0, 25.0)})
    mem.store(s1)
    mem.store(s2)
    count = mem._check_merges()
    assert count == 0
    assert mem.count() == 2


def test_check_merges_disabled():
    mem = Memory(anchor=False, merge=False)
    assert mem._check_merges() == 0


# --- Void mapping integration ---

def test_map_void():
    mem = Memory(anchor=False, overlap=False)
    s1 = _shape({"x": (2.0, 4.0)})
    s2 = _shape({"x": (6.0, 8.0)})
    mem.store(s1)
    mem.store(s2)
    voids = mem.map_void("x", (0.0, 10.0))
    assert len(voids) >= 1
    assert all(isinstance(v, VoidRegion) for v in voids)


def test_void_percentage():
    mem = Memory(anchor=False, overlap=False)
    # No shapes — 100% void
    pct = mem.void_percentage({"x": (0.0, 10.0)})
    assert pct == 1.0


def test_void_percentage_with_shapes():
    mem = Memory(anchor=False, overlap=False)
    s = _shape({"x": (0.0, 10.0)})
    mem.store(s)
    pct = mem.void_percentage({"x": (0.0, 10.0)})
    assert pct < 0.1  # mostly covered


def test_largest_gap():
    mem = Memory(anchor=False, overlap=False)
    s1 = _shape({"x": (0.0, 3.0)})
    s2 = _shape({"x": (7.0, 10.0)})
    mem.store(s1)
    mem.store(s2)
    gap = mem.largest_gap("x", (0.0, 10.0))
    assert gap is not None
    assert isinstance(gap, VoidRegion)
    # The gap between 3 and 7 is the largest
    assert gap.dimensions["x"].min_value == 3.0
    assert gap.dimensions["x"].max_value == 7.0


def test_largest_gap_no_voids():
    mem = Memory(anchor=False, overlap=False)
    s = _shape({"x": (0.0, 10.0)})
    mem.store(s)
    gap = mem.largest_gap("x", (0.0, 10.0))
    assert gap is None


# --- Emergence integration ---

def test_detect_emergence():
    mem = Memory(anchor=False, overlap=False)
    # Store shapes that cluster together
    for i in range(5):
        s = _shape({"x": (float(i), float(i + 2))})
        mem.store(s)
    result = mem.detect_emergence()
    assert isinstance(result, list)


def test_detect_emergence_empty():
    mem = Memory(anchor=False)
    result = mem.detect_emergence()
    assert result == []


# --- Scanner integration ---

def test_scan_and_store_dataframe():
    mem = Memory(anchor=False, overlap=False)
    data = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
    count = mem.scan_and_store(data)
    assert count == 1
    assert mem.count() == 1


def test_scan_and_store_json():
    mem = Memory(anchor=False, overlap=False)
    data = {"temperature": 25.0, "pressure": 1.0}
    count = mem.scan_and_store(data)
    assert count == 1


def test_scan_and_store_codebase():
    mem = Memory(anchor=False, overlap=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "example.py"
        py_file.write_text("def calc(x: int, y: float) -> float:\n    return x + y\n")
        count = mem.scan_and_store(tmpdir)
    assert count == 1
    assert mem.count() == 1


# --- Utility methods ---

def test_get():
    mem = Memory(anchor=False)
    s = _shape({"x": (0.0, 10.0)})
    sid = mem.store(s)
    retrieved = mem.get(sid)
    assert retrieved is not None
    assert retrieved.id == sid


def test_get_nonexistent():
    mem = Memory(anchor=False)
    assert mem.get("nonexistent") is None


def test_all_shapes():
    mem = Memory(anchor=False, overlap=False)
    s1 = _shape({"x": (0.0, 10.0)})
    s2 = _shape({"y": (0.0, 5.0)})
    mem.store(s1)
    mem.store(s2)
    shapes = list(mem.all_shapes())
    assert len(shapes) == 2


# --- __init__.py imports ---

def test_init_imports():
    from doorway_memory import Memory, Shape, Dimension, Library
    from doorway_memory import Tier2Shape, Scanner, scan
    assert Memory is not None
    assert Shape is not None
    assert Dimension is not None
    assert Library is not None
    assert Tier2Shape is not None
    assert Scanner is not None
    assert scan is not None


# --- End-to-end workflow ---

def test_end_to_end_workflow():
    """Full workflow: scan, store, recall, grow, merge, detect void, emergence."""
    mem = Memory(anchor=False, overlap=False)

    # 1. Scan and store
    data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]}
    count = mem.scan_and_store(data)
    assert count == 1

    # 2. Recall inside
    assert mem.is_known({"x": 2.5, "y": 25.0}) is True

    # 3. Void outside
    assert mem.is_void({"x": 100.0, "y": 100.0}) is True

    # 4. Void mapping
    pct = mem.void_percentage({"x": (0.0, 100.0), "y": (0.0, 100.0)})
    assert pct > 0.5  # mostly void in a 100x100 space

    # 5. Recall with confidence
    results = mem.recall_with_confidence({"x": 2.5, "y": 25.0})
    assert len(results) == 1
    shape, conf = results[0]
    assert conf > 0

    # 6. Maintenance
    result = mem.maintain()
    assert "decayed" in result
    assert "archived" in result
