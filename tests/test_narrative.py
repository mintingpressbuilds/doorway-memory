"""Tests for narrative.py — trajectories, prediction, and common paths."""

import math

from doorway_memory.narrative import (
    Trajectory,
    record_point,
    estimate_velocity,
    predict_next,
    trajectory_distance,
    find_common_paths,
    trajectory_direction,
)


def _make_linear_trajectory(
    tid: str, start: float, step: float, n: int
) -> Trajectory:
    """Create a trajectory with linearly increasing x values."""
    t = Trajectory(id=tid)
    for i in range(n):
        record_point(t, {"x": start + step * i}, float(i))
    return t


def test_record_point():
    t = Trajectory(id="t1")
    record_point(t, {"x": 1.0, "y": 2.0}, 0.0)
    assert t.length == 1
    assert t.points[0] == {"x": 1.0, "y": 2.0}
    assert t.timestamps[0] == 0.0


def test_trajectory_length():
    t = Trajectory(id="t1")
    assert t.length == 0
    record_point(t, {"x": 1.0}, 0.0)
    assert t.length == 1


def test_trajectory_dimensions():
    t = Trajectory(id="t1")
    record_point(t, {"x": 1.0, "y": 2.0}, 0.0)
    record_point(t, {"x": 3.0, "z": 4.0}, 1.0)
    assert set(t.dimensions) == {"x", "y", "z"}


def test_estimate_velocity_linear():
    t = _make_linear_trajectory("t1", 0.0, 2.0, 5)
    # x goes 0, 2, 4, 6, 8 over times 0, 1, 2, 3, 4
    v = estimate_velocity(t)
    assert abs(v["x"] - 2.0) < 1e-10


def test_estimate_velocity_empty():
    t = Trajectory(id="t1")
    assert estimate_velocity(t) == {}


def test_estimate_velocity_single_point():
    t = Trajectory(id="t1")
    record_point(t, {"x": 5.0}, 0.0)
    assert estimate_velocity(t) == {}


def test_estimate_velocity_same_time():
    t = Trajectory(id="t1")
    record_point(t, {"x": 0.0}, 0.0)
    record_point(t, {"x": 5.0}, 0.0)
    v = estimate_velocity(t)
    assert v["x"] == 0.0


def test_estimate_velocity_window():
    t = Trajectory(id="t1")
    # First 3 points going slowly, last 3 going fast
    for i in range(3):
        record_point(t, {"x": float(i)}, float(i))
    for i in range(3):
        record_point(t, {"x": 2.0 + 10.0 * (i + 1)}, float(3 + i))
    # With window=3, only the last 3 points matter
    v = estimate_velocity(t, window=3)
    assert v["x"] > 5.0  # Should reflect fast movement


def test_predict_next_linear():
    t = _make_linear_trajectory("t1", 0.0, 2.0, 5)
    predicted = predict_next(t, dt=1.0)
    # Velocity is 2.0, last point is x=8.0, so next should be ~10.0
    assert predicted is not None
    assert abs(predicted["x"] - 10.0) < 1e-10


def test_predict_next_too_short():
    t = Trajectory(id="t1")
    record_point(t, {"x": 1.0}, 0.0)
    assert predict_next(t) is None


def test_predict_next_empty():
    t = Trajectory(id="t1")
    assert predict_next(t) is None


def test_predict_next_custom_dt():
    t = _make_linear_trajectory("t1", 0.0, 3.0, 4)
    # velocity = 3.0, last x = 9.0
    predicted = predict_next(t, dt=2.0)
    assert predicted is not None
    assert abs(predicted["x"] - 15.0) < 1e-10


def test_predict_next_2d():
    t = Trajectory(id="t1")
    record_point(t, {"x": 0.0, "y": 0.0}, 0.0)
    record_point(t, {"x": 1.0, "y": 2.0}, 1.0)
    record_point(t, {"x": 2.0, "y": 4.0}, 2.0)
    predicted = predict_next(t, dt=1.0)
    assert abs(predicted["x"] - 3.0) < 1e-10
    assert abs(predicted["y"] - 6.0) < 1e-10


def test_trajectory_distance_identical():
    t1 = _make_linear_trajectory("t1", 0.0, 1.0, 5)
    t2 = _make_linear_trajectory("t2", 0.0, 1.0, 5)
    assert trajectory_distance(t1, t2) == 0.0


def test_trajectory_distance_offset():
    t1 = _make_linear_trajectory("t1", 0.0, 1.0, 5)
    t2 = _make_linear_trajectory("t2", 10.0, 1.0, 5)
    dist = trajectory_distance(t1, t2)
    assert dist == 10.0  # Constant 10-unit offset


def test_trajectory_distance_empty():
    t1 = Trajectory(id="t1")
    t2 = _make_linear_trajectory("t2", 0.0, 1.0, 5)
    assert trajectory_distance(t1, t2) == float('inf')


def test_find_common_paths_similar():
    t1 = _make_linear_trajectory("t1", 0.0, 1.0, 5)
    t2 = _make_linear_trajectory("t2", 0.5, 1.0, 5)
    results = find_common_paths([t1, t2], distance_threshold=1.0)
    assert len(results) == 1


def test_find_common_paths_different():
    t1 = _make_linear_trajectory("t1", 0.0, 1.0, 5)
    t2 = _make_linear_trajectory("t2", 100.0, 1.0, 5)
    results = find_common_paths([t1, t2], distance_threshold=5.0)
    assert len(results) == 0


def test_find_common_paths_multiple():
    t1 = _make_linear_trajectory("t1", 0.0, 1.0, 5)
    t2 = _make_linear_trajectory("t2", 0.1, 1.0, 5)
    t3 = _make_linear_trajectory("t3", 0.2, 1.0, 5)
    results = find_common_paths([t1, t2, t3], distance_threshold=1.0)
    # All three are similar, so 3 pairs
    assert len(results) == 3


def test_find_common_paths_empty():
    assert find_common_paths([]) == []


def test_trajectory_direction_linear():
    t = _make_linear_trajectory("t1", 0.0, 5.0, 3)
    direction = trajectory_direction(t)
    assert abs(direction["x"] - 1.0) < 1e-10  # Unit vector pointing positive x


def test_trajectory_direction_negative():
    t = Trajectory(id="t1")
    record_point(t, {"x": 10.0}, 0.0)
    record_point(t, {"x": 0.0}, 1.0)
    direction = trajectory_direction(t)
    assert abs(direction["x"] - (-1.0)) < 1e-10


def test_trajectory_direction_2d():
    t = Trajectory(id="t1")
    record_point(t, {"x": 0.0, "y": 0.0}, 0.0)
    record_point(t, {"x": 3.0, "y": 4.0}, 1.0)
    direction = trajectory_direction(t)
    # Direction should be unit vector [3/5, 4/5]
    assert abs(direction["x"] - 0.6) < 1e-10
    assert abs(direction["y"] - 0.8) < 1e-10


def test_trajectory_direction_single_point():
    t = Trajectory(id="t1")
    record_point(t, {"x": 5.0}, 0.0)
    assert trajectory_direction(t) == {}


def test_trajectory_direction_stationary():
    t = Trajectory(id="t1")
    record_point(t, {"x": 5.0, "y": 3.0}, 0.0)
    record_point(t, {"x": 5.0, "y": 3.0}, 1.0)
    direction = trajectory_direction(t)
    assert direction["x"] == 0.0
    assert direction["y"] == 0.0
