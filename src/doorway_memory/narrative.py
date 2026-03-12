"""Trajectories, prediction, and common paths.

A trajectory is a time-ordered sequence of points through dimensional
space. Narratives track how a system moves through known territory
and void, enabling prediction of likely next positions and detection
of common movement patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Minimum trajectory length for prediction
MIN_PREDICTION_LENGTH = 2

# Default number of recent points used for velocity estimation
VELOCITY_WINDOW = 3

# Similarity threshold for common path detection (cosine similarity)
PATH_SIMILARITY_THRESHOLD = 0.8


@dataclass
class Trajectory:
    """A time-ordered sequence of points through dimensional space."""
    id: str
    points: List[Dict[str, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    metadata: Optional[Dict] = None

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def dimensions(self) -> List[str]:
        """All dimension names seen across points."""
        dims = set()
        for p in self.points:
            dims.update(p.keys())
        return sorted(dims)


def record_point(
    trajectory: Trajectory,
    point: Dict[str, float],
    timestamp: float,
) -> None:
    """Add a point to a trajectory."""
    trajectory.points.append(point)
    trajectory.timestamps.append(timestamp)


def _points_to_matrix(
    points: List[Dict[str, float]], dims: List[str]
) -> np.ndarray:
    """Convert list of point dicts to a numpy matrix (points x dims)."""
    matrix = np.zeros((len(points), len(dims)))
    for i, p in enumerate(points):
        for j, d in enumerate(dims):
            matrix[i, j] = p.get(d, 0.0)
    return matrix


def estimate_velocity(
    trajectory: Trajectory, window: int = VELOCITY_WINDOW
) -> Dict[str, float]:
    """
    Estimate current velocity (rate of change per unit time) from recent points.

    Uses the last `window` points to compute average velocity.
    Returns a dict mapping dimension name to velocity.
    """
    if trajectory.length < 2:
        return {}

    n = min(window, trajectory.length)
    recent_points = trajectory.points[-n:]
    recent_times = trajectory.timestamps[-n:]

    dims = trajectory.dimensions
    matrix = _points_to_matrix(recent_points, dims)

    dt = recent_times[-1] - recent_times[0]
    if dt == 0.0:
        return {d: 0.0 for d in dims}

    deltas = matrix[-1] - matrix[0]
    velocity = deltas / dt

    return {d: float(velocity[i]) for i, d in enumerate(dims)}


def predict_next(
    trajectory: Trajectory,
    dt: float = 1.0,
    window: int = VELOCITY_WINDOW,
) -> Optional[Dict[str, float]]:
    """
    Predict the next point based on current velocity.

    Uses linear extrapolation from the last point plus velocity * dt.
    Returns None if trajectory is too short for prediction.
    """
    if trajectory.length < MIN_PREDICTION_LENGTH:
        return None

    velocity = estimate_velocity(trajectory, window)
    if not velocity:
        return None

    last_point = trajectory.points[-1]
    predicted = {}
    for dim in trajectory.dimensions:
        predicted[dim] = last_point.get(dim, 0.0) + velocity.get(dim, 0.0) * dt

    return predicted


def trajectory_distance(
    traj_a: Trajectory, traj_b: Trajectory
) -> float:
    """
    Compute distance between two trajectories using mean point-wise distance.

    Trajectories are resampled to the same length (shorter one's length).
    """
    if traj_a.length == 0 or traj_b.length == 0:
        return float('inf')

    dims = sorted(set(traj_a.dimensions) | set(traj_b.dimensions))

    # Use the shorter trajectory's length
    n = min(traj_a.length, traj_b.length)

    # Sample evenly from each trajectory
    idx_a = np.linspace(0, traj_a.length - 1, n, dtype=int)
    idx_b = np.linspace(0, traj_b.length - 1, n, dtype=int)

    mat_a = _points_to_matrix([traj_a.points[i] for i in idx_a], dims)
    mat_b = _points_to_matrix([traj_b.points[i] for i in idx_b], dims)

    distances = np.linalg.norm(mat_a - mat_b, axis=1)
    return float(np.mean(distances))


def find_common_paths(
    trajectories: List[Trajectory],
    distance_threshold: float = 5.0,
) -> List[Tuple[Trajectory, Trajectory, float]]:
    """
    Find pairs of trajectories that follow similar paths.

    Returns list of (traj_a, traj_b, distance) tuples for pairs
    whose mean point-wise distance is below the threshold.
    """
    results = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            dist = trajectory_distance(trajectories[i], trajectories[j])
            if dist <= distance_threshold:
                results.append((trajectories[i], trajectories[j], dist))
    return results


def trajectory_direction(trajectory: Trajectory) -> Dict[str, float]:
    """
    Compute the overall direction vector of a trajectory (first to last point).

    Returns a unit vector as a dict, or empty dict if trajectory has < 2 points.
    """
    if trajectory.length < 2:
        return {}

    dims = trajectory.dimensions
    first = _points_to_matrix([trajectory.points[0]], dims)[0]
    last = _points_to_matrix([trajectory.points[-1]], dims)[0]

    direction = last - first
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return {d: 0.0 for d in dims}

    unit = direction / norm
    return {d: float(unit[i]) for i, d in enumerate(dims)}
