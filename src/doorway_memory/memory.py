"""High-level geometric memory API.

Orchestrates all geometric memory mechanics: storage, recall, growth,
overlap detection, decay, merge, narrative trajectories, void mapping,
emergence detection, and scanning.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from .shape import Shape, Dimension, extract_point
from .library import Library
from .growth import try_grow, detect_near_miss
from .overlap import find_overlap, pairwise_overlaps
from .decay import decay_shape, should_archive, archive_shape
from .merge import merge_all, find_merge_candidates, merge_shapes
from .narrative import (
    Trajectory, record_point, estimate_velocity,
    predict_next as _predict_next, find_common_paths as _find_common_paths,
)
from .void_map import (
    VoidRegion, find_void_regions_1d, void_density, nearest_void,
)
from .emergence import (
    Tier2Shape, detect_tier2, geometric_coherence_score,
)

try:
    import xycore
    HAS_XYCORE = True
except ImportError:
    HAS_XYCORE = False


class Memory:
    """
    Geometric memory engine.

    Integrates all mechanics: growth, overlap, decay, merge,
    narrative, void mapping, emergence detection, and scanning.
    """

    def __init__(self, namespace="default", backend="memory",
                 anchor=True, path=None, config=None,
                 growth=True, overlap=True, decay=True,
                 merge=True, narrative=True):
        self.namespace = namespace
        self.library = Library(backend=backend, path=path, config=config)
        self.anchor_enabled = anchor and HAS_XYCORE
        self._chain_id = f"memory:{namespace}"

        self.growth_enabled = growth
        self.overlap_enabled = overlap
        self.decay_enabled = decay
        self.merge_enabled = merge
        self.narrative_enabled = narrative

        self._trajectories: Dict[str, Trajectory] = {}
        self._archived: List[Shape] = []

    # --- Core API ---

    def store(self, shape: Shape, metadata=None) -> str:
        """Store shape. Returns shape ID."""
        if metadata:
            shape.metadata = {**(shape.metadata or {}), **metadata}

        if self.anchor_enabled:
            anchor = xycore.Anchor.create(
                data=shape.to_dict(), chain_id=self._chain_id
            )
            shape.anchor_id = anchor.id

        shape_id = self.library.add(shape)

        if self.overlap_enabled:
            all_shapes = list(self.library.all())
            for other in all_shapes:
                if other.id == shape_id:
                    continue
                overlap = find_overlap(shape, other)
                if overlap is not None:
                    overlap.metadata = {
                        **(overlap.metadata or {}),
                        "overlap_parents": [shape.id, other.id],
                    }

        return shape_id

    def recall(self, state: Dict, raw=False) -> List[Shape]:
        """Find shapes containing state. Set raw=True for unextracted input."""
        point = extract_point(state) if raw else state
        results = self.library.query(point)

        if self.growth_enabled and not results:
            all_shapes = list(self.library.all())
            for shape in all_shapes:
                grown = try_grow(shape, point)
                if grown is not None:
                    self.library.remove(shape.id)
                    self.library.add(grown)
                    results.append(grown)
                    if self.merge_enabled:
                        self._check_merges()
                    break

        return results

    def recall_with_confidence(self, state: Dict, raw=False) -> List[Tuple[Shape, float]]:
        """Find shapes containing state, with confidence scores."""
        point = extract_point(state) if raw else state
        shapes = self.library.query(point)
        return [(s, s.confidence) for s in shapes]

    def is_known(self, state: Dict, raw=False) -> bool:
        """True if state is inside known territory."""
        point = extract_point(state) if raw else state
        return len(self.library.query(point)) > 0

    def is_void(self, state: Dict, raw=False) -> bool:
        """True if state is in void (unknown territory)."""
        return not self.is_known(state, raw=raw)

    # --- Verification (xycore) ---

    def verify(self, shape_id: str) -> Optional[Dict]:
        """Get verification proof for shape."""
        if not self.anchor_enabled:
            raise ImportError("Verification requires xycore.")

        shape = self.library.get(shape_id)
        if not shape or not shape.anchor_id:
            return None

        anchor = xycore.Anchor.get(shape.anchor_id)
        return {
            "anchor_id": anchor.id,
            "timestamp": anchor.timestamp,
            "hash": anchor.hash,
            "chain_id": self._chain_id,
            "verified": xycore.Verify.check(anchor)
        }

    def replay(self, start=None, end=None) -> Iterator[Shape]:
        """Walk memory history, yielding shapes in order."""
        if not self.anchor_enabled:
            raise ImportError("Replay requires xycore.")

        for anchor in xycore.walk(self._chain_id, start=start, end=end):
            shape = Shape.from_dict(anchor.data)
            shape.anchor_id = anchor.id
            yield shape

    # --- Narrative ---

    def store_in_trajectory(self, shape: Shape, trajectory_id: str,
                            metadata=None, timestamp: float = 0.0) -> str:
        """Store shape and record it in a trajectory."""
        shape_id = self.store(shape, metadata=metadata)

        if self.narrative_enabled:
            if trajectory_id not in self._trajectories:
                self._trajectories[trajectory_id] = Trajectory(
                    id=trajectory_id,
                )
            traj = self._trajectories[trajectory_id]
            point = {}
            for name, dim in shape.dimensions.items():
                point[name] = (dim.min_value + dim.max_value) / 2.0
            record_point(traj, point, timestamp)

        return shape_id

    def predict_next(self, trajectory_id: str, dt: float = 1.0) -> Optional[Dict[str, float]]:
        """Predict next point in a trajectory."""
        if not self.narrative_enabled:
            return None
        traj = self._trajectories.get(trajectory_id)
        if traj is None:
            return None
        return _predict_next(traj, dt=dt)

    def find_common_paths(self, distance_threshold: float = 5.0) -> List[Tuple[Trajectory, Trajectory, float]]:
        """Find similar trajectory pairs."""
        if not self.narrative_enabled:
            return []
        trajs = list(self._trajectories.values())
        return _find_common_paths(trajs, distance_threshold=distance_threshold)

    # --- Decay ---

    def maintain(self) -> Dict[str, int]:
        """Run maintenance cycle: decay and archive stale shapes."""
        if not self.decay_enabled:
            return {"decayed": 0, "archived": 0}

        all_shapes = list(self.library.all())
        decayed_count = 0
        archived_count = 0

        for shape in all_shapes:
            decayed = decay_shape(shape)
            if should_archive(decayed):
                archived = archive_shape(decayed)
                self._archived.append(archived)
                self.library.remove(shape.id)
                archived_count += 1
            elif decayed.id != shape.id:
                self.library.remove(shape.id)
                self.library.add(decayed)
                decayed_count += 1

        return {"decayed": decayed_count, "archived": archived_count}

    # --- Merge ---

    def _check_merges(self) -> int:
        """Run merge detection on all shapes. Returns count of merges performed."""
        if not self.merge_enabled:
            return 0

        all_shapes = list(self.library.all())
        merged = merge_all(all_shapes)

        if len(merged) < len(all_shapes):
            merge_count = len(all_shapes) - len(merged)
            for shape in all_shapes:
                self.library.remove(shape.id)
            for shape in merged:
                self.library.add(shape)
            return merge_count
        return 0

    # --- Void mapping ---

    def map_void(self, dimension: str, bounds: Tuple[float, float]) -> List[VoidRegion]:
        """Find void regions along a single dimension."""
        all_shapes = list(self.library.all())
        return find_void_regions_1d(all_shapes, dimension, bounds)

    def void_percentage(self, bounds: Dict[str, Tuple[float, float]],
                        resolution: int = 10) -> float:
        """Estimate fraction of bounded space that is void (0.0–1.0)."""
        all_shapes = list(self.library.all())
        return void_density(all_shapes, bounds, resolution=resolution)

    def largest_gap(self, dimension: str,
                    bounds: Tuple[float, float]) -> Optional[VoidRegion]:
        """Find the largest void region along a dimension."""
        regions = self.map_void(dimension, bounds)
        if not regions:
            return None
        return max(regions, key=lambda r: r.volume())

    # --- Emergence ---

    def detect_emergence(self, gcs_threshold: float = 0.3,
                         is_threshold: float = 0.1,
                         min_cluster_size: int = 2) -> List[Tier2Shape]:
        """Detect Tier 2 emergent patterns in the library."""
        all_shapes = list(self.library.all())
        return detect_tier2(
            all_shapes,
            gcs_threshold=gcs_threshold,
            is_threshold=is_threshold,
            min_cluster_size=min_cluster_size,
        )

    # --- Scanner ---

    def scan_and_store(self, source: Any, name: str = "auto") -> int:
        """Scan a data source and store all extracted shapes. Returns count stored."""
        from .scanner import scan
        result = scan(source, name=name)
        for shape in result.shapes:
            self.store(shape)
        return len(result.shapes)

    # --- Utilities ---

    def count(self) -> int:
        return self.library.count()

    def get(self, shape_id: str) -> Optional[Shape]:
        return self.library.get(shape_id)

    def all_shapes(self) -> Iterator[Shape]:
        return self.library.all()
