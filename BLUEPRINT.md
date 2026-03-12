# BLUEPRINT.md — doorway-memory

## Geometric Memory Engine

**Package:** doorway-memory
**Version:** 0.1.0
**Author:** Luke H
**License:** Apache 2.0

-----

## Build Contract

This blueprint defines exactly what to build, in what order, with verification criteria at each phase. Do not skip phases. Do not start Phase N+1 until Phase N tests pass.

-----

## Phase 1 — Base Layer

**Goal:** A working memory system that stores shapes and retrieves by containment. No advanced mechanics. No emergence. Just the foundation.

### 1.1 shape.py

**Build:**

- `Dimension` dataclass: `name: str`, `min_value: float`, `max_value: float`
- `Shape` dataclass: `dimensions: Dict[str, Dimension]`, `metadata: Optional[Dict]`, `id: Optional[str]`, `anchor_id: Optional[str]`
- `Shape.__post_init__()`: auto-generate ID from dimension hash if not provided
- `Shape._generate_id()`: SHA-256 of sorted dimension structure, truncated to 16 chars
- `Shape.contains(point: Dict[str, float]) -> bool`: True if point within bounds on all shared dimensions
- `Shape.distance_to_boundary(point: Dict[str, float]) -> float`: positive if inside, negative if outside
- `Shape.to_dict() -> Dict`: serialize for storage
- `Shape.from_dict(data: Dict) -> Shape`: deserialize
- `extract_point(input_data: Dict) -> Dict[str, float]`: extract dimensional signature from raw input
- Helper functions: `_encode_structure()`, `_encode_implication()`, `_compute_complexity()`

**Verify:**

```
pytest tests/test_shape.py -v
```

- test_contains_point_inside → True
- test_contains_point_outside → False
- test_contains_point_on_boundary → True
- test_distance_inside → positive float
- test_distance_outside → negative float
- test_to_dict_from_dict_roundtrip → equal
- test_id_generation_deterministic → same dims produce same ID

### 1.2 intersect.py

**Build:**

- `point_in_shape(point, shape) -> bool`: wrapper for shape.contains()
- `find_containing_shapes(point, shapes) -> List[Shape]`: all shapes containing point
- `find_nearest_shapes(point, shapes, limit) -> List[Tuple[Shape, float]]`: sorted by distance
- `find_void(point, shapes) -> bool`: True if no shape contains point

**Verify:**

```
pytest tests/test_intersect.py -v
```

- test_find_containing_two_overlapping → returns both
- test_find_containing_one_only → returns one
- test_find_void_true → point outside all shapes
- test_find_void_false → point inside at least one
- test_find_nearest_sorted → sorted by distance descending

### 1.3 library.py

**Build:**

- `Library` class with `backend` parameter: “memory”, “file”, “supabase”
- `Library.add(shape) -> str`: store shape, return ID
- `Library.get(shape_id) -> Optional[Shape]`: retrieve by ID
- `Library.remove(shape_id) -> bool`: remove, return True if existed
- `Library.query(point) -> List[Shape]`: find all containing shapes
- `Library.is_void(point) -> bool`: True if no shapes contain point
- `Library.all() -> Iterator[Shape]`: iterate all shapes
- `Library.count() -> int`: shape count
- File backend: JSON persistence with `_load_from_file()` / `_save_to_file()`
- Supabase backend: stub methods (implementation later)

**Verify:**

```
pytest tests/test_library.py -v
```

- test_add_and_get → roundtrip works
- test_remove → returns True, get returns None
- test_query_containment → correct shapes returned
- test_is_void → True/False correct
- test_count → accurate after add/remove
- test_file_backend_persistence → save, reload, data intact

### 1.4 anchor.py

**Build:**

- `anchor_shape(shape_data, chain_id) -> Optional[str]`: anchor to xycore, return anchor ID
- `verify_anchor(anchor_id) -> Optional[Dict]`: verify integrity
- `generate_receipt(anchor_id) -> Optional[str]`: shareable receipt
- All functions return None if xycore not installed
- Import wrapped in try/except

**Verify:**

```
pytest tests/test_anchor.py -v
```

- test_no_xycore_returns_none → graceful without xycore

### 1.5 memory.py

**Build:**

- `Memory` class: `namespace`, `backend`, `anchor`, `path`, `config` parameters
- `Memory.store(shape, metadata) -> str`: store shape, optionally anchor
- `Memory.recall(state, raw) -> List[Shape]`: find containing shapes
- `Memory.is_known(state, raw) -> bool`: True if inside known territory
- `Memory.is_void(state, raw) -> bool`: True if in void
- `Memory.verify(shape_id) -> Optional[Dict]`: verification proof
- `Memory.replay(start, end) -> Iterator[Shape]`: walk chain history
- `Memory.count() -> int`: shape count

**Verify:**

```
pytest tests/test_memory.py -v
```

- test_store_and_recall → store shape, recall point inside it
- test_void_detection → point outside all shapes returns True
- test_is_known → point inside returns True
- test_count → accurate
- test_works_without_xycore → anchor=False, full functionality

### Phase 1 Gate

```bash
pytest tests/test_shape.py tests/test_intersect.py tests/test_library.py tests/test_memory.py -v
```

**All tests must pass before proceeding to Phase 2.**

-----

## Phase 2 — Advanced Mechanics

**Goal:** The library becomes alive. Shapes grow, overlap detection finds cross-domain patterns, unused knowledge decays, and trajectories form narratives.

### 2.1 growth.py

**Build:**

- `NEAR_MISS_MARGIN = 0.15`: relative to dimension range
- `GROWTH_THRESHOLD = 5`: near-misses before expansion
- `MAX_EXPANSION_RATIO = 0.25`: maximum single expansion
- `GrowthTracker` class: tracks near-misses per shape per dimension
- `GrowthTracker.record_query(point, shape) -> Optional[Shape]`: record query, return expanded shape if growth triggered
- `GrowthTracker.get_near_miss_count(shape_id) -> Dict[str, int]`: current counts

**Integration:** Memory.recall() calls growth tracker on non-matching shapes. If growth returns expanded shape, old shape removed, new shape stored.

**Verify:**

```
pytest tests/test_growth.py -v
```

- test_no_growth_inside → point inside, no expansion
- test_near_miss_recorded → point just outside, count increases
- test_growth_triggers → after threshold near-misses, shape expands
- test_expansion_capped → expansion doesn’t exceed MAX_EXPANSION_RATIO

### 2.2 overlap.py

**Build:**

- `compute_overlap(shape_a, shape_b) -> Optional[Shape]`: intersection region
- `compute_overlap_volume(overlap) -> float`: volume of intersection
- `find_all_overlaps(shapes, min_volume) -> List[Shape]`: all pairwise overlaps
- `find_overlaps_for_shape(shape, library_shapes) -> List[Shape]`: overlaps for new shape

**Integration:** Memory.store() checks new shape against library for overlaps. Overlap shapes stored with parent metadata.

**Verify:**

```
pytest tests/test_overlap.py -v
```

- test_overlap_exists → two overlapping shapes produce intersection
- test_no_overlap → non-overlapping shapes return None
- test_overlap_volume → correct volume calculation
- test_overlap_metadata → parent IDs preserved

### 2.3 decay.py

**Build:**

- `DECAY_GRACE_PERIOD = 604800`: 7 days before decay starts
- `DECAY_RATE = 0.02`: 2% shrink per cycle
- `MIN_DIMENSION_RANGE = 0.01`: collapse threshold
- `ARCHIVE_VOLUME_THRESHOLD = 0.001`: archive threshold
- `DecayTracker` class: tracks access times and counts
- `DecayTracker.record_access(shape_id)`: mark shape as accessed
- `DecayTracker.record_creation(shape_id)`: mark shape as created
- `DecayTracker.apply_decay(shapes) -> Tuple[List[Shape], List[Shape]]`: returns (active, archived)
- Access resistance: frequently accessed shapes decay slower (up to 90% resistance)

**Integration:** Memory.maintain() runs decay cycle. Archived shapes removed from library, anchored to chain.

**Verify:**

```
pytest tests/test_decay.py -v
```

- test_no_decay_in_grace_period → shape unchanged during grace period
- test_decay_shrinks_boundaries → boundaries contract after grace period
- test_access_resistance → frequently accessed shapes shrink less
- test_archive_threshold → shape archived when volume too small

### 2.4 narrative.py

**Build:**

- `Trajectory` dataclass: `id`, `name`, ordered list of shape IDs, transition metadata
- `Trajectory.add_step(shape_id, metadata)`: append step
- `Trajectory.steps() -> List[str]`: shape IDs in order
- `Trajectory.to_dict()` / `from_dict()`: serialization
- `NarrativeEngine` class: manages trajectories and transition frequencies
- `NarrativeEngine.start_trajectory(id, name) -> Trajectory`
- `NarrativeEngine.record_step(trajectory_id, shape_id, metadata)`
- `NarrativeEngine.predict_next(current_shape_id, limit) -> List[Tuple[str, float]]`: prediction by frequency
- `NarrativeEngine.find_common_trajectories(min_length, min_occurrences) -> List[List[str]]`
- `NarrativeEngine.get_transition_map() -> Dict[str, Dict[str, int]]`

**Integration:** Memory.store_in_trajectory(), Memory.predict_next(), Memory.find_common_paths()

**Verify:**

```
pytest tests/test_narrative.py -v
```

- test_trajectory_add_steps → steps in order
- test_predict_next → most frequent transition returned first
- test_common_trajectories → shared subsequences detected
- test_transition_map → correct frequencies

### 2.5 Confidence additions to shape.py

**Build:**

- `Shape.confidence(point) -> float`: 0.0 at boundary, 1.0 at center
- `Shape.confidence_breakdown(point) -> Dict[str, float]`: per-dimension confidence

**Integration:** Memory.recall_with_confidence() returns (shape, confidence) tuples.

**Verify:**

```
pytest tests/test_shape.py -v -k confidence
```

- test_confidence_center → 1.0
- test_confidence_boundary → 0.0
- test_confidence_outside → 0.0
- test_confidence_midpoint → ~0.5
- test_confidence_breakdown → per-dim values correct

### Phase 2 Gate

```bash
pytest tests/ -v
```

**All tests must pass before proceeding to Phase 3.**

-----

## Phase 3 — Structural Layer

**Goal:** Shapes merge when they meet. The void is mapped as geometry. Tier 2 patterns emerge from the library. GCS and IS are detectable.

### 3.1 merge.py

**Build:**

- `MERGE_OVERLAP_RATIO = 0.50`: overlap must exceed 50% of smaller shape
- `compute_volume(shape) -> float`: total shape volume
- `should_merge(shape_a, shape_b, threshold) -> bool`: merge check
- `merge_shapes(shape_a, shape_b) -> Shape`: union envelope with parent metadata
- `MergeDetector` class: finds and applies merges
- `MergeDetector.find_merge_candidates(shapes) -> List[Tuple[Shape, Shape]]`
- `MergeDetector.apply_merges(shapes) -> Tuple[List[Shape], List[Shape]]`: returns (active, archived)

**Integration:** Memory._check_merges() runs after growth events.

**Verify:**

```
pytest tests/test_merge.py -v
```

- test_should_merge_high_overlap → True
- test_should_not_merge_low_overlap → False
- test_merge_shapes_union → merged boundaries are outer envelope
- test_merge_metadata_preserved → parent IDs in metadata
- test_apply_merges → parents archived, merged shape in active list

### 3.2 void_map.py

**Build:**

- `VoidRegion` dataclass: dimensions, neighboring_shapes, volume(), center()
- `VoidMapper` class with configurable resolution
- `VoidMapper.compute_bounds(shapes, margin) -> Dict[str, Dimension]`: library bounding box
- `VoidMapper.sample_void(shapes, bounds, sample_count) -> List[Dict]`: random void points
- `VoidMapper.find_void_regions(shapes, bounds, sample_count, cluster_threshold) -> List[VoidRegion]`
- `VoidMapper.void_percentage(shapes, bounds, sample_count) -> float`
- `VoidMapper.largest_void(shapes, bounds) -> Optional[VoidRegion]`
- Uses numpy for sampling and distance-based clustering

**Integration:** Memory.map_void(), Memory.void_percentage(), Memory.largest_gap()

**Verify:**

```
pytest tests/test_void_map.py -v
```

- test_void_percentage_empty_library → 1.0 (all void)
- test_void_percentage_full_coverage → ~0.0
- test_void_regions_detected → gaps between shapes found
- test_void_neighbors → correct neighboring shapes identified
- test_largest_void → biggest region returned first

### 3.3 emergence.py

**Build:**

- `Tier2Shape` dataclass: shape, parent_ids, domains, strength
- `EmergenceDetector` class: `min_shapes=3`, `min_domains=2`, `volume_threshold=0.001`
- `EmergenceDetector.detect(shapes) -> List[Tier2Shape]`: find all Tier 2 patterns
- `EmergenceDetector._find_subspace_groups(shapes) -> Dict[str, List[Shape]]`: group by shared dimensions
- `EmergenceDetector._compute_multi_intersection(shapes, dim_key) -> Optional[Shape]`: progressive intersection
- `EmergenceDetector._extract_domains(shapes) -> List[str]`: from metadata
- `EmergenceDetector.detect_gcs(shapes, growth_history) -> Optional[Tier2Shape]`: growth pattern self-recognition
- `EmergenceDetector.detect_is(shapes) -> Optional[Tier2Shape]`: intelligence mechanism self-recognition

**Integration:** Memory.detect_emergence(), Memory.detect_gcs(), Memory.detect_is()

**Verify:**

```
pytest tests/test_emergence.py -v
```

- test_no_emergence_few_shapes → empty list with < 3 shapes
- test_tier2_detected → 3+ shapes sharing dimensions produce Tier2Shape
- test_tier2_requires_min_domains → single domain filtered out
- test_multi_intersection → progressive intersection correct
- test_gcs_not_detected_small_history → None with < 10 events
- test_is_not_detected_no_intelligence_shapes → None without markers

### Phase 3 Gate

```bash
pytest tests/ -v
```

**All tests must pass before proceeding to Phase 4.**

-----

## Phase 4 — Onboarding

**Goal:** Automatic shape extraction from existing systems. Zero cold start.

### 4.1 scanner.py

**Build:**

- `ScanResult` dataclass: `source`, `source_type`, `shapes: List[Shape]`
- `Scanner` class with configurable `min_unique_values=3` and `margin=0.05`
- `Scanner.scan_dataframe(df, name) -> ScanResult`: numeric columns → dimensions, min/max → boundaries
- `Scanner.scan_database(connection_string, tables) -> ScanResult`: one shape per table, SQL MIN/MAX queries
- `Scanner.scan_json(data, name) -> ScanResult`: walk nested structure, extract numeric fields with ranges
- `Scanner.scan_openapi(spec, name) -> ScanResult`: one shape per endpoint with numeric params
- `Scanner.scan_codebase(path, name) -> ScanResult`: ast parsing, function signatures, type hints → dimensions
- `scan(source) -> ScanResult`: auto-detect convenience function
- Helper methods: `_extract_numeric_fields()`, `_schema_to_dimension()`, `_annotation_to_dimension()`

**Optional deps:**

- pandas (for scan_dataframe)
- sqlalchemy (for scan_database)
- pyyaml (for YAML OpenAPI specs)

**Integration:** Memory.scan_and_store(source) — scan and store in one call.

**Verify:**

```
pytest tests/test_scanner.py -v
```

- test_scan_dataframe → numeric columns become dimensions with correct ranges
- test_scan_dataframe_skips_non_numeric → string columns ignored
- test_scan_dataframe_skips_low_cardinality → boolean/flag columns filtered
- test_scan_json_flat → simple dict produces shape
- test_scan_json_nested → nested fields use dot-path names
- test_scan_json_list → list of dicts extracts ranges across all records
- test_scan_openapi_with_constraints → min/max from spec become boundaries
- test_scan_openapi_without_constraints → default ranges used
- test_scan_codebase_typed_function → int/float params become dimensions
- test_scan_codebase_no_types → untyped functions skipped
- test_scan_auto_detect_dataframe → scan() routes to scan_dataframe
- test_scan_auto_detect_dict → scan() routes to scan_json
- test_scan_auto_detect_directory → scan() routes to scan_codebase
- test_scan_and_store_integration → Memory.scan_and_store() populates library

### Phase 4 Gate

```bash
pytest tests/ -v
```

**All tests must pass before proceeding to Phase 5.**

-----

## Phase 5 — Integration + Ship

**Goal:** Memory class integrates all mechanics. Full test suite. README. PyPI publish.

### 5.1 Update memory.py

Update the Memory class to integrate all Phase 2, 3, and 4 mechanics:

```python
class Memory:
    def __init__(self, namespace="default", backend="memory",
                 anchor=True, path=None, config=None,
                 growth=True, overlap=True, decay=True,
                 merge=True, narrative=True):
        # ... base init ...
        self.growth_enabled = growth
        self.overlap_enabled = overlap
        self.decay_enabled = decay
        self.merge_enabled = merge
        self.narrative_enabled = narrative
        
        if growth: self.growth_tracker = GrowthTracker()
        if overlap: pass  # Runs inline on store
        if decay: self.decay_tracker = DecayTracker()
        if merge: self.merge_detector = MergeDetector()
        if narrative: self.narrative = NarrativeEngine()
        
        self.void_mapper = VoidMapper()
        self.emergence_detector = EmergenceDetector()
```

**New methods to add:**

- `recall_with_confidence(state, raw)` → List[Tuple[Shape, float]]
- `store_in_trajectory(shape, trajectory_id, metadata)` → str
- `predict_next(current_shape_id, limit)` → List[Tuple[str, float]]
- `find_common_paths(min_length, min_occurrences)` → List[List[str]]
- `maintain()` → run decay cycle
- `_check_merges()` → run after growth
- `map_void(sample_count)` → List[VoidRegion]
- `void_percentage(sample_count)` → float
- `largest_gap()` → Optional[VoidRegion]
- `detect_emergence()` → List[Tier2Shape]
- `detect_gcs(growth_history)` → Optional[Tier2Shape]
- `detect_is()` → Optional[Tier2Shape]
- `scan_and_store(source)` → int (number of shapes stored)

### 5.2 Update **init**.py

```python
from .shape import Shape, Dimension
from .memory import Memory
from .library import Library
from .emergence import Tier2Shape
from .scanner import Scanner, scan

__all__ = ["Memory", "Shape", "Dimension", "Library", "Tier2Shape", "Scanner", "scan"]
```

### 5.3 Full Test Suite

```bash
pytest tests/ -v --cov=doorway_memory
```

Target: 90%+ coverage. All tests pass.

### 5.4 README.md

Lead with standalone positioning:

- What it is (geometric memory, not vector DB)
- Install: `pip install doorway-memory`
- Quick start: 10 lines to store and recall
- Scanner: point at your system, shapes minted automatically
- API reference: store, recall, is_known, is_void, scan_and_store
- Advanced: growth, overlap, decay, narrative, confidence
- Structural: merge, void map, emergence
- Optional: xycore anchoring, supabase persistence
- Part of Doorway stack (but standalone)

### 5.5 Publish

```bash
python -m build
twine upload dist/*
```

Verify: `pip install doorway-memory` works clean.

### Phase 5 Gate

- All tests pass
- Coverage > 90%
- `pip install doorway-memory` works
- `from doorway_memory import Memory, Shape, Dimension, scan` works
- README renders correctly on PyPI

-----

## Dependency Map

```
emergence.py
    ↓ uses
overlap.py ← merge.py
    ↓ uses
intersect.py ← void_map.py
    ↓ uses
shape.py ← growth.py, decay.py

narrative.py (independent, uses shape IDs only)
anchor.py (independent, wraps xycore)
scanner.py (uses shape, optional pandas/sqlalchemy/pyyaml)
library.py (uses shape + intersect)
memory.py (orchestrates everything)
```

-----

## Threshold Reference

All thresholds are module-level constants. Tunable without code changes.

|Constant                  |File        |Default|Purpose                                       |
|--------------------------|------------|-------|----------------------------------------------|
|NEAR_MISS_MARGIN          |growth.py   |0.15   |How far outside counts as near-miss           |
|GROWTH_THRESHOLD          |growth.py   |5      |Near-misses before expansion                  |
|MAX_EXPANSION_RATIO       |growth.py   |0.25   |Maximum single expansion                      |
|MERGE_OVERLAP_RATIO       |merge.py    |0.50   |Overlap % to trigger merge                    |
|DECAY_GRACE_PERIOD        |decay.py    |604800 |7 days before decay starts                    |
|DECAY_RATE                |decay.py    |0.02   |2% boundary shrink per cycle                  |
|MIN_DIMENSION_RANGE       |decay.py    |0.01   |Collapse threshold                            |
|ARCHIVE_VOLUME_THRESHOLD  |decay.py    |0.001  |Archive threshold                             |
|MIN_SHAPES_FOR_EMERGENCE  |emergence.py|3      |Minimum shapes for Tier 2                     |
|MIN_DOMAINS_FOR_EMERGENCE |emergence.py|2      |Minimum domains for Tier 2                    |
|EMERGENCE_VOLUME_THRESHOLD|emergence.py|0.001  |Minimum intersection volume                   |
|MINIMUM_SHAPE_CONFIDENCE  |shape.py    |0.10   |Below this, shape match ignored               |
|MIN_UNIQUE_VALUES         |scanner.py  |3      |Min unique values to treat column as dimension|
|SCANNER_MARGIN            |scanner.py  |0.05   |Boundary margin on scanned ranges             |

-----

## What Comes Next (Not In This Build)

- **namespace.py** — Scoping: private shapes, shared shapes, access control
- **events.py** — Hooks: subscribe to growth, merge, decay, emergence events
- **Dashboard rendering** — Chain visualization, live containment testing, library overview
- **Doorway integration** — Memory as the geometric layer under the reasoning engine
- **API server** — REST endpoints for memory operations

These are planned but not part of this build. Ship the package first.

-----

*doorway-memory · Geometric Memory Engine · © 2026 Doorway · doorwayagi.com*
