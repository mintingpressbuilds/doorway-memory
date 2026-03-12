# doorway-memory

**Geometric memory engine. Store knowledge as shapes. Retrieve by containment. Verify by chain.**

[![PyPI version](https://img.shields.io/pypi/v/doorway-memory.svg)](https://pypi.org/project/doorway-memory/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

-----

## What This Is

A memory system where knowledge is geometry. Not embeddings. Not vectors. Not key-value pairs. Shapes.

A shape is a region in dimensional space with defined boundaries. Store a shape and you're saying "I know what happens inside these boundaries." Query a point and the system answers a binary question: is this point inside known territory, or is it in the void?

That's a different question than "what's similar to this?" Vector databases answer similarity. doorway-memory answers containment. Similarity is fuzzy — close enough counts. Containment is precise — you're inside or you're not.

```bash
pip install doorway-memory
```

```python
from doorway_memory import Memory, Shape, Dimension

mem = Memory()

# Define a region of known territory
shape = Shape(dimensions={
    "temperature": Dimension("temperature", 90.0, 110.0),
    "pressure": Dimension("pressure", 0.9, 1.1),
})
mem.store(shape)

# Is this point inside known territory?
mem.is_known({"temperature": 100.0, "pressure": 1.0})  # True

# Is this point in the void?
mem.is_void({"temperature": 200.0, "pressure": 1.0})   # True — unknown territory
```

No AI required. No specific domain. Any system that operates in a dimensional space can use this.

-----

## Scanner — Zero Cold Start

Point doorway-memory at your existing system. It reads the structure and mints shapes automatically.

```python
from doorway_memory import Memory, scan

mem = Memory()

# Scan a pandas DataFrame
import pandas as pd
df = pd.read_csv("orders.csv")
mem.scan_and_store(df)
# Every numeric column becomes a dimension.
# Observed min/max become boundaries. Done.

# Scan a database
mem.scan_and_store("postgresql://user:pass@localhost/mydb")
# Each table becomes a shape. Numeric columns become dimensions.

# Scan a JSON file or API response
mem.scan_and_store("data.json")
# Walks nested structure. Extracts numeric fields with ranges.

# Scan an OpenAPI spec
mem.scan_and_store("openapi.yaml")
# Each endpoint with numeric parameters becomes a shape.

# Scan a Python codebase
mem.scan_and_store("./src")
# Functions with typed numeric parameters become shapes.
```

Day one: your library is populated with the geometric territory of your own infrastructure. No manual shape definition. No cold start.

The `scan()` function auto-detects source type. One function, any source.

-----

## Confidence — How Deeply Known

Containment isn't binary when you need nuance. `confidence()` tells you how deep inside known territory a point sits.

```python
shape = Shape(dimensions={
    "x": Dimension("x", 0.0, 100.0),
})
mem.store(shape)

# Center of shape — maximum confidence
shape.confidence({"x": 50.0})   # 1.0

# Near the edge — low confidence
shape.confidence({"x": 95.0})   # 0.1

# On the boundary — zero
shape.confidence({"x": 100.0})  # 0.0

# Outside — zero
shape.confidence({"x": 110.0})  # 0.0

# Per-dimension breakdown
shape.confidence_breakdown({"x": 50.0, "y": 90.0})
# {"x": 1.0, "y": 0.2} — strong on x, weak on y
```

Edge knowledge and core knowledge are distinguishable.

-----

## Growth — Shapes That Learn

Shapes expand from use. When points consistently land just outside a boundary, the shape grows to absorb them.

```python
mem = Memory(growth=True)
shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
mem.store(shape)

# Point at 10.5 is a near-miss — just outside the boundary
mem.recall({"x": 10.5})  # Empty — in void
mem.recall({"x": 10.5})  # Still tracking...
mem.recall({"x": 10.5})  # Near-miss count: 3
mem.recall({"x": 10.5})  # Near-miss count: 4
mem.recall({"x": 10.5})  # Threshold hit — shape expands

mem.is_known({"x": 10.5})  # True — boundary grew
```

The library calibrates itself from actual queries. No manual tuning.

-----

## Overlap — Cross-Domain Emergence

When two shapes share a region of dimensional space, that intersection is detected automatically and stored as a new derived shape.

```python
shape_a = Shape(
    dimensions={"x": Dimension("x", 0.0, 10.0)},
    metadata={"domain": "physics"}
)
shape_b = Shape(
    dimensions={"x": Dimension("x", 5.0, 15.0)},
    metadata={"domain": "economics"}
)
mem.store(shape_a)
mem.store(shape_b)
# Overlap detected: x=[5.0, 10.0]
# New derived shape stored with both parents linked.
# A cross-domain pattern nobody explicitly defined.
```

-----

## Decay — Knowledge That Fades

Shapes that aren't queried shrink over time. Frequently accessed shapes resist decay.

```python
mem = Memory(decay=True)
mem.store(shape)

# Shape is active — accessed regularly
mem.recall({"x": 5.0})  # Access recorded

# Much later — no queries for weeks...
mem.maintain()  # Boundaries contract slightly

# Even later — still no queries...
mem.maintain()  # Shape archived — removed from active library,
                # preserved in chain for history
```

The library self-prunes. Active knowledge stays strong. Dead knowledge fades gracefully.

-----

## Merge — Shape Fusion

When growth causes two shapes to touch, they merge into one continuous region.

```python
# Shape A: x=[0, 10]
# Shape B: x=[8, 20]  (overlaps significantly)
# After merge: single shape x=[0, 20]
# Parents archived. Merged shape replaces both.
```

The library consolidates instead of accumulating redundant overlapping shapes.

-----

## Narrative — Trajectories Through Knowledge

Shapes stored in sequence form trajectories. The system predicts what comes next.

```python
# Record a learning trajectory
mem.store_in_trajectory(shape_a, "session-1")
mem.store_in_trajectory(shape_b, "session-1")
mem.store_in_trajectory(shape_c, "session-1")

# Another session follows a similar path
mem.store_in_trajectory(shape_a, "session-2")
mem.store_in_trajectory(shape_b, "session-2")

# Predict: after shape_b, what usually comes next?
mem.predict_next(shape_b.id)
# [("shape_c_id", 0.67), ...]

# Find common paths across all sessions
mem.find_common_paths(min_length=2)
# [["shape_a_id", "shape_b_id"]] — this sequence appears in multiple trajectories
```

Memory has temporal structure. Not just what you know, but how you learned it.

-----

## Void Mapping — The Shape of What You Don't Know

The void has structure. Boundaries. Size. Neighbors.

```python
# What percentage of the bounded space is unknown?
mem.void_percentage()  # 0.73 — 73% is void

# Where are the biggest gaps?
regions = mem.map_void()
for region in regions:
    print(region.volume())           # How big is this gap?
    print(region.center())           # Where is it?
    print(region.neighboring_shapes) # What borders it?

# What's the single largest gap?
biggest = mem.largest_gap()
```

The gap detector expressed as persistent geometry.

-----

## Emergence — Tier 2 Pattern Detection

Patterns that span many shapes across many domains emerge automatically.

```python
# After storing shapes from multiple domains...
tier2_patterns = mem.detect_emergence()

for pattern in tier2_patterns:
    print(pattern.shape)       # The geometric intersection
    print(pattern.parent_ids)  # Which shapes contribute
    print(pattern.domains)     # Which domains it spans
    print(pattern.strength)    # How many shapes participate

# Generative Complexity System — the system recognizing its own growth
gcs = mem.detect_gcs(growth_history)

# Intelligence System — the mechanism recognizing the mechanism
is_pattern = mem.detect_is()
```

Nobody programs emergence. It's detected from the geometry.

-----

## Verified Memory — Cryptographic Chain

Every store, every growth event, every merge, every archive is anchored to an [xycore](https://pypi.org/project/xycore/) cryptographic chain. Optional but powerful.

```bash
pip install doorway-memory[anchor]
```

```python
mem = Memory(anchor=True)
shape_id = mem.store(shape)

# Verify any shape's provenance
proof = mem.verify(shape_id)
# {"anchor_id": "...", "timestamp": ..., "hash": "...", "verified": True}

# Replay memory history
for shape in mem.replay():
    print(shape.id, shape.metadata)
```

Provable, replayable history of everything the system has ever learned.

-----

## Storage Backends

```python
# In-memory (default — no persistence)
mem = Memory()

# File persistence
mem = Memory(backend="file", path="./memory.json")

# Supabase (cloud persistence)
pip install doorway-memory[supabase]
mem = Memory(backend="supabase", config={
    "url": "https://xxx.supabase.co",
    "key": "your-service-key"
})
```

-----

## Configuration

Every threshold is tunable.

```python
from doorway_memory.growth import GROWTH_THRESHOLD, NEAR_MISS_MARGIN
from doorway_memory.decay import DECAY_RATE, DECAY_GRACE_PERIOD
from doorway_memory.merge import MERGE_OVERLAP_RATIO
from doorway_memory.emergence import MIN_SHAPES_FOR_EMERGENCE
```

|Threshold               |Default|What It Controls                   |
|------------------------|-------|-----------------------------------|
|NEAR_MISS_MARGIN        |0.15   |How far outside counts as near-miss|
|GROWTH_THRESHOLD        |5      |Near-misses before expansion       |
|DECAY_RATE              |0.02   |Boundary shrink per cycle          |
|DECAY_GRACE_PERIOD      |7 days |Time before decay starts           |
|MERGE_OVERLAP_RATIO     |0.50   |Overlap % to trigger merge         |
|MIN_SHAPES_FOR_EMERGENCE|3      |Minimum shapes for Tier 2          |

-----

## Full API

```python
from doorway_memory import Memory, Shape, Dimension, scan

mem = Memory()

# ── Store ──────────────────────────────
mem.store(shape)                              # Store a shape
mem.store_in_trajectory(shape, "session-1")   # Store as trajectory step
mem.scan_and_store(source)                    # Scan system, store all shapes

# ── Recall ─────────────────────────────
mem.recall(point)                             # Shapes containing this point
mem.recall_with_confidence(point)             # With confidence gradients
mem.is_known(point)                           # Inside known territory?
mem.is_void(point)                            # In the void?

# ── Predict ────────────────────────────
mem.predict_next(shape_id)                    # What comes after this?
mem.find_common_paths()                       # Shared trajectories

# ── Map ────────────────────────────────
mem.map_void()                                # Characterize unknown territory
mem.void_percentage()                         # % of space that's void
mem.largest_gap()                             # Biggest void region

# ── Emerge ─────────────────────────────
mem.detect_emergence()                        # Find Tier 2 patterns
mem.detect_gcs(history)                       # Growth pattern self-recognition
mem.detect_is()                               # Intelligence mechanism detection

# ── Verify ─────────────────────────────
mem.verify(shape_id)                          # Cryptographic proof
mem.replay()                                  # Walk memory history

# ── Maintain ───────────────────────────
mem.maintain()                                # Run decay cycle
mem.count()                                   # Shape count
```

-----

## Not a Vector Database

|                   |Vector DB                    |doorway-memory                      |
|-------------------|-----------------------------|------------------------------------|
|**Stores**         |Embeddings (points)          |Shapes (regions)                    |
|**Retrieves by**   |Similarity (nearest neighbor)|Containment (inside or not)         |
|**Answer type**    |"Here's what's close"        |"You're inside / you're in the void"|
|**Confidence**     |Distance score               |Depth gradient (center to boundary) |
|**Learns from use**|No                           |Yes — growth, decay, merge          |
|**Cross-domain**   |No                           |Yes — overlap detection             |
|**Maps unknown**   |No                           |Yes — void mapping                  |
|**Emergence**      |No                           |Yes — Tier 2 pattern detection      |
|**Verified**       |No                           |Yes — cryptographic chain           |

-----

## Part of Doorway

doorway-memory is a standalone package. It does not require Doorway, AI, or any specific domain.

It's also the geometric memory layer of the [Doorway](https://doorwayagi.com) reasoning stack.

|Package                                               |What It Is                                |
|------------------------------------------------------|------------------------------------------|
|[xycore](https://pypi.org/project/xycore/)            |Cryptographic chain primitive             |
|[pruv](https://pypi.org/project/pruv/)                |Verification infrastructure               |
|**doorway-memory**                                    |**Geometric memory engine (this package)**|
|[doorway-agi](https://pypi.org/project/doorway-agi/)  |AGI reasoning engine                      |
|[vantagepoint](https://pypi.org/project/vantagepoint/)|Structured thinking methodology           |

-----

## License

Apache License 2.0 — see <LICENSE> for details.

© 2026 Doorway · [doorwayagi.com](https://doorwayagi.com)

Created by Luke H
