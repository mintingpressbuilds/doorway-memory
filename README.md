# doorway-memory

Geometric memory engine. Stores knowledge as shapes — n-dimensional bounded regions — and retrieves by containment testing. Not a vector database. Not embeddings. Pure geometry.

## Install

```bash
pip install doorway-memory
```

## Quick Start

```python
from doorway_memory import Memory, Shape, Dimension

# Create memory
mem = Memory(anchor=False)

# Define a shape (knowledge region)
shape = Shape(dimensions={
    "temperature": Dimension("temperature", 15.0, 35.0),
    "humidity": Dimension("humidity", 30.0, 80.0),
})

# Store it
mem.store(shape)

# Query — is this point known?
mem.is_known({"temperature": 25.0, "humidity": 50.0})  # True
mem.is_void({"temperature": -10.0, "humidity": 95.0})  # True

# Recall matching shapes
shapes = mem.recall({"temperature": 25.0, "humidity": 50.0})
```

## Scanner — Zero Cold Start

Point the scanner at your existing systems. Shapes are minted automatically.

```python
from doorway_memory import Memory, scan

mem = Memory(anchor=False)

# Scan a dataframe (dict of lists)
mem.scan_and_store({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

# Scan JSON data
mem.scan_and_store({"temperature": 25.0, "pressure": 1.0})

# Scan a Python codebase (extracts typed function signatures)
mem.scan_and_store("/path/to/your/code")

# Scan an OpenAPI spec
mem.scan_and_store({"paths": {"/users": {"get": {"parameters": [...]}}}})

# Scan a database schema
mem.scan_and_store({"users": {"age": {"type": "integer", "min": 0, "max": 150}}})

# Or use scan() directly for inspection
from doorway_memory import scan
result = scan({"x": [1.0, 2.0, 3.0]})
print(result.source_type)  # "dataframe"
print(result.shapes)       # [Shape(...)]
```

## API Reference

### Core

| Method | Description |
|--------|-------------|
| `mem.store(shape)` | Store a shape, returns shape ID |
| `mem.recall(point)` | Find all shapes containing point |
| `mem.is_known(point)` | True if point is inside known territory |
| `mem.is_void(point)` | True if point is in the void |
| `mem.recall_with_confidence(point)` | Returns `[(shape, confidence), ...]` |
| `mem.scan_and_store(source)` | Scan and store, returns count |
| `mem.count()` | Number of stored shapes |
| `mem.get(shape_id)` | Retrieve shape by ID |

### Advanced Mechanics

```python
# Growth — shapes expand toward near-miss queries
mem = Memory(anchor=False, growth=True)

# Decay — unused shapes shrink over time
result = mem.maintain()  # {"decayed": n, "archived": n}

# Merge — overlapping shapes fuse
mem._check_merges()

# Narrative — track trajectories through shape space
mem.store_in_trajectory(shape, "trajectory_id", timestamp=0.0)
prediction = mem.predict_next("trajectory_id")

# Void mapping — characterize what you don't know
voids = mem.map_void("x", (0.0, 100.0))
pct = mem.void_percentage({"x": (0.0, 100.0)})
gap = mem.largest_gap("x", (0.0, 100.0))

# Emergence — detect Tier 2 patterns
tier2 = mem.detect_emergence()
```

### Configuration

Toggle mechanics on/off:

```python
mem = Memory(
    anchor=False,     # xycore anchoring (requires xycore)
    growth=True,      # near-miss expansion
    overlap=True,     # overlap detection on store
    decay=True,       # shape decay on maintain()
    merge=True,       # shape fusion
    narrative=True,   # trajectory tracking
)
```

### Structural Primitives

```python
from doorway_memory import Shape, Dimension

# Shape = bounded region in n-dimensional space
shape = Shape(dimensions={
    "x": Dimension("x", 0.0, 10.0),
    "y": Dimension("y", 0.0, 5.0),
})

shape.contains({"x": 5.0, "y": 2.5})     # True
shape.distance_to_boundary({"x": 5.0})    # positive (inside)
shape.volume()                             # 50.0
shape.to_dict()                            # serializable
Shape.from_dict(shape.to_dict())           # deserializable
```

## Optional Dependencies

- **xycore** — Blockchain anchoring for verifiable memory
- **pandas** — DataFrame scanning
- **sqlalchemy** — Database scanning
- **pyyaml** — YAML OpenAPI spec scanning

All optional. Core functionality works with numpy only.

## Part of Doorway

doorway-memory is the geometric memory layer of the [Doorway](https://doorwayagi.com) stack. It works standalone — no other Doorway packages required.

## License

Apache 2.0
