# Doorway Memory Build Specification

## Geometric Memory Engine

**Package:** `doorway-memory`
**Version:** 0.1.0
**Author:** Luke H
**License:** Apache 2.0

-----

## 1. Overview

### What It Is

A geometric memory engine. Knowledge is stored as shapes — regions in n-dimensional space with defined boundaries. Retrieval is containment testing — is this point inside known territory, or is it in the void? Every operation can be cryptographically anchored for verifiable history.

This is a standalone package. It does not require Doorway, does not require AI, and does not require any specific domain. Anyone building any system that needs geometric memory — where the question is “have I seen this before?” answered by boundary math, not keyword search — can use it.

```bash
pip install doorway-memory
```

```python
from doorway_memory import Memory, Shape, Dimension

# Create memory
mem = Memory()

# Define a region of known territory
shape = Shape(dimensions={
    "temperature": Dimension("temperature", 90.0, 110.0),
    "pressure": Dimension("pressure", 0.9, 1.1),
})
mem.store(shape)

# Test if a point is inside known territory
mem.is_known({"temperature": 100.0, "pressure": 1.0})  # True
mem.is_void({"temperature": 200.0, "pressure": 1.0})   # True — unknown territory
```

### What It Is Not

Not a vector database. Vector databases find similar items. doorway-memory tests whether a point is inside a geometric boundary. Similarity asks “how close?” Containment asks “is it inside?” Different question, different math, different answer.

Not a key-value store. Retrieval is not by ID or keyword. You give it a point in dimensional space and it returns every shape that contains that point. The query is the geometry itself.

Not tied to AI. It works with Doorway’s reasoning engine, but it works equally well with any system that operates in a dimensional space — sensor data, financial models, game state, scientific measurement, anything with numeric dimensions and boundaries.

### How It Works

You define dimensions. You create shapes as regions in those dimensions. You store shapes in a library. When you need to recall, you provide a point — a set of dimensional coordinates — and the library returns every shape that contains it. If no shape contains it, the point is in the void. That’s meaningful information: you’ve never seen this territory before.

```
POINT (dimensional coordinates)
  ↓
Library query
  ↓
┌──────────────────────────────────┐
│  Containing shapes found?        │
│                                  │
│  YES → Known territory           │
│        Returns: list of shapes   │
│                                  │
│  NO  → Void                      │
│        Returns: empty list       │
│        (genuinely unknown)       │
└──────────────────────────────────┘
```

Optional: every store and recall operation can be anchored to an xycore cryptographic chain. This gives you verifiable, replayable history — proof of what was stored, when, and in what order. You can walk the chain forward through time and reconstruct the entire memory trajectory.

### Integration with Doorway

doorway-memory integrates naturally with the Doorway reasoning stack because the dimensional space is the same. Doorway’s shape library stores geometric patterns as semantic descriptors matched by similarity. doorway-memory stores the same patterns as geometric regions tested by containment. Same coordinate system, different query type.

```
INPUT
  ↓
Feature extraction (shared)
  ↓
Dimensional signature (the "point")
  ↓
┌─────────────┬─────────────┐
│  SEMANTIC   │  GEOMETRIC  │
│  (doorway)  │  (memory)   │
│             │             │
│  Similarity │ Containment │
│  scoring    │ testing     │
│             │             │
│  "How close │ "Is it      │
│   is this?" │  inside?"   │
└─────────────┴─────────────┘
```

But this integration is optional. doorway-memory stands alone. You define your own dimensions, your own shapes, your own points. The package doesn’t import doorway and doesn’t need it installed.

### Dependencies

- **numpy** — geometric math (required)
- **xycore** — cryptographic anchoring, chain, replay (optional: `pip install doorway-memory[anchor]`)
- **supabase** — cloud persistence (optional: `pip install doorway-memory[supabase]`)

-----

## 2. Xycore Addition: walk()

**Location:** xycore/chain.py (or wherever Chain lives)

```python
def walk(chain_id: str, start: str = None, end: str = None) -> Iterator[Anchor]:
    """
    Traverse a chain, yielding anchors in order.
    
    Parameters:
        chain_id: Identifier for the chain to walk
        start: Optional anchor_id to start from (inclusive)
        end: Optional anchor_id to stop at (inclusive)
    
    Yields:
        Anchor objects in chain order (oldest to newest)
    
    Behavior:
        - If start is None, begin at chain origin
        - If end is None, walk to chain head
        - Each anchor contains: id, data, timestamp, previous_id
        - Raises ChainNotFound if chain_id doesn't exist
        - Raises AnchorNotFound if start/end not in chain
    """
```

Implementation: follow previous_id pointers from end to start, reverse to yield oldest-first.

Estimated lines: ~50

-----

## 3. shape.py

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import hashlib
import json

@dataclass
class Dimension:
    """A single axis in the geometric space."""
    name: str
    min_value: float
    max_value: float

@dataclass  
class Shape:
    """
    A geometric region in n-dimensional space.
    
    Attributes:
        dimensions: Dict mapping dimension name to Dimension object
        metadata: Optional descriptive data
        id: Unique identifier (hash of structure)
        anchor_id: Optional xycore anchor reference
    """
    dimensions: Dict[str, Dimension]
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    anchor_id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Hash of dimensional structure."""
        content = json.dumps({
            name: {"min": d.min_value, "max": d.max_value}
            for name, d in self.dimensions.items()
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def contains(self, point: Dict[str, float]) -> bool:
        """
        Check if point is inside this shape's boundaries.
        
        Returns True if point is within bounds on ALL shared dimensions.
        Dimensions in shape but not in point: ignored.
        Dimensions in point but not in shape: ignored.
        """
        for name, dim in self.dimensions.items():
            if name in point:
                value = point[name]
                if value < dim.min_value or value > dim.max_value:
                    return False
        return True
    
    def distance_to_boundary(self, point: Dict[str, float]) -> float:
        """
        Minimum distance from point to any boundary edge.
        
        Returns:
            Positive value if inside (distance to nearest edge)
            Negative value if outside (distance past nearest edge)
            0.0 if exactly on boundary
        """
        if not self.contains(point):
            max_violation = 0.0
            for name, dim in self.dimensions.items():
                if name in point:
                    value = point[name]
                    if value < dim.min_value:
                        max_violation = max(max_violation, dim.min_value - value)
                    elif value > dim.max_value:
                        max_violation = max(max_violation, value - dim.max_value)
            return -max_violation
        
        min_distance = float('inf')
        for name, dim in self.dimensions.items():
            if name in point:
                value = point[name]
                dist_to_min = value - dim.min_value
                dist_to_max = dim.max_value - value
                min_distance = min(min_distance, dist_to_min, dist_to_max)
        return min_distance
    
    def to_dict(self) -> Dict:
        """Serialize for storage/anchoring."""
        return {
            "id": self.id,
            "dimensions": {
                name: {"min": d.min_value, "max": d.max_value}
                for name, d in self.dimensions.items()
            },
            "metadata": self.metadata,
            "anchor_id": self.anchor_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Shape":
        """Deserialize from storage."""
        dimensions = {
            name: Dimension(name=name, min_value=d["min"], max_value=d["max"])
            for name, d in data["dimensions"].items()
        }
        return cls(
            dimensions=dimensions,
            metadata=data.get("metadata"),
            id=data.get("id"),
            anchor_id=data.get("anchor_id")
        )


def extract_point(input_data: Dict) -> Dict[str, float]:
    """
    Extract dimensional signature from input.
    
    Must match doorway's feature extraction so semantic 
    and geometric modes operate on the same coordinate system.
    
    Standard dimensions (from doorway shape library):
        - structure_type: numeric encoding of structure
        - element_count: number of elements
        - constraint_count: number of constraints
        - implication_direction: forward=1, reverse=-1, bidirectional=0
        - complexity_score: derived metric
    """
    point = {}
    
    if "structure" in input_data:
        point["structure_type"] = _encode_structure(input_data["structure"])
    if "elements" in input_data:
        point["element_count"] = float(len(input_data["elements"]))
    if "constraints" in input_data:
        point["constraint_count"] = float(len(input_data["constraints"]))
    if "implication" in input_data:
        point["implication_direction"] = _encode_implication(input_data["implication"])
    
    point["complexity_score"] = _compute_complexity(input_data)
    
    return point


def _encode_structure(structure: str) -> float:
    encoding = {
        "causal": 1.0, "compositional": 2.0, "relational": 3.0,
        "temporal": 4.0, "conditional": 5.0
    }
    return encoding.get(structure, 0.0)


def _encode_implication(implication: str) -> float:
    encoding = {"forward": 1.0, "reverse": -1.0, "bidirectional": 0.0}
    return encoding.get(implication, 0.0)


def _compute_complexity(input_data: Dict) -> float:
    elements = len(input_data.get("elements", []))
    constraints = len(input_data.get("constraints", []))
    return float(elements * constraints) if constraints > 0 else float(elements)
```

Estimated lines: ~150

-----

## 4. intersect.py

```python
from typing import List, Dict, Tuple
from .shape import Shape

def point_in_shape(point: Dict[str, float], shape: Shape) -> bool:
    """Wrapper for shape.contains()."""
    return shape.contains(point)


def find_containing_shapes(
    point: Dict[str, float], 
    shapes: List[Shape]
) -> List[Shape]:
    """Find all shapes that contain this point."""
    return [s for s in shapes if s.contains(point)]


def find_nearest_shapes(
    point: Dict[str, float],
    shapes: List[Shape],
    limit: int = 5
) -> List[Tuple[Shape, float]]:
    """
    Find shapes nearest to this point, even if outside.
    
    Returns list of (shape, distance) tuples sorted by distance.
    Positive = inside. Negative = outside.
    """
    distances = []
    for shape in shapes:
        dist = shape.distance_to_boundary(point)
        distances.append((shape, dist))
    
    distances.sort(key=lambda x: (-x[1] if x[1] > 0 else float('inf') + abs(x[1])))
    return distances[:limit]


def find_void(point: Dict[str, float], shapes: List[Shape]) -> bool:
    """True if no shape contains this point."""
    return len(find_containing_shapes(point, shapes)) == 0
```

Estimated lines: ~60

-----

## 5. library.py

```python
from typing import List, Dict, Optional, Iterator
from .shape import Shape
from .intersect import find_containing_shapes, find_void

class Library:
    """
    Storage and retrieval for geometric shapes.
    
    Backends: "memory" (default), "file", "supabase"
    """
    
    def __init__(self, backend="memory", path=None, config=None):
        self.backend = backend
        self.path = path
        self.config = config or {}
        self._shapes: Dict[str, Shape] = {}
        self._load()
    
    def _load(self):
        if self.backend == "file" and self.path:
            self._load_from_file()
        elif self.backend == "supabase":
            self._load_from_supabase()
    
    def _save(self):
        if self.backend == "file" and self.path:
            self._save_to_file()
        elif self.backend == "supabase":
            self._save_to_supabase()
    
    def add(self, shape: Shape) -> str:
        """Add shape. Returns shape ID."""
        self._shapes[shape.id] = shape
        self._save()
        return shape.id
    
    def get(self, shape_id: str) -> Optional[Shape]:
        """Retrieve shape by ID."""
        return self._shapes.get(shape_id)
    
    def remove(self, shape_id: str) -> bool:
        """Remove shape. Returns True if existed."""
        if shape_id in self._shapes:
            del self._shapes[shape_id]
            self._save()
            return True
        return False
    
    def query(self, point: Dict[str, float]) -> List[Shape]:
        """Find all shapes containing this point."""
        return find_containing_shapes(point, list(self._shapes.values()))
    
    def is_void(self, point: Dict[str, float]) -> bool:
        """Check if point is in void."""
        return find_void(point, list(self._shapes.values()))
    
    def all(self) -> Iterator[Shape]:
        """Iterate all shapes."""
        yield from self._shapes.values()
    
    def count(self) -> int:
        return len(self._shapes)
    
    def _load_from_file(self):
        import json
        from pathlib import Path
        p = Path(self.path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
                for shape_data in data.get("shapes", []):
                    shape = Shape.from_dict(shape_data)
                    self._shapes[shape.id] = shape
    
    def _save_to_file(self):
        import json
        from pathlib import Path
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "shapes": [s.to_dict() for s in self._shapes.values()]
            }, f, indent=2)
    
    def _load_from_supabase(self):
        pass  # Implementation depends on supabase client
    
    def _save_to_supabase(self):
        pass  # Implementation depends on supabase client
```

Estimated lines: ~130

-----

## 6. memory.py

```python
from typing import Dict, List, Optional, Any, Iterator
from .shape import Shape, extract_point
from .library import Library

try:
    import xycore
    HAS_XYCORE = True
except ImportError:
    HAS_XYCORE = False

class Memory:
    """
    Geometric memory engine.
    
    API:
        store(shape)      - Add shape to memory
        recall(state)     - Find shapes containing state
        verify(shape_id)  - Get proof chain for shape
        replay(start,end) - Walk memory history
    """
    
    def __init__(self, namespace="default", backend="memory",
                 anchor=True, path=None, config=None):
        self.namespace = namespace
        self.library = Library(backend=backend, path=path, config=config)
        self.anchor_enabled = anchor and HAS_XYCORE
        self._chain_id = f"memory:{namespace}"
    
    def store(self, shape: Shape, metadata=None) -> str:
        """Store shape. Returns shape ID."""
        if metadata:
            shape.metadata = {**(shape.metadata or {}), **metadata}
        
        if self.anchor_enabled:
            anchor = xycore.Anchor.create(
                data=shape.to_dict(), chain_id=self._chain_id
            )
            shape.anchor_id = anchor.id
        
        return self.library.add(shape)
    
    def recall(self, state: Dict, raw=False) -> List[Shape]:
        """Find shapes containing state. Set raw=True for unextracted input."""
        point = extract_point(state) if raw else state
        return self.library.query(point)
    
    def is_known(self, state: Dict, raw=False) -> bool:
        """True if state is inside known territory."""
        return len(self.recall(state, raw=raw)) > 0
    
    def is_void(self, state: Dict, raw=False) -> bool:
        """True if state is in void (unknown territory)."""
        return not self.is_known(state, raw=raw)
    
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
    
    def count(self) -> int:
        return self.library.count()
```

Estimated lines: ~140

-----

## 7. anchor.py

```python
"""xycore integration for doorway-memory."""

from typing import Dict, Optional

try:
    import xycore
    HAS_XYCORE = True
except ImportError:
    HAS_XYCORE = False


def anchor_shape(shape_data: Dict, chain_id: str) -> Optional[str]:
    """Anchor shape data to xycore chain. Returns anchor ID."""
    if not HAS_XYCORE:
        return None
    anchor = xycore.Anchor.create(data=shape_data, chain_id=chain_id)
    return anchor.id


def verify_anchor(anchor_id: str) -> Optional[Dict]:
    """Verify anchor integrity."""
    if not HAS_XYCORE:
        return None
    anchor = xycore.Anchor.get(anchor_id)
    return {
        "id": anchor.id, "hash": anchor.hash,
        "timestamp": anchor.timestamp,
        "valid": xycore.Verify.check(anchor)
    }


def generate_receipt(anchor_id: str) -> Optional[str]:
    """Generate shareable receipt."""
    if not HAS_XYCORE:
        return None
    anchor = xycore.Anchor.get(anchor_id)
    return anchor.receipt() if hasattr(anchor, 'receipt') else anchor.id
```

Estimated lines: ~60

-----

## 8. File Structure

```
doorway-memory/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── doorway_memory/
│       ├── __init__.py
│       ├── shape.py
│       ├── intersect.py
│       ├── library.py
│       ├── memory.py
│       └── anchor.py
└── tests/
    ├── __init__.py
    ├── test_shape.py
    ├── test_intersect.py
    ├── test_library.py
    └── test_memory.py
```

-----

## 9. Dependencies

```toml
[project]
name = "doorway-memory"
version = "0.1.0"
description = "Geometric memory engine for AI"
readme = "README.md"
requires-python = ">=3.9"
license = "Apache-2.0"

dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
anchor = ["xycore>=0.1.0"]
supabase = ["supabase>=1.0.0"]
full = ["xycore>=0.1.0", "supabase>=1.0.0"]

[project.urls]
Homepage = "https://doorwayagi.com"
Repository = "https://github.com/mintingpressbuilds/doorway-memory"
```

-----

## 10. Test Cases

```python
# test_shape.py

def test_contains_inside():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0)
    })
    assert shape.contains({"x": 5.0, "y": 5.0}) == True

def test_contains_outside():
    shape = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0)
    })
    assert shape.contains({"x": 15.0, "y": 5.0}) == False

def test_contains_boundary():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.contains({"x": 0.0}) == True
    assert shape.contains({"x": 10.0}) == True

def test_distance_inside():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.distance_to_boundary({"x": 5.0}) == 5.0
    assert shape.distance_to_boundary({"x": 2.0}) == 2.0

def test_distance_outside():
    shape = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    assert shape.distance_to_boundary({"x": 12.0}) == -2.0

# test_library.py

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
    assert lib.is_void({"x": 5.0}) == False
    assert lib.is_void({"x": 15.0}) == True

# test_memory.py

def test_store_and_recall():
    mem = Memory(anchor=False)
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert len(mem.recall({"x": 5.0})) == 1

def test_void():
    mem = Memory(anchor=False)
    mem.store(Shape(dimensions={"x": Dimension("x", 0.0, 10.0)}))
    assert mem.is_void({"x": 5.0}) == False
    assert mem.is_void({"x": 15.0}) == True
```

-----

## Build Order

1. Add `walk()` to xycore (~50 lines)
1. Build shape.py (~150 lines)
1. Build intersect.py (~60 lines)
1. Build library.py (~130 lines)
1. Build memory.py (~140 lines)
1. Build anchor.py (~60 lines)
1. Write tests (~100 lines)
1. Package setup (pyproject.toml, README)

**Total: ~690 lines · 3-4 days**

-----

## Notes

**Feature extraction alignment:** `extract_point()` must match doorway’s feature extraction. If doorway’s dimensions change, this must update.

**Chain ID convention:** Memory uses `"memory:{namespace}"` to keep memory chains separate from other xycore uses.

**Void as explicit state:** `is_void()` returning True is meaningful information, not an error. It means the state is in genuinely unknown territory.

**Future extensions:** Spatial indexing (R-tree), shape versioning, cross-namespace queries, compression for large libraries.

-----

*doorway-memory · Geometric Memory Engine · © 2026 Doorway · doorwayagi.com*
