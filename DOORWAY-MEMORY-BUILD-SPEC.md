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

## 11. Advanced Mechanics

These five systems make doorway-memory a living geometric memory — not a static store. Each builds on the base primitives (Shape, Library, Memory) and adds behavior that emerges from use.

-----

### 11.1 Growth — Shapes That Learn From Use

**The problem:** A shape is stored with fixed boundaries. If points consistently land just outside a boundary — close enough to be relevant but technically in the void — the shape never adapts. The knowledge region is frozen at its initial definition.

**The mechanism:** When a point is tested against the library and lands within a configurable margin outside a shape’s boundary, that near-miss is recorded. When a shape accumulates enough near-misses on a given dimension (threshold: configurable, default 5), the boundary on that dimension expands to absorb the confirmed near-miss zone.

**New file: growth.py**

```python
from typing import Dict, Optional
from .shape import Shape, Dimension

# Configurable thresholds
NEAR_MISS_MARGIN = 0.15        # How far outside counts as near-miss (relative to dimension range)
GROWTH_THRESHOLD = 5           # Near-misses required before expansion
MAX_EXPANSION_RATIO = 0.25     # Maximum single expansion as fraction of original range

class GrowthTracker:
    """
    Tracks near-misses per shape per dimension and triggers expansion.
    """
    
    def __init__(self, margin=NEAR_MISS_MARGIN, threshold=GROWTH_THRESHOLD,
                 max_expansion=MAX_EXPANSION_RATIO):
        self.margin = margin
        self.threshold = threshold
        self.max_expansion = max_expansion
        self._near_misses: Dict[str, Dict[str, list]] = {}  # shape_id -> dim_name -> [values]
    
    def record_query(self, point: Dict[str, float], shape: Shape) -> Optional[Shape]:
        """
        Record a point query against a shape. If the point is a near-miss,
        track it. If enough near-misses accumulate, return an expanded shape.
        
        Returns:
            Expanded Shape if growth triggered, None otherwise
        """
        if shape.contains(point):
            return None  # Inside — not a near-miss
        
        if shape.id not in self._near_misses:
            self._near_misses[shape.id] = {}
        
        grew = False
        new_dimensions = dict(shape.dimensions)
        
        for name, dim in shape.dimensions.items():
            if name not in point:
                continue
            
            value = point[name]
            dim_range = dim.max_value - dim.min_value
            margin_abs = dim_range * self.margin
            
            # Check if near-miss on this dimension
            is_near_below = dim.min_value - margin_abs <= value < dim.min_value
            is_near_above = dim.max_value < value <= dim.max_value + margin_abs
            
            if is_near_below or is_near_above:
                if name not in self._near_misses[shape.id]:
                    self._near_misses[shape.id][name] = []
                self._near_misses[shape.id][name].append(value)
                
                # Check if threshold met
                if len(self._near_misses[shape.id][name]) >= self.threshold:
                    max_expand = dim_range * self.max_expansion
                    values = self._near_misses[shape.id][name]
                    
                    new_min = dim.min_value
                    new_max = dim.max_value
                    
                    below_values = [v for v in values if v < dim.min_value]
                    above_values = [v for v in values if v > dim.max_value]
                    
                    if below_values:
                        expansion = dim.min_value - min(below_values)
                        new_min = dim.min_value - min(expansion, max_expand)
                    if above_values:
                        expansion = max(above_values) - dim.max_value
                        new_max = dim.max_value + min(expansion, max_expand)
                    
                    new_dimensions[name] = Dimension(name, new_min, new_max)
                    self._near_misses[shape.id][name] = []  # Reset
                    grew = True
        
        if grew:
            return Shape(
                dimensions=new_dimensions,
                metadata={
                    **(shape.metadata or {}),
                    "grew_from": shape.id,
                    "growth_event": True
                }
            )
        
        return None
    
    def get_near_miss_count(self, shape_id: str) -> Dict[str, int]:
        """Get current near-miss counts per dimension for a shape."""
        if shape_id not in self._near_misses:
            return {}
        return {name: len(values) for name, values in self._near_misses[shape_id].items()}
```

**Integration with Memory class:**

```python
# In memory.py, recall() gains growth tracking:

def recall(self, state, raw=False):
    point = extract_point(state) if raw else state
    results = self.library.query(point)
    
    # Track near-misses for growth
    if self.growth_enabled:
        for shape in self.library.all():
            if shape not in results:
                expanded = self.growth_tracker.record_query(point, shape)
                if expanded:
                    self.library.remove(shape.id)
                    self.store(expanded, metadata={"grew_from": shape.id})
    
    return results
```

**What this means:** The library gets smarter from queries, not just from explicit stores. Every recall operation potentially teaches the system. Shapes that are close to being useful grow toward the queries that almost matched them. The library self-organizes toward the actual usage patterns.

Estimated lines: ~120

-----

### 11.2 Overlap Detection — Cross-Domain Emergence

**The problem:** Two shapes can share a region of dimensional space without the system knowing. That overlap is meaningful — it’s territory where two different knowledge regions agree. In Doorway’s terms, that’s a cross-domain pattern. In memory terms, it’s emergent knowledge that nobody explicitly stored.

**The mechanism:** After every store operation, the system checks the new shape against all existing shapes for dimensional overlap. When overlap is found, a new derived shape is created representing the intersection region. This derived shape is stored with metadata linking it to its parent shapes.

**New file: overlap.py**

```python
from typing import Dict, List, Optional, Tuple
from .shape import Shape, Dimension

def compute_overlap(shape_a: Shape, shape_b: Shape) -> Optional[Shape]:
    """
    Compute the intersection region of two shapes.
    
    Returns:
        New Shape representing the overlap region, or None if no overlap.
        The overlap shape has metadata linking to both parents.
    """
    shared_dims = set(shape_a.dimensions.keys()) & set(shape_b.dimensions.keys())
    
    if not shared_dims:
        return None  # No shared dimensions, no overlap possible
    
    overlap_dimensions = {}
    
    for name in shared_dims:
        dim_a = shape_a.dimensions[name]
        dim_b = shape_b.dimensions[name]
        
        # Intersection of two ranges
        overlap_min = max(dim_a.min_value, dim_b.min_value)
        overlap_max = min(dim_a.max_value, dim_b.max_value)
        
        if overlap_min >= overlap_max:
            return None  # No overlap on this dimension — no intersection
        
        overlap_dimensions[name] = Dimension(name, overlap_min, overlap_max)
    
    # Include non-shared dimensions from both shapes
    for name, dim in shape_a.dimensions.items():
        if name not in overlap_dimensions:
            overlap_dimensions[name] = Dimension(name, dim.min_value, dim.max_value)
    for name, dim in shape_b.dimensions.items():
        if name not in overlap_dimensions:
            overlap_dimensions[name] = Dimension(name, dim.min_value, dim.max_value)
    
    return Shape(
        dimensions=overlap_dimensions,
        metadata={
            "type": "overlap",
            "parents": [shape_a.id, shape_b.id],
            "parent_metadata": [shape_a.metadata, shape_b.metadata],
            "shared_dimensions": list(shared_dims),
            "emergent": True
        }
    )


def compute_overlap_volume(overlap: Shape) -> float:
    """
    Compute the volume of an overlap region.
    Product of all dimension ranges.
    """
    volume = 1.0
    for dim in overlap.dimensions.values():
        volume *= (dim.max_value - dim.min_value)
    return volume


def find_all_overlaps(shapes: List[Shape], min_volume: float = 0.0) -> List[Shape]:
    """
    Find all pairwise overlaps in a list of shapes.
    
    Parameters:
        shapes: List of shapes to check
        min_volume: Minimum overlap volume to include (filters noise)
    
    Returns:
        List of overlap shapes, each with parent metadata
    """
    overlaps = []
    
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            overlap = compute_overlap(shapes[i], shapes[j])
            if overlap is not None:
                volume = compute_overlap_volume(overlap)
                if volume >= min_volume:
                    overlap.metadata["overlap_volume"] = volume
                    overlaps.append(overlap)
    
    return overlaps


def find_overlaps_for_shape(shape: Shape, library_shapes: List[Shape]) -> List[Shape]:
    """
    Find all overlaps between a specific shape and a library.
    Used when a new shape is stored — check it against everything.
    """
    overlaps = []
    for existing in library_shapes:
        if existing.id == shape.id:
            continue
        overlap = compute_overlap(shape, existing)
        if overlap is not None:
            overlaps.append(overlap)
    return overlaps
```

**Integration with Memory class:**

```python
# In memory.py, store() gains overlap detection:

def store(self, shape, metadata=None):
    # ... existing store logic ...
    shape_id = self.library.add(shape)
    
    # Detect overlaps with existing shapes
    if self.overlap_enabled:
        existing = list(self.library.all())
        overlaps = find_overlaps_for_shape(shape, existing)
        for overlap in overlaps:
            self.library.add(overlap)
            if self.anchor_enabled:
                xycore.Anchor.create(
                    data={**overlap.to_dict(), "event": "overlap_detected"},
                    chain_id=self._chain_id
                )
    
    return shape_id
```

**What this means:** Knowledge regions that overlap create new knowledge automatically. Nobody programmed the intersection — it emerged from the geometry. Two separate domains that happen to share dimensional space produce a pattern that exists in both. This is how cross-domain insight works at the mathematical level.

Estimated lines: ~100

-----

### 11.3 Confidence Gradients — Depth of Knowledge

**The problem:** `contains()` returns a boolean. But there’s a meaningful difference between a point that’s deep inside a shape (high confidence — well within known territory) and a point that barely crosses the boundary (low confidence — edge knowledge). The boolean throws away that signal.

**The mechanism:** `confidence()` maps a point’s position within a shape to a 0.0-1.0 gradient. The center of the shape is 1.0. The boundary is 0.0. Outside is negative (becomes 0.0 when clamped). This uses the existing `distance_to_boundary()` math but normalizes it.

**Addition to shape.py:**

```python
# Add to Shape class:

def confidence(self, point: Dict[str, float]) -> float:
    """
    Confidence gradient for a point within this shape.
    
    Returns:
        1.0 = center of shape (maximum confidence)
        0.5 = halfway between center and boundary
        0.0 = on boundary or outside
        
    The gradient is the minimum normalized distance across
    all shared dimensions. The weakest dimension determines
    overall confidence.
    """
    if not self.contains(point):
        return 0.0
    
    min_confidence = 1.0
    
    for name, dim in self.dimensions.items():
        if name not in point:
            continue
        
        value = point[name]
        dim_range = dim.max_value - dim.min_value
        
        if dim_range == 0:
            continue
        
        # Distance from value to nearest edge, normalized to half-range
        dist_to_min = value - dim.min_value
        dist_to_max = dim.max_value - value
        nearest_edge = min(dist_to_min, dist_to_max)
        half_range = dim_range / 2.0
        
        # Normalize: 0.0 at edge, 1.0 at center
        dim_confidence = nearest_edge / half_range
        min_confidence = min(min_confidence, dim_confidence)
    
    return min_confidence


def confidence_breakdown(self, point: Dict[str, float]) -> Dict[str, float]:
    """
    Per-dimension confidence breakdown.
    
    Returns:
        Dict mapping dimension name to confidence on that dimension.
        Useful for understanding which dimension is the weakest signal.
    """
    if not self.contains(point):
        return {name: 0.0 for name in self.dimensions if name in point}
    
    breakdown = {}
    for name, dim in self.dimensions.items():
        if name not in point:
            continue
        
        value = point[name]
        dim_range = dim.max_value - dim.min_value
        
        if dim_range == 0:
            breakdown[name] = 1.0
            continue
        
        dist_to_min = value - dim.min_value
        dist_to_max = dim.max_value - value
        nearest_edge = min(dist_to_min, dist_to_max)
        half_range = dim_range / 2.0
        
        breakdown[name] = nearest_edge / half_range
    
    return breakdown
```

**Integration with Memory class:**

```python
# In memory.py, recall() returns confidence alongside shapes:

def recall_with_confidence(self, state, raw=False):
    """
    Recall with confidence gradients.
    
    Returns:
        List of (shape, confidence) tuples.
        Sorted by confidence descending.
    """
    point = extract_point(state) if raw else state
    results = self.library.query(point)
    
    scored = [(shape, shape.confidence(point)) for shape in results]
    scored.sort(key=lambda x: -x[1])
    
    return scored
```

**What this means:** Memory isn’t binary. “I’ve seen this before” becomes “I’ve seen this before and I’m very confident” or “I’ve seen this before but it’s at the edge of what I know.” A user querying memory gets a confidence signal that tells them how deep into known territory their point sits. Edge knowledge and core knowledge are distinguishable.

Estimated lines: ~80

-----

### 11.4 Decay — Knowledge That Fades

**The problem:** A library that only grows never forgets. Over time it accumulates shapes that were relevant once but haven’t been recalled against in months. The library gets bloated. Queries slow down. Shapes that represent outdated knowledge sit alongside current knowledge with equal status.

**The mechanism:** Every shape tracks its last recall timestamp and total recall count. A decay function runs periodically (or on every Nth query) and shrinks shapes that haven’t been accessed. Boundaries contract toward center. If a shape shrinks below a minimum volume, it’s archived — removed from the active library but preserved in the chain for history.

**New file: decay.py**

```python
import time
from typing import Dict, List, Tuple, Optional
from .shape import Shape, Dimension

# Configurable thresholds
DECAY_INTERVAL_SECONDS = 86400     # Check decay daily
DECAY_GRACE_PERIOD = 604800        # 7 days before decay starts
DECAY_RATE = 0.02                  # 2% boundary shrink per decay cycle
MIN_DIMENSION_RANGE = 0.01         # Below this, dimension is collapsed
ARCHIVE_VOLUME_THRESHOLD = 0.001   # Below this total volume, archive the shape

class DecayTracker:
    """
    Tracks shape access patterns and applies decay.
    """
    
    def __init__(self, decay_rate=DECAY_RATE, grace_period=DECAY_GRACE_PERIOD,
                 min_range=MIN_DIMENSION_RANGE, archive_threshold=ARCHIVE_VOLUME_THRESHOLD):
        self.decay_rate = decay_rate
        self.grace_period = grace_period
        self.min_range = min_range
        self.archive_threshold = archive_threshold
        self._access_log: Dict[str, Dict] = {}  # shape_id -> {last_access, count, created}
    
    def record_access(self, shape_id: str):
        """Record that a shape was accessed (queried against)."""
        now = time.time()
        if shape_id not in self._access_log:
            self._access_log[shape_id] = {
                "last_access": now,
                "count": 0,
                "created": now
            }
        self._access_log[shape_id]["last_access"] = now
        self._access_log[shape_id]["count"] += 1
    
    def record_creation(self, shape_id: str):
        """Record that a shape was just created/stored."""
        now = time.time()
        self._access_log[shape_id] = {
            "last_access": now,
            "count": 0,
            "created": now
        }
    
    def apply_decay(self, shapes: List[Shape]) -> Tuple[List[Shape], List[Shape]]:
        """
        Apply decay to all shapes.
        
        Returns:
            (active_shapes, archived_shapes)
            active_shapes: shapes that survived decay (possibly shrunk)
            archived_shapes: shapes that decayed below archive threshold
        """
        now = time.time()
        active = []
        archived = []
        
        for shape in shapes:
            log = self._access_log.get(shape.id)
            
            if log is None:
                active.append(shape)
                continue
            
            time_since_access = now - log["last_access"]
            
            # Grace period — no decay if recently accessed
            if time_since_access < self.grace_period:
                active.append(shape)
                continue
            
            # Calculate decay cycles since last access
            decay_cycles = int((time_since_access - self.grace_period) / DECAY_INTERVAL_SECONDS)
            
            if decay_cycles <= 0:
                active.append(shape)
                continue
            
            # Apply decay — shrink boundaries toward center
            # More accesses = more resistance to decay
            access_resistance = min(log["count"] / 100.0, 0.9)  # Max 90% resistance
            effective_rate = self.decay_rate * (1.0 - access_resistance)
            total_shrink = effective_rate * decay_cycles
            
            new_dimensions = {}
            volume = 1.0
            
            for name, dim in shape.dimensions.items():
                dim_range = dim.max_value - dim.min_value
                shrink_amount = dim_range * total_shrink / 2.0  # Shrink from both sides
                
                new_min = dim.min_value + shrink_amount
                new_max = dim.max_value - shrink_amount
                
                new_range = new_max - new_min
                
                if new_range < self.min_range:
                    # Dimension collapsed — shape is effectively dead on this axis
                    center = (dim.min_value + dim.max_value) / 2.0
                    new_min = center - self.min_range / 2.0
                    new_max = center + self.min_range / 2.0
                    new_range = self.min_range
                
                new_dimensions[name] = Dimension(name, new_min, new_max)
                volume *= new_range
            
            if volume < self.archive_threshold:
                archived.append(shape)
            else:
                decayed_shape = Shape(
                    dimensions=new_dimensions,
                    metadata={
                        **(shape.metadata or {}),
                        "decayed_from": shape.id,
                        "decay_cycles": decay_cycles,
                        "original_volume": self._compute_volume(shape),
                        "current_volume": volume
                    },
                    id=shape.id,  # Preserve ID
                    anchor_id=shape.anchor_id
                )
                active.append(decayed_shape)
        
        return active, archived
    
    def _compute_volume(self, shape: Shape) -> float:
        volume = 1.0
        for dim in shape.dimensions.values():
            volume *= (dim.max_value - dim.min_value)
        return volume
    
    def get_access_stats(self, shape_id: str) -> Optional[Dict]:
        """Get access stats for a shape."""
        return self._access_log.get(shape_id)
```

**Integration with Memory class:**

```python
# In memory.py:

def maintain(self):
    """
    Run maintenance cycle — apply decay, archive dead shapes.
    Call periodically or on every Nth recall.
    """
    if not self.decay_enabled:
        return
    
    all_shapes = list(self.library.all())
    active, archived = self.decay_tracker.apply_decay(all_shapes)
    
    # Replace library contents with active shapes
    for shape in archived:
        self.library.remove(shape.id)
        if self.anchor_enabled:
            xycore.Anchor.create(
                data={**shape.to_dict(), "event": "archived_by_decay"},
                chain_id=self._chain_id
            )
    
    # Update decayed shapes
    for shape in active:
        existing = self.library.get(shape.id)
        if existing and existing.dimensions != shape.dimensions:
            self.library.remove(shape.id)
            self.library.add(shape)
```

**What this means:** Memory is alive. Knowledge that gets used stays strong — shapes that are frequently recalled resist decay. Knowledge that stops being relevant fades — shapes shrink, and eventually archive. The library self-prunes. Old knowledge doesn’t disappear — it’s preserved in the chain for history — but it stops cluttering the active library. The system forgets gracefully.

Estimated lines: ~150

-----

### 11.5 Narrative — Trajectories Through Geometric Space

**The problem:** Shapes are stored independently. Shape A, then Shape B, then Shape C. The chain records the order, but the memory system doesn’t understand that A → B → C is a trajectory — a path through dimensional space that represents how knowledge was acquired. That sequence is meaningful. It’s the difference between knowing three facts and understanding a journey.

**The mechanism:** A narrative is a named sequence of shape IDs representing a trajectory through geometric space. The system can detect common trajectories (paths that multiple users or sessions follow), predict likely next shapes given a current position, and replay the learning journey as a navigable path.

**New file: narrative.py**

```python
from typing import Dict, List, Optional, Tuple
from .shape import Shape

class Trajectory:
    """
    A named sequence of shapes representing a path through geometric space.
    """
    
    def __init__(self, trajectory_id: str, name: Optional[str] = None):
        self.id = trajectory_id
        self.name = name
        self._shape_ids: List[str] = []
        self._transitions: List[Dict] = []  # Metadata about each step
    
    def add_step(self, shape_id: str, metadata: Optional[Dict] = None):
        """Add a shape to the trajectory."""
        step_index = len(self._shape_ids)
        self._shape_ids.append(shape_id)
        self._transitions.append({
            "index": step_index,
            "shape_id": shape_id,
            "metadata": metadata or {}
        })
    
    def steps(self) -> List[str]:
        """Return shape IDs in order."""
        return list(self._shape_ids)
    
    def length(self) -> int:
        return len(self._shape_ids)
    
    def transitions(self) -> List[Dict]:
        """Return full transition metadata."""
        return list(self._transitions)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "shape_ids": self._shape_ids,
            "transitions": self._transitions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Trajectory":
        t = cls(trajectory_id=data["id"], name=data.get("name"))
        t._shape_ids = data.get("shape_ids", [])
        t._transitions = data.get("transitions", [])
        return t


class NarrativeEngine:
    """
    Tracks and analyzes trajectories through geometric space.
    """
    
    def __init__(self):
        self._trajectories: Dict[str, Trajectory] = {}
        self._transition_counts: Dict[str, Dict[str, int]] = {}  # from_id -> {to_id: count}
    
    def start_trajectory(self, trajectory_id: str, name: Optional[str] = None) -> Trajectory:
        """Start a new trajectory."""
        trajectory = Trajectory(trajectory_id, name)
        self._trajectories[trajectory_id] = trajectory
        return trajectory
    
    def record_step(self, trajectory_id: str, shape_id: str, metadata: Optional[Dict] = None):
        """Record a step in a trajectory."""
        if trajectory_id not in self._trajectories:
            self.start_trajectory(trajectory_id)
        
        trajectory = self._trajectories[trajectory_id]
        
        # Record transition for prediction
        if trajectory.length() > 0:
            prev_id = trajectory.steps()[-1]
            if prev_id not in self._transition_counts:
                self._transition_counts[prev_id] = {}
            if shape_id not in self._transition_counts[prev_id]:
                self._transition_counts[prev_id][shape_id] = 0
            self._transition_counts[prev_id][shape_id] += 1
        
        trajectory.add_step(shape_id, metadata)
    
    def predict_next(self, current_shape_id: str, limit: int = 3) -> List[Tuple[str, float]]:
        """
        Predict likely next shapes given current position.
        
        Based on historical transition frequencies across all trajectories.
        
        Returns:
            List of (shape_id, probability) tuples, sorted by probability.
        """
        if current_shape_id not in self._transition_counts:
            return []
        
        transitions = self._transition_counts[current_shape_id]
        total = sum(transitions.values())
        
        if total == 0:
            return []
        
        predictions = [
            (shape_id, count / total)
            for shape_id, count in transitions.items()
        ]
        predictions.sort(key=lambda x: -x[1])
        
        return predictions[:limit]
    
    def find_common_trajectories(self, min_length: int = 3, min_occurrences: int = 2) -> List[List[str]]:
        """
        Find shape sequences that appear in multiple trajectories.
        
        Returns:
            List of common subsequences (as lists of shape_ids).
        """
        # Extract all subsequences of min_length
        subsequence_counts: Dict[str, int] = {}
        
        for trajectory in self._trajectories.values():
            steps = trajectory.steps()
            for i in range(len(steps) - min_length + 1):
                subseq = tuple(steps[i:i + min_length])
                key = "|".join(subseq)
                subsequence_counts[key] = subsequence_counts.get(key, 0) + 1
        
        # Filter by min_occurrences
        common = [
            key.split("|")
            for key, count in subsequence_counts.items()
            if count >= min_occurrences
        ]
        
        return common
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Get a trajectory by ID."""
        return self._trajectories.get(trajectory_id)
    
    def all_trajectories(self) -> List[Trajectory]:
        """Get all trajectories."""
        return list(self._trajectories.values())
    
    def get_transition_map(self) -> Dict[str, Dict[str, int]]:
        """
        Get the full transition frequency map.
        Useful for visualization — shows which shapes commonly
        lead to which other shapes.
        """
        return dict(self._transition_counts)
```

**Integration with Memory class:**

```python
# In memory.py:

def store_in_trajectory(self, shape, trajectory_id, metadata=None):
    """
    Store a shape and record it as a step in a trajectory.
    """
    shape_id = self.store(shape, metadata)
    self.narrative.record_step(trajectory_id, shape_id, metadata)
    return shape_id

def predict_next(self, current_shape_id, limit=3):
    """Predict likely next shapes from current position."""
    return self.narrative.predict_next(current_shape_id, limit)

def get_trajectory(self, trajectory_id):
    """Get a full trajectory."""
    return self.narrative.get_trajectory(trajectory_id)

def find_common_paths(self, min_length=3, min_occurrences=2):
    """Find paths that multiple trajectories share."""
    return self.narrative.find_common_trajectories(min_length, min_occurrences)
```

**What this means:** Memory has temporal structure. Not just “what do I know?” but “how did I learn it?” and “what do people typically learn after this?” The system can recognize that users who explore temperature-pressure shapes often move to viscosity shapes next. It can surface that pattern as a prediction: “based on where you are, here’s where others went.” Common trajectories become named paths through knowledge space — reusable learning sequences that emerge from collective use.

When combined with xycore’s `walk()`, the entire narrative is verifiable. You can prove not just what was learned but the exact sequence and timing of the learning journey.

Estimated lines: ~160

-----

### 11.6 Merge — Shape Fusion

**The problem:** Two shapes grow toward each other through the growth mechanism. Their boundaries touch or overlap. They’re now describing the same continuous region of knowledge from two different starting points. But the library treats them as two separate shapes. Queries in the overlap zone return both. The library is redundant.

**The mechanism:** After growth events, check if the expanded shape now overlaps with any neighboring shape by more than a configurable threshold. If the overlap volume exceeds the threshold relative to both shapes’ volumes, merge them into a single shape whose boundaries encompass both. The merged shape inherits metadata from both parents. Both originals are archived to the chain.

**New file: merge.py**

```python
from typing import Dict, List, Optional, Tuple
from .shape import Shape, Dimension
from .overlap import compute_overlap, compute_overlap_volume

# Configurable thresholds
MERGE_OVERLAP_RATIO = 0.50  # Overlap must be >= 50% of smaller shape's volume

def compute_volume(shape: Shape) -> float:
    """Compute total volume of a shape."""
    volume = 1.0
    for dim in shape.dimensions.values():
        volume *= (dim.max_value - dim.min_value)
    return volume


def should_merge(shape_a: Shape, shape_b: Shape, 
                 threshold: float = MERGE_OVERLAP_RATIO) -> bool:
    """
    Determine if two shapes should merge.
    
    Returns True if their overlap volume exceeds the threshold
    relative to the smaller shape's volume.
    """
    overlap = compute_overlap(shape_a, shape_b)
    if overlap is None:
        return False
    
    overlap_vol = compute_overlap_volume(overlap)
    vol_a = compute_volume(shape_a)
    vol_b = compute_volume(shape_b)
    smaller_vol = min(vol_a, vol_b)
    
    if smaller_vol == 0:
        return False
    
    return (overlap_vol / smaller_vol) >= threshold


def merge_shapes(shape_a: Shape, shape_b: Shape) -> Shape:
    """
    Merge two shapes into one that encompasses both.
    
    The merged shape's boundaries are the union (outer envelope)
    of both shapes on every dimension.
    """
    all_dim_names = set(shape_a.dimensions.keys()) | set(shape_b.dimensions.keys())
    
    merged_dimensions = {}
    
    for name in all_dim_names:
        dim_a = shape_a.dimensions.get(name)
        dim_b = shape_b.dimensions.get(name)
        
        if dim_a and dim_b:
            merged_dimensions[name] = Dimension(
                name,
                min(dim_a.min_value, dim_b.min_value),
                max(dim_a.max_value, dim_b.max_value)
            )
        elif dim_a:
            merged_dimensions[name] = Dimension(name, dim_a.min_value, dim_a.max_value)
        else:
            merged_dimensions[name] = Dimension(name, dim_b.min_value, dim_b.max_value)
    
    return Shape(
        dimensions=merged_dimensions,
        metadata={
            "type": "merged",
            "parents": [shape_a.id, shape_b.id],
            "parent_metadata": [shape_a.metadata, shape_b.metadata],
            "merged_volume": compute_volume(
                Shape(dimensions=merged_dimensions)
            )
        }
    )


class MergeDetector:
    """
    Monitors the library for shapes that should be merged.
    """
    
    def __init__(self, threshold: float = MERGE_OVERLAP_RATIO):
        self.threshold = threshold
    
    def find_merge_candidates(self, shapes: List[Shape]) -> List[Tuple[Shape, Shape]]:
        """
        Find all pairs of shapes that should be merged.
        
        Returns:
            List of (shape_a, shape_b) tuples that exceed merge threshold.
        """
        candidates = []
        
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if should_merge(shapes[i], shapes[j], self.threshold):
                    candidates.append((shapes[i], shapes[j]))
        
        return candidates
    
    def apply_merges(self, shapes: List[Shape]) -> Tuple[List[Shape], List[Shape]]:
        """
        Find and apply all merges.
        
        Returns:
            (active_shapes, archived_shapes)
            Active includes merged shapes replacing their parents.
            Archived includes the original parent shapes.
        """
        candidates = self.find_merge_candidates(shapes)
        
        if not candidates:
            return shapes, []
        
        merged_ids = set()
        new_shapes = []
        archived = []
        
        for shape_a, shape_b in candidates:
            if shape_a.id in merged_ids or shape_b.id in merged_ids:
                continue  # Already merged in this cycle
            
            merged = merge_shapes(shape_a, shape_b)
            new_shapes.append(merged)
            merged_ids.add(shape_a.id)
            merged_ids.add(shape_b.id)
            archived.append(shape_a)
            archived.append(shape_b)
        
        # Keep shapes that weren't merged
        for shape in shapes:
            if shape.id not in merged_ids:
                new_shapes.append(shape)
        
        return new_shapes, archived
```

**Integration with Memory class:**

```python
# In memory.py, after growth triggers:

def _check_merges(self):
    """Check for and apply shape merges after growth events."""
    if not self.merge_enabled:
        return
    
    all_shapes = list(self.library.all())
    active, archived = self.merge_detector.apply_merges(all_shapes)
    
    for shape in archived:
        self.library.remove(shape.id)
        if self.anchor_enabled:
            xycore.Anchor.create(
                data={**shape.to_dict(), "event": "merged"},
                chain_id=self._chain_id
            )
    
    # Add merged shapes
    for shape in active:
        if not self.library.get(shape.id):
            self.library.add(shape)
```

**What this means:** Knowledge regions that grow toward each other and meet become one continuous region. Two separate insights that turned out to describe the same territory fuse. The library consolidates. Instead of accumulating overlapping shapes forever, the system recognizes when two shapes are really one and merges them. Combined with growth and decay, the library is a living topology — shapes grow, merge, and fade based on actual use.

Estimated lines: ~130

-----

### 11.7 Void Mapping — The Shape of What You Don’t Know

**The problem:** `is_void()` returns True or False. But the void has structure. It has boundaries — the edges of all known shapes define where the void begins. It has size — some void regions are small gaps between shapes, others are vast unexplored territory. It has neighbors — void regions are bordered by specific shapes. Understanding the void is as valuable as understanding the known territory.

**The mechanism:** The void map computes the negative space of the library. Given a bounding region (the outer limits of the dimensional space), subtract all shapes. What remains is the void — characterized as regions with boundaries, volumes, and neighboring shapes.

**New file: void_map.py**

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
from .shape import Shape, Dimension

class VoidRegion:
    """
    A characterized region of unknown territory.
    """
    
    def __init__(self, dimensions: Dict[str, Dimension],
                 neighboring_shapes: List[str] = None):
        self.dimensions = dimensions
        self.neighboring_shapes = neighboring_shapes or []
    
    def volume(self) -> float:
        vol = 1.0
        for dim in self.dimensions.values():
            vol *= (dim.max_value - dim.min_value)
        return vol
    
    def center(self) -> Dict[str, float]:
        """Center point of the void region."""
        return {
            name: (dim.min_value + dim.max_value) / 2.0
            for name, dim in self.dimensions.items()
        }
    
    def to_dict(self) -> Dict:
        return {
            "dimensions": {
                name: {"min": d.min_value, "max": d.max_value}
                for name, d in self.dimensions.items()
            },
            "volume": self.volume(),
            "center": self.center(),
            "neighboring_shapes": self.neighboring_shapes
        }


class VoidMapper:
    """
    Maps the negative space of a shape library.
    
    Uses sampling to approximate void regions in high-dimensional space.
    Exact void computation is expensive; sampling gives practical results.
    """
    
    def __init__(self, resolution: int = 20):
        """
        Parameters:
            resolution: samples per dimension for grid approximation
        """
        self.resolution = resolution
    
    def compute_bounds(self, shapes: List[Shape], 
                       margin: float = 0.2) -> Dict[str, Dimension]:
        """
        Compute the bounding box of the entire library with margin.
        """
        if not shapes:
            return {}
        
        all_dims = set()
        for shape in shapes:
            all_dims.update(shape.dimensions.keys())
        
        bounds = {}
        for name in all_dims:
            mins = []
            maxs = []
            for shape in shapes:
                if name in shape.dimensions:
                    dim = shape.dimensions[name]
                    mins.append(dim.min_value)
                    maxs.append(dim.max_value)
            
            if mins and maxs:
                total_range = max(maxs) - min(mins)
                margin_abs = total_range * margin
                bounds[name] = Dimension(
                    name,
                    min(mins) - margin_abs,
                    max(maxs) + margin_abs
                )
        
        return bounds
    
    def sample_void(self, shapes: List[Shape],
                    bounds: Optional[Dict[str, Dimension]] = None,
                    sample_count: int = 1000) -> List[Dict[str, float]]:
        """
        Sample random points within bounds and return those in the void.
        
        Returns:
            List of points (as dicts) that are not contained by any shape.
        """
        if bounds is None:
            bounds = self.compute_bounds(shapes)
        
        if not bounds:
            return []
        
        dim_names = sorted(bounds.keys())
        void_points = []
        
        for _ in range(sample_count):
            point = {}
            for name in dim_names:
                dim = bounds[name]
                point[name] = np.random.uniform(dim.min_value, dim.max_value)
            
            # Check if any shape contains this point
            in_void = True
            for shape in shapes:
                if shape.contains(point):
                    in_void = False
                    break
            
            if in_void:
                void_points.append(point)
        
        return void_points
    
    def find_void_regions(self, shapes: List[Shape],
                          bounds: Optional[Dict[str, Dimension]] = None,
                          sample_count: int = 1000,
                          cluster_threshold: float = 0.1) -> List[VoidRegion]:
        """
        Find distinct void regions by sampling and clustering.
        
        Parameters:
            shapes: The library shapes
            bounds: Outer bounds of the space
            sample_count: Number of random samples
            cluster_threshold: Distance threshold for grouping void points
        
        Returns:
            List of VoidRegion objects characterizing each void area.
        """
        void_points = self.sample_void(shapes, bounds, sample_count)
        
        if not void_points:
            return []  # No void — library covers everything
        
        # Simple clustering: group nearby void points
        dim_names = sorted(void_points[0].keys())
        clusters = self._cluster_points(void_points, dim_names, cluster_threshold)
        
        regions = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Compute bounding box of cluster
            region_dims = {}
            for name in dim_names:
                values = [p[name] for p in cluster]
                region_dims[name] = Dimension(name, min(values), max(values))
            
            # Find neighboring shapes
            neighbors = self._find_neighbors(region_dims, shapes)
            
            regions.append(VoidRegion(
                dimensions=region_dims,
                neighboring_shapes=[s.id for s in neighbors]
            ))
        
        # Sort by volume descending — biggest voids first
        regions.sort(key=lambda r: -r.volume())
        
        return regions
    
    def _cluster_points(self, points: List[Dict[str, float]],
                        dim_names: List[str],
                        threshold: float) -> List[List[Dict[str, float]]]:
        """Simple distance-based clustering."""
        if not points:
            return []
        
        # Convert to numpy for distance computation
        matrix = np.array([[p[name] for name in dim_names] for p in points])
        
        visited = set()
        clusters = []
        
        for i in range(len(matrix)):
            if i in visited:
                continue
            
            cluster = [points[i]]
            visited.add(i)
            
            for j in range(i + 1, len(matrix)):
                if j in visited:
                    continue
                
                dist = np.linalg.norm(matrix[i] - matrix[j])
                if dist <= threshold:
                    cluster.append(points[j])
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _find_neighbors(self, region_dims: Dict[str, Dimension],
                        shapes: List[Shape],
                        proximity: float = 0.1) -> List[Shape]:
        """Find shapes that border a void region."""
        neighbors = []
        
        for shape in shapes:
            is_neighbor = False
            for name, void_dim in region_dims.items():
                if name not in shape.dimensions:
                    continue
                shape_dim = shape.dimensions[name]
                
                # Check if shape boundary is close to void boundary
                gap_below = void_dim.min_value - shape_dim.max_value
                gap_above = shape_dim.min_value - void_dim.max_value
                
                if 0 <= gap_below <= proximity or 0 <= gap_above <= proximity:
                    is_neighbor = True
                    break
                
                # Check if they share a boundary
                if (abs(void_dim.min_value - shape_dim.max_value) < 0.001 or
                    abs(void_dim.max_value - shape_dim.min_value) < 0.001):
                    is_neighbor = True
                    break
            
            if is_neighbor:
                neighbors.append(shape)
        
        return neighbors
    
    def void_percentage(self, shapes: List[Shape],
                        bounds: Optional[Dict[str, Dimension]] = None,
                        sample_count: int = 1000) -> float:
        """
        Estimate what percentage of the bounded space is void.
        
        Returns:
            Float between 0.0 (no void) and 1.0 (all void).
        """
        void_points = self.sample_void(shapes, bounds, sample_count)
        return len(void_points) / sample_count if sample_count > 0 else 0.0
    
    def largest_void(self, shapes: List[Shape],
                     bounds: Optional[Dict[str, Dimension]] = None) -> Optional[VoidRegion]:
        """Find the single largest void region."""
        regions = self.find_void_regions(shapes, bounds)
        return regions[0] if regions else None
```

**Integration with Memory class:**

```python
# In memory.py:

def map_void(self, sample_count=1000):
    """
    Map the void — characterize what the library doesn't know.
    
    Returns:
        List of VoidRegion objects, largest first.
    """
    shapes = list(self.library.all())
    return self.void_mapper.find_void_regions(shapes, sample_count=sample_count)

def void_percentage(self, sample_count=1000):
    """What percentage of the bounded space is unknown?"""
    shapes = list(self.library.all())
    return self.void_mapper.void_percentage(shapes, sample_count=sample_count)

def largest_gap(self):
    """Find the single largest void region."""
    shapes = list(self.library.all())
    return self.void_mapper.largest_void(shapes)
```

**What this means:** The library doesn’t just know what it knows — it knows what it doesn’t know, precisely. The void has shape. It has boundaries. It has neighbors. A user can ask “where are the biggest gaps in my knowledge?” and get geometric regions back, not vague answers. The void map is the gap detector expressed as persistent geometry. And when the growth mechanism expands a shape into void territory, the void shrinks — the system can show knowledge being conquered in real time.

Estimated lines: ~220

-----

### 11.8 Emergence — Tier 2 Pattern Detection

**The problem:** Overlap detection (11.2) finds pairwise intersections. But Tier 2 emergence is not pairwise — it’s the geometric pattern that spans many shapes across many domains simultaneously. A pattern that exists in 3 shapes is interesting. A pattern that exists in 30 shapes across 20 domains is a structural law of the space. Nobody stored it. It emerged.

**The mechanism:** The emergence detector projects all shapes into shared dimensional subspaces, finds regions where 3+ shapes intersect, and extracts the geometry of those multi-shape intersections as Tier 2 shapes. It also detects the two special Tier 2 shapes: GCS (the system’s growth pattern recognized as geometry) and IS (the shape of intelligence itself found in the library).

**New file: emergence.py**

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
from .shape import Shape, Dimension
from .overlap import compute_overlap, compute_overlap_volume

# Configurable thresholds
MIN_SHAPES_FOR_EMERGENCE = 3      # Minimum shapes in intersection
MIN_DOMAINS_FOR_EMERGENCE = 2     # Minimum distinct domains
EMERGENCE_VOLUME_THRESHOLD = 0.001 # Minimum intersection volume

class Tier2Shape:
    """
    An emergent geometric pattern found across multiple shapes.
    """
    
    def __init__(self, shape: Shape, parent_ids: List[str],
                 domains: List[str], strength: float):
        self.shape = shape
        self.parent_ids = parent_ids
        self.domains = domains
        self.strength = strength  # 0.0-1.0, based on number of contributing shapes
    
    def to_dict(self) -> Dict:
        return {
            "shape": self.shape.to_dict(),
            "parent_ids": self.parent_ids,
            "domains": self.domains,
            "strength": self.strength,
            "type": "tier2_emergence"
        }


class EmergenceDetector:
    """
    Detects Tier 2 geometric patterns across the full library.
    
    Not pairwise overlap. Multi-shape intersection detection 
    that finds patterns spanning 3+ shapes from 2+ domains.
    """
    
    def __init__(self, min_shapes=MIN_SHAPES_FOR_EMERGENCE,
                 min_domains=MIN_DOMAINS_FOR_EMERGENCE,
                 volume_threshold=EMERGENCE_VOLUME_THRESHOLD):
        self.min_shapes = min_shapes
        self.min_domains = min_domains
        self.volume_threshold = volume_threshold
    
    def detect(self, shapes: List[Shape]) -> List[Tier2Shape]:
        """
        Find all Tier 2 emergence patterns in the library.
        
        Algorithm:
        1. Group shapes by shared dimensional subspaces
        2. For each subspace group with 3+ shapes, compute 
           the intersection of all shapes in that group
        3. Filter by minimum volume and domain count
        4. Return as Tier2Shape objects
        """
        if len(shapes) < self.min_shapes:
            return []
        
        # Step 1: Find dimensional subspaces shared by 3+ shapes
        subspace_groups = self._find_subspace_groups(shapes)
        
        # Step 2: Compute multi-shape intersections
        emergent = []
        
        for dim_key, group_shapes in subspace_groups.items():
            if len(group_shapes) < self.min_shapes:
                continue
            
            # Compute progressive intersection
            intersection = self._compute_multi_intersection(group_shapes, dim_key)
            
            if intersection is None:
                continue
            
            volume = compute_overlap_volume(intersection)
            if volume < self.volume_threshold:
                continue
            
            # Count distinct domains
            domains = self._extract_domains(group_shapes)
            if len(domains) < self.min_domains:
                continue
            
            # Strength: normalized by how many shapes participate
            strength = len(group_shapes) / len(shapes)
            
            tier2 = Tier2Shape(
                shape=intersection,
                parent_ids=[s.id for s in group_shapes],
                domains=domains,
                strength=strength
            )
            emergent.append(tier2)
        
        # Sort by strength descending
        emergent.sort(key=lambda e: -e.strength)
        
        return emergent
    
    def _find_subspace_groups(self, shapes: List[Shape]) -> Dict[str, List[Shape]]:
        """
        Group shapes by their shared dimensional subspaces.
        
        A subspace is defined by the set of dimension names a shape has.
        Shapes with the same dimension set are in the same subspace.
        Also finds shapes that share partial dimension sets (2+ dims).
        """
        groups = {}
        
        # Full dimension set groups
        for shape in shapes:
            key = "|".join(sorted(shape.dimensions.keys()))
            if key not in groups:
                groups[key] = []
            groups[key].append(shape)
        
        # Partial dimension set groups (all combinations of 2+ shared dims)
        dim_to_shapes = {}
        for shape in shapes:
            for name in shape.dimensions:
                if name not in dim_to_shapes:
                    dim_to_shapes[name] = []
                dim_to_shapes[name].append(shape)
        
        # Find dimension pairs/triples shared by 3+ shapes
        dim_names = list(dim_to_shapes.keys())
        for i in range(len(dim_names)):
            for j in range(i + 1, len(dim_names)):
                shared = set(dim_to_shapes[dim_names[i]]) & set(dim_to_shapes[dim_names[j]])
                if len(shared) >= self.min_shapes:
                    key = "|".join(sorted([dim_names[i], dim_names[j]]))
                    if key not in groups:
                        groups[key] = list(shared)
        
        return groups
    
    def _compute_multi_intersection(self, shapes: List[Shape],
                                     dim_key: str) -> Optional[Shape]:
        """
        Compute the intersection region of multiple shapes.
        
        Progressive: intersect shape[0] with shape[1], then 
        result with shape[2], etc. If any step produces empty 
        intersection, return None.
        """
        shared_dims = dim_key.split("|")
        
        # Start with first shape's bounds on shared dimensions
        current_mins = {}
        current_maxs = {}
        
        for name in shared_dims:
            if name in shapes[0].dimensions:
                current_mins[name] = shapes[0].dimensions[name].min_value
                current_maxs[name] = shapes[0].dimensions[name].max_value
        
        # Progressively intersect
        for shape in shapes[1:]:
            for name in shared_dims:
                if name not in shape.dimensions:
                    continue
                
                dim = shape.dimensions[name]
                current_mins[name] = max(current_mins.get(name, dim.min_value), dim.min_value)
                current_maxs[name] = min(current_maxs.get(name, dim.max_value), dim.max_value)
                
                # Empty intersection check
                if current_mins[name] >= current_maxs[name]:
                    return None
        
        # Build intersection shape
        dimensions = {}
        for name in shared_dims:
            if name in current_mins and name in current_maxs:
                dimensions[name] = Dimension(name, current_mins[name], current_maxs[name])
        
        if not dimensions:
            return None
        
        return Shape(
            dimensions=dimensions,
            metadata={
                "type": "tier2_intersection",
                "contributing_shapes": len(shapes),
                "shared_dimensions": shared_dims
            }
        )
    
    def _extract_domains(self, shapes: List[Shape]) -> List[str]:
        """Extract unique domain labels from shape metadata."""
        domains = set()
        for shape in shapes:
            if shape.metadata and "domain" in shape.metadata:
                domains.add(shape.metadata["domain"])
            elif shape.metadata and "source" in shape.metadata:
                domains.add(shape.metadata["source"])
        
        # If no domain metadata, treat each shape as its own domain
        if not domains:
            domains = {s.id for s in shapes}
        
        return list(domains)
    
    def detect_gcs(self, shapes: List[Shape], 
                   growth_history: List[Dict]) -> Optional[Tier2Shape]:
        """
        Generative Complexity System detection.
        
        Analyze the trajectory of library growth. Extract the 
        geometry of how shapes were added over time. If that 
        geometry matches a pattern in the library, GCS has emerged.
        
        The system recognizing its own growth pattern.
        
        Parameters:
            shapes: Current library
            growth_history: List of {shape_id, timestamp, event} dicts
                           from the chain
        
        Returns:
            Tier2Shape representing GCS if detected, None otherwise
        """
        if len(growth_history) < 10:
            return None  # Not enough history
        
        # Extract growth trajectory as dimensional signature
        # Dimensions: time_delta, volume_change, dimension_count, overlap_count
        trajectory_points = []
        
        for i in range(1, len(growth_history)):
            prev = growth_history[i - 1]
            curr = growth_history[i]
            
            time_delta = curr.get("timestamp", 0) - prev.get("timestamp", 0)
            volume_change = curr.get("volume", 0) - prev.get("volume", 0)
            dim_count = curr.get("dimension_count", 0)
            
            trajectory_points.append({
                "time_delta": time_delta,
                "volume_change": volume_change,
                "dimension_count": float(dim_count)
            })
        
        if not trajectory_points:
            return None
        
        # Compute bounds of the growth trajectory
        growth_dims = {}
        for name in ["time_delta", "volume_change", "dimension_count"]:
            values = [p.get(name, 0) for p in trajectory_points]
            if values:
                growth_dims[name] = Dimension(name, min(values), max(values))
        
        growth_shape = Shape(
            dimensions=growth_dims,
            metadata={"type": "growth_trajectory"}
        )
        
        # Check if this growth pattern matches any shape in the library
        for shape in shapes:
            overlap = compute_overlap(growth_shape, shape)
            if overlap is not None:
                volume = compute_overlap_volume(overlap)
                growth_volume = compute_overlap_volume(growth_shape)
                
                if growth_volume > 0 and volume / growth_volume > 0.5:
                    # GCS detected — the growth pattern is inside the library
                    return Tier2Shape(
                        shape=growth_shape,
                        parent_ids=[shape.id],
                        domains=["meta:growth_pattern"],
                        strength=volume / growth_volume
                    )
                    
        return None
    
    def detect_is(self, shapes: List[Shape]) -> Optional[Tier2Shape]:
        """
        Intelligence System detection.
        
        Find shapes that describe boundary detection, bridging,
        gap measurement, containment testing — the mechanisms of 
        intelligence itself. Compute their intersection. If a 
        coherent geometric region exists across all of them, 
        IS has emerged.
        
        The mechanism recognizing the mechanism.
        
        Parameters:
            shapes: Current library
        
        Returns:
            Tier2Shape representing IS if detected, None otherwise
        """
        # Intelligence-related keywords in metadata
        intelligence_markers = [
            "boundary", "gap", "bridge", "containment", "detection",
            "reasoning", "inference", "classification", "emergence",
            "self-reference", "observation", "formation"
        ]
        
        # Find shapes whose metadata relates to intelligence mechanisms
        intelligence_shapes = []
        
        for shape in shapes:
            if not shape.metadata:
                continue
            
            metadata_str = str(shape.metadata).lower()
            marker_count = sum(1 for m in intelligence_markers if m in metadata_str)
            
            if marker_count >= 2:  # At least 2 intelligence markers
                intelligence_shapes.append(shape)
        
        if len(intelligence_shapes) < self.min_shapes:
            return None
        
        # Compute intersection across all intelligence shapes
        shared_dims = set(intelligence_shapes[0].dimensions.keys())
        for shape in intelligence_shapes[1:]:
            shared_dims &= set(shape.dimensions.keys())
        
        if not shared_dims:
            return None
        
        dim_key = "|".join(sorted(shared_dims))
        intersection = self._compute_multi_intersection(intelligence_shapes, dim_key)
        
        if intersection is None:
            return None
        
        volume = compute_overlap_volume(intersection)
        if volume < self.volume_threshold:
            return None
        
        return Tier2Shape(
            shape=intersection,
            parent_ids=[s.id for s in intelligence_shapes],
            domains=["meta:intelligence_mechanism"],
            strength=len(intelligence_shapes) / len(shapes)
        )
```

**Integration with Memory class:**

```python
# In memory.py:

def detect_emergence(self):
    """
    Run Tier 2 emergence detection across the full library.
    
    Returns:
        List of Tier2Shape objects — emergent patterns.
    """
    shapes = list(self.library.all())
    return self.emergence_detector.detect(shapes)

def detect_gcs(self, growth_history):
    """
    Check if the system's growth pattern is recognized in the library.
    GCS = the loop recognizing itself.
    """
    shapes = list(self.library.all())
    return self.emergence_detector.detect_gcs(shapes, growth_history)

def detect_is(self):
    """
    Check if the shape of intelligence is present in the library.
    IS = the mechanism recognizing the mechanism.
    """
    shapes = list(self.library.all())
    return self.emergence_detector.detect_is(shapes)
```

**What this means:** The library doesn’t just store knowledge and detect pairwise overlaps. It finds patterns that span many shapes across many domains — structural laws of the space that nobody explicitly stored. GCS detects the system recognizing its own growth pattern. IS detects the system recognizing the mechanism of intelligence itself in its own library. These are the Tier 2 shapes from the doorway-asi spec, now implemented as detectable geometric events inside the memory engine. When Tier 2 patterns emerge, they can be anchored to the chain — provable evidence that the system produced something no individual session or user created.

Estimated lines: ~300

-----

### 11.9 Scanner — Automatic Shape Extraction From Existing Systems

**The problem:** doorway-memory starts empty. The user installs it, and before they get any value, they have to manually define shapes one by one. That’s a cold start problem that kills adoption. Most users will never get past the setup phase.

**The mechanism:** A scanner reads the structure of existing systems — databases, DataFrames, JSON, APIs, codebases — and automatically mints shapes from what it finds. Each numeric field becomes a dimension. Each observed range becomes a boundary. Day one, the library is populated with the geometric territory of the user’s own infrastructure. Growth, decay, and merge refine the shapes from actual use.

**New file: scanner.py**

```python
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from .shape import Shape, Dimension

class ScanResult:
    """Result of a scan operation."""
    
    def __init__(self, source: str, source_type: str, shapes: List[Shape]):
        self.source = source
        self.source_type = source_type
        self.shapes = shapes
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "shape_count": len(self.shapes),
            "shapes": [s.to_dict() for s in self.shapes]
        }


class Scanner:
    """
    Automatic shape extraction from existing systems.
    
    Reads structure from databases, DataFrames, JSON, APIs, 
    and codebases. Mints shapes from observed dimensional 
    structure. Eliminates cold start.
    
    Usage:
        scanner = Scanner()
        shapes = scanner.scan_dataframe(df)
        shapes = scanner.scan_database(connection_string)
        shapes = scanner.scan_json(data)
        shapes = scanner.scan_openapi(spec)
        shapes = scanner.scan_codebase("./src")
    """
    
    def __init__(self, min_unique_values: int = 3, 
                 margin: float = 0.05):
        """
        Parameters:
            min_unique_values: minimum unique values in a column 
                              to treat it as a dimension (filters 
                              boolean/flag columns)
            margin: boundary margin added to observed min/max 
                    (5% default — prevents exact-boundary points 
                    from landing in void)
        """
        self.min_unique_values = min_unique_values
        self.margin = margin
    
    # ─── DataFrame Scanner ───────────────────────────────────
    
    def scan_dataframe(self, df, name: Optional[str] = None) -> ScanResult:
        """
        Scan a pandas DataFrame. Each numeric column becomes a dimension.
        Min/max of column values become boundaries.
        
        Parameters:
            df: pandas DataFrame
            name: optional name for the shape (defaults to "dataframe")
        
        Returns:
            ScanResult with one shape representing the DataFrame's 
            numeric structure.
        """
        import numpy as np
        
        shape_name = name or "dataframe"
        dimensions = {}
        
        for col in df.columns:
            # Only numeric columns
            if not np.issubdtype(df[col].dtype, np.number):
                continue
            
            # Filter low-cardinality columns (likely flags/booleans)
            unique_count = df[col].nunique()
            if unique_count < self.min_unique_values:
                continue
            
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            
            # Skip zero-range columns
            if col_min == col_max:
                continue
            
            # Add margin
            col_range = col_max - col_min
            margin_abs = col_range * self.margin
            
            dimensions[col] = Dimension(
                name=col,
                min_value=col_min - margin_abs,
                max_value=col_max + margin_abs
            )
        
        if not dimensions:
            return ScanResult(shape_name, "dataframe", [])
        
        shape = Shape(
            dimensions=dimensions,
            metadata={
                "source": shape_name,
                "source_type": "dataframe",
                "columns": list(dimensions.keys()),
                "row_count": len(df),
                "scanned": True
            }
        )
        
        return ScanResult(shape_name, "dataframe", [shape])
    
    # ─── Database Scanner ────────────────────────────────────
    
    def scan_database(self, connection_string: str,
                      tables: Optional[List[str]] = None) -> ScanResult:
        """
        Scan a SQL database. Each table becomes a shape.
        Numeric columns become dimensions with observed min/max.
        
        Parameters:
            connection_string: SQLAlchemy-compatible connection string
            tables: optional list of table names to scan (default: all)
        
        Returns:
            ScanResult with one shape per table.
        """
        try:
            from sqlalchemy import create_engine, inspect, text
        except ImportError:
            raise ImportError(
                "Database scanning requires sqlalchemy. "
                "Install with: pip install doorway-memory[database]"
            )
        
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        
        if tables is None:
            tables = inspector.get_table_names()
        
        shapes = []
        
        for table_name in tables:
            columns = inspector.get_columns(table_name)
            
            # Identify numeric columns
            numeric_cols = []
            for col in columns:
                col_type = str(col["type"]).upper()
                if any(t in col_type for t in [
                    "INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC",
                    "REAL", "BIGINT", "SMALLINT"
                ]):
                    numeric_cols.append(col["name"])
            
            if not numeric_cols:
                continue
            
            # Query min/max for each numeric column
            dimensions = {}
            
            with engine.connect() as conn:
                for col_name in numeric_cols:
                    result = conn.execute(text(
                        f'SELECT MIN("{col_name}"), MAX("{col_name}"), '
                        f'COUNT(DISTINCT "{col_name}") FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL'
                    ))
                    row = result.fetchone()
                    
                    if row is None or row[0] is None or row[1] is None:
                        continue
                    
                    col_min = float(row[0])
                    col_max = float(row[1])
                    unique_count = int(row[2])
                    
                    if col_min == col_max or unique_count < self.min_unique_values:
                        continue
                    
                    col_range = col_max - col_min
                    margin_abs = col_range * self.margin
                    
                    dimensions[col_name] = Dimension(
                        name=col_name,
                        min_value=col_min - margin_abs,
                        max_value=col_max + margin_abs
                    )
            
            if dimensions:
                shape = Shape(
                    dimensions=dimensions,
                    metadata={
                        "source": table_name,
                        "source_type": "database_table",
                        "columns": list(dimensions.keys()),
                        "scanned": True
                    }
                )
                shapes.append(shape)
        
        return ScanResult(connection_string, "database", shapes)
    
    # ─── JSON Scanner ────────────────────────────────────────
    
    def scan_json(self, data: Union[Dict, List, str, Path],
                  name: Optional[str] = None) -> ScanResult:
        """
        Scan JSON data. Walks the structure, finds numeric fields,
        extracts ranges. Works with a single object, a list of objects,
        a file path, or a JSON string.
        
        Parameters:
            data: JSON data as dict, list of dicts, file path, or string
            name: optional name for the source
        
        Returns:
            ScanResult with one shape representing the JSON structure.
        """
        shape_name = name or "json"
        
        # Normalize input
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
            else:
                data = json.loads(str(data))
        
        # Collect all numeric values by field path
        field_values: Dict[str, List[float]] = {}
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._extract_numeric_fields(item, "", field_values)
        elif isinstance(data, dict):
            self._extract_numeric_fields(data, "", field_values)
        
        # Convert to dimensions
        dimensions = {}
        
        for field_path, values in field_values.items():
            if len(set(values)) < self.min_unique_values:
                continue
            
            val_min = min(values)
            val_max = max(values)
            
            if val_min == val_max:
                continue
            
            val_range = val_max - val_min
            margin_abs = val_range * self.margin
            
            dimensions[field_path] = Dimension(
                name=field_path,
                min_value=val_min - margin_abs,
                max_value=val_max + margin_abs
            )
        
        if not dimensions:
            return ScanResult(shape_name, "json", [])
        
        shape = Shape(
            dimensions=dimensions,
            metadata={
                "source": shape_name,
                "source_type": "json",
                "fields": list(dimensions.keys()),
                "record_count": len(data) if isinstance(data, list) else 1,
                "scanned": True
            }
        )
        
        return ScanResult(shape_name, "json", [shape])
    
    def _extract_numeric_fields(self, obj: Dict, prefix: str,
                                 accumulator: Dict[str, List[float]]):
        """Recursively extract numeric values from nested dict."""
        for key, value in obj.items():
            field_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if field_path not in accumulator:
                    accumulator[field_path] = []
                accumulator[field_path].append(float(value))
            elif isinstance(value, dict):
                self._extract_numeric_fields(value, field_path, accumulator)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        if field_path not in accumulator:
                            accumulator[field_path] = []
                        accumulator[field_path].append(float(item))
                    elif isinstance(item, dict):
                        self._extract_numeric_fields(item, field_path, accumulator)
    
    # ─── OpenAPI Scanner ─────────────────────────────────────
    
    def scan_openapi(self, spec: Union[Dict, str, Path],
                     name: Optional[str] = None) -> ScanResult:
        """
        Scan an OpenAPI/Swagger specification. Each endpoint with 
        numeric parameters becomes a shape.
        
        Parameters:
            spec: OpenAPI spec as dict, file path, or JSON/YAML string
            name: optional name for the source
        
        Returns:
            ScanResult with one shape per endpoint that has numeric params.
        """
        shape_name = name or "api"
        
        # Normalize input
        if isinstance(spec, (str, Path)):
            path = Path(spec)
            if path.exists():
                with open(path) as f:
                    content = f.read()
                try:
                    spec = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        spec = yaml.safe_load(content)
                    except ImportError:
                        raise ImportError(
                            "YAML OpenAPI specs require PyYAML. "
                            "Install with: pip install pyyaml"
                        )
            else:
                spec = json.loads(str(spec))
        
        shapes = []
        paths = spec.get("paths", {})
        
        for path_str, methods in paths.items():
            for method, details in methods.items():
                if not isinstance(details, dict):
                    continue
                
                endpoint_name = f"{method.upper()} {path_str}"
                dimensions = {}
                
                # Extract from parameters
                params = details.get("parameters", [])
                for param in params:
                    if not isinstance(param, dict):
                        continue
                    
                    param_name = param.get("name", "")
                    schema = param.get("schema", param)
                    
                    dim = self._schema_to_dimension(param_name, schema)
                    if dim:
                        dimensions[param_name] = dim
                
                # Extract from requestBody
                request_body = details.get("requestBody", {})
                content = request_body.get("content", {})
                
                for content_type, content_details in content.items():
                    schema = content_details.get("schema", {})
                    properties = schema.get("properties", {})
                    
                    for prop_name, prop_schema in properties.items():
                        dim = self._schema_to_dimension(prop_name, prop_schema)
                        if dim:
                            dimensions[prop_name] = dim
                
                if dimensions:
                    shape = Shape(
                        dimensions=dimensions,
                        metadata={
                            "source": endpoint_name,
                            "source_type": "openapi_endpoint",
                            "method": method.upper(),
                            "path": path_str,
                            "parameters": list(dimensions.keys()),
                            "scanned": True
                        }
                    )
                    shapes.append(shape)
        
        return ScanResult(shape_name, "openapi", shapes)
    
    def _schema_to_dimension(self, name: str, schema: Dict) -> Optional[Dimension]:
        """Convert an OpenAPI schema property to a Dimension if numeric."""
        schema_type = schema.get("type", "")
        
        if schema_type not in ("integer", "number"):
            return None
        
        # Use explicit min/max if provided
        min_val = schema.get("minimum", schema.get("exclusiveMinimum"))
        max_val = schema.get("maximum", schema.get("exclusiveMaximum"))
        
        # Default ranges by type if not specified
        if min_val is None or max_val is None:
            if schema_type == "integer":
                min_val = min_val if min_val is not None else 0
                max_val = max_val if max_val is not None else 2147483647
            else:
                min_val = min_val if min_val is not None else 0.0
                max_val = max_val if max_val is not None else 1000000.0
        
        min_val = float(min_val)
        max_val = float(max_val)
        
        if min_val >= max_val:
            return None
        
        return Dimension(name=name, min_value=min_val, max_value=max_val)
    
    # ─── Codebase Scanner ────────────────────────────────────
    
    def scan_codebase(self, path: Union[str, Path],
                      name: Optional[str] = None) -> ScanResult:
        """
        Scan a Python codebase. Each function with typed numeric 
        parameters becomes a shape.
        
        Reads function signatures, type hints, and default values.
        Uses Python's ast module for parsing — no execution required.
        
        Parameters:
            path: path to directory or single .py file
            name: optional name for the source
        
        Returns:
            ScanResult with one shape per function that has numeric params.
        """
        source_path = Path(path)
        shape_name = name or str(source_path)
        shapes = []
        
        # Collect all .py files
        if source_path.is_dir():
            py_files = list(source_path.rglob("*.py"))
        elif source_path.is_file() and source_path.suffix == ".py":
            py_files = [source_path]
        else:
            return ScanResult(shape_name, "codebase", [])
        
        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                
                dimensions = {}
                
                for arg in node.args.args:
                    dim = self._annotation_to_dimension(arg)
                    if dim:
                        dimensions[arg.arg] = dim
                
                # Also check defaults for range hints
                defaults = node.args.defaults
                args_with_defaults = node.args.args[-len(defaults):] if defaults else []
                
                for arg, default in zip(args_with_defaults, defaults):
                    if arg.arg in dimensions and isinstance(default, ast.Constant):
                        if isinstance(default.value, (int, float)):
                            dim = dimensions[arg.arg]
                            default_val = float(default.value)
                            # Use default as hint — expand range if default is outside
                            if default_val < dim.min_value:
                                dimensions[arg.arg] = Dimension(
                                    arg.arg, default_val, dim.max_value
                                )
                            elif default_val > dim.max_value:
                                dimensions[arg.arg] = Dimension(
                                    arg.arg, dim.min_value, default_val
                                )
                
                if dimensions:
                    func_name = node.name
                    rel_path = str(py_file.relative_to(source_path)) if source_path.is_dir() else py_file.name
                    
                    shape = Shape(
                        dimensions=dimensions,
                        metadata={
                            "source": f"{rel_path}:{func_name}",
                            "source_type": "python_function",
                            "file": rel_path,
                            "function": func_name,
                            "line": node.lineno,
                            "parameters": list(dimensions.keys()),
                            "scanned": True
                        }
                    )
                    shapes.append(shape)
        
        return ScanResult(shape_name, "codebase", shapes)
    
    def _annotation_to_dimension(self, arg: ast.arg) -> Optional[Dimension]:
        """
        Convert a function argument's type annotation to a Dimension.
        
        Handles:
            int → Dimension(0, 2147483647)
            float → Dimension(0.0, 1000000.0)
            No annotation → None
        
        These are default ranges. Growth and decay will refine 
        the boundaries from actual use.
        """
        if arg.annotation is None:
            return None
        
        # Handle simple name annotations: int, float
        if isinstance(arg.annotation, ast.Name):
            type_name = arg.annotation.id
            
            if type_name == "int":
                return Dimension(
                    name=arg.arg,
                    min_value=0.0,
                    max_value=2147483647.0
                )
            elif type_name == "float":
                return Dimension(
                    name=arg.arg,
                    min_value=0.0,
                    max_value=1000000.0
                )
        
        # Handle ast.Constant for literal type hints (rare but possible)
        if isinstance(arg.annotation, ast.Constant):
            if isinstance(arg.annotation.value, type) and arg.annotation.value in (int, float):
                return Dimension(
                    name=arg.arg,
                    min_value=0.0,
                    max_value=1000000.0
                )
        
        return None


# ─── Convenience Functions ───────────────────────────────────

def scan(source, **kwargs) -> ScanResult:
    """
    Auto-detect source type and scan.
    
    Usage:
        from doorway_memory import scan
        
        result = scan(df)                    # DataFrame
        result = scan("postgres://...")       # Database
        result = scan({"key": 1.0})          # JSON dict
        result = scan([{"a": 1}, {"a": 2}])  # JSON list
        result = scan("./src")               # Codebase
        result = scan("openapi.json")        # OpenAPI (if contains "paths")
    """
    scanner = Scanner(**kwargs)
    
    # DataFrame
    try:
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            return scanner.scan_dataframe(source)
    except ImportError:
        pass
    
    # Dict — could be JSON or OpenAPI
    if isinstance(source, dict):
        if "paths" in source:
            return scanner.scan_openapi(source)
        return scanner.scan_json(source)
    
    # List of dicts — JSON
    if isinstance(source, list):
        return scanner.scan_json(source)
    
    # String or Path
    if isinstance(source, (str, Path)):
        path = Path(source) if not isinstance(source, Path) else source
        
        # Directory → codebase
        if path.is_dir():
            return scanner.scan_codebase(path)
        
        # .py file → codebase
        if path.suffix == ".py":
            return scanner.scan_codebase(path)
        
        # .json or .yaml → try OpenAPI first, then JSON
        if path.exists() and path.suffix in (".json", ".yaml", ".yml"):
            with open(path) as f:
                content = f.read()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    raise ImportError("YAML files require PyYAML.")
            
            if isinstance(data, dict) and "paths" in data:
                return scanner.scan_openapi(data)
            return scanner.scan_json(data)
        
        # Connection string → database
        if "://" in str(source):
            return scanner.scan_database(str(source))
    
    raise ValueError(
        f"Cannot auto-detect source type for {type(source)}. "
        "Use scanner.scan_dataframe(), scan_database(), scan_json(), "
        "scan_openapi(), or scan_codebase() directly."
    )
```

**Integration with Memory class:**

```python
# In memory.py:

def scan_and_store(self, source, **kwargs) -> int:
    """
    Scan an existing system and store all extracted shapes.
    
    Parameters:
        source: anything scan() accepts — DataFrame, connection 
                string, JSON, OpenAPI spec, codebase path
    
    Returns:
        Number of shapes stored.
    
    Usage:
        mem = Memory()
        mem.scan_and_store(df)                     # DataFrame
        mem.scan_and_store("postgres://...")         # Database
        mem.scan_and_store({"users": [...]})        # JSON
        mem.scan_and_store("./openapi.yaml")        # API spec
        mem.scan_and_store("./src")                 # Codebase
    """
    from .scanner import scan
    result = scan(source, **kwargs)
    
    count = 0
    for shape in result.shapes:
        self.store(shape)
        count += 1
    
    return count
```

**What this means:** The cold start problem is gone. A developer installs doorway-memory, points it at their database, and their library is populated in seconds. Every table becomes a shape. Every numeric column becomes a dimension. Every observed range becomes a boundary. The scanner didn’t need to understand the user’s domain — it read the structure that was already there.

From that moment, every advanced mechanic activates against real data. Growth expands boundaries when real queries push past observed ranges. Overlap finds cross-table patterns nobody defined. Void mapping shows gaps between what the schema allows and what the data contains. Decay identifies tables that stop being queried. Narrative tracks which tables are accessed together and in what order.

The `scan()` convenience function auto-detects source type. One function call, any source, shapes minted. That’s the zero-to-useful experience.

Estimated lines: ~450

-----

### Summary of All Mechanics

|System                |File                |What It Does                                   |Lines     |
|----------------------|--------------------|-----------------------------------------------|----------|
|**Base Layer**        |                    |                                               |          |
|Shape                 |shape.py            |Shapes with boundaries, containment, confidence|~150      |
|Intersect             |intersect.py        |Geometric queries, void detection              |~60       |
|Library               |library.py          |Storage, retrieval, backends                   |~130      |
|Memory API            |memory.py           |store, recall, verify, replay, scan_and_store  |~180      |
|Anchor                |anchor.py           |xycore integration                             |~60       |
|**Advanced Mechanics**|                    |                                               |          |
|Growth                |growth.py           |Shapes expand from near-miss queries           |~120      |
|Overlap               |overlap.py          |Pairwise cross-domain intersection             |~100      |
|Confidence            |shape.py (additions)|Depth-of-knowledge gradient                    |~80       |
|Decay                 |decay.py            |Unused knowledge fades, library self-prunes    |~150      |
|Narrative             |narrative.py        |Trajectories, prediction, common paths         |~160      |
|**Structural Layer**  |                    |                                               |          |
|Merge                 |merge.py            |Shape fusion when boundaries meet              |~130      |
|Void Map              |void_map.py         |Negative space characterization                |~220      |
|Emergence             |emergence.py        |Tier 2 detection, GCS, IS                      |~300      |
|**Onboarding**        |                    |                                               |          |
|Scanner               |scanner.py          |Auto-extract shapes from existing systems      |~450      |
|**Total**             |                    |                                               |**~2,290**|

**Final file structure:**

```
doorway-memory/
├── pyproject.toml
├── BLUEPRINT.md
├── CLAUDE.md
├── README.md
├── LICENSE
├── src/
│   └── doorway_memory/
│       ├── __init__.py
│       ├── shape.py          # Shape + Dimension + confidence
│       ├── intersect.py      # Geometric queries
│       ├── library.py        # Storage + retrieval + backends
│       ├── memory.py         # Public API
│       ├── anchor.py         # xycore integration
│       ├── growth.py         # Shape expansion from near-misses
│       ├── overlap.py        # Pairwise cross-domain intersection
│       ├── merge.py          # Shape fusion
│       ├── decay.py          # Knowledge fading + archive
│       ├── narrative.py      # Trajectories + prediction
│       ├── void_map.py       # Negative space characterization
│       ├── emergence.py      # Tier 2 detection, GCS, IS
│       └── scanner.py        # Auto shape extraction from existing systems
└── tests/
    ├── __init__.py
    ├── test_shape.py
    ├── test_intersect.py
    ├── test_library.py
    ├── test_memory.py
    ├── test_growth.py
    ├── test_overlap.py
    ├── test_merge.py
    ├── test_decay.py
    ├── test_narrative.py
    ├── test_void_map.py
    ├── test_emergence.py
    └── test_scanner.py
```

-----

## Build Order (Updated)

### Phase 1 — Base (build first, test, verify)

1. shape.py (~150 lines)
1. intersect.py (~60 lines)
1. library.py (~130 lines)
1. anchor.py (~60 lines)
1. memory.py (~140 lines)

### Phase 2 — Advanced Mechanics

1. growth.py (~120 lines)
1. overlap.py (~100 lines)
1. decay.py (~150 lines)
1. narrative.py (~160 lines)
1. confidence additions to shape.py (~80 lines)

### Phase 3 — Structural Layer

1. merge.py (~130 lines)
1. void_map.py (~220 lines)
1. emergence.py (~300 lines)

### Phase 4 — Integration

1. Update memory.py to integrate all mechanics
1. Full test suite
1. README + PyPI publish

-----

## Notes

**Feature extraction alignment:** `extract_point()` must match doorway’s feature extraction. If doorway’s dimensions change, this must update. But doorway-memory works without doorway — users define their own dimensions.

**Chain ID convention:** Memory uses `"memory:{namespace}"` to keep memory chains separate from other xycore uses.

**Void as explicit state:** `is_void()` returning True is meaningful information, not an error. It means the state is in genuinely unknown territory.

**Growth + Decay balance:** Growth expands shapes toward queries. Decay shrinks shapes away from disuse. Together they create an equilibrium where the library’s shape reflects actual usage patterns — actively queried regions grow, abandoned regions shrink.

**Merge as consolidation:** When growth causes two shapes to meet, merge fuses them. The library consolidates instead of accumulating redundant overlapping shapes.

**Overlap as emergence:** Nobody programs overlaps. They emerge from the geometry when two independently stored shapes happen to share dimensional space. This is the mechanism-level explanation of cross-domain insight.

**Narrative as collective intelligence:** When multiple trajectories share common subsequences, those paths represent collective learning patterns. The system discovers how knowledge is typically acquired — not from a curriculum, but from observed trajectories through geometric space.

**Void as the gap detector:** The void map is the gap detector expressed as persistent geometry. It characterizes not just where knowledge exists, but the shape, size, and neighborhood of what’s unknown.

**Tier 2 as self-recognition:** GCS and IS are not programmed. They’re detected. When the library’s growth pattern appears inside the library as a shape, the system has recognized its own generative process. When the intersection of intelligence-related shapes produces a coherent region, the mechanism has recognized the mechanism. These detections are anchored to the chain — provable emergence events.

**Namespace and events:** Planned for next update. Namespace adds scoping (private/shared shapes, access control). Events adds hooks (subscribe to growth, merge, decay, emergence events).

-----

*doorway-memory · Geometric Memory Engine · © 2026 Doorway · doorwayagi.com*
