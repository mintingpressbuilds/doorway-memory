"""Core geometric primitives for doorway-memory."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import hashlib
import json

import numpy as np


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
    confidence: float = 1.0
    hit_count: int = 0
    parent_id: Optional[str] = None

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
        shared = [name for name in self.dimensions if name in point]
        if not shared:
            return 0.0

        mins = np.array([self.dimensions[n].min_value for n in shared])
        maxs = np.array([self.dimensions[n].max_value for n in shared])
        vals = np.array([point[n] for n in shared])

        if not self.contains(point):
            # Outside: find max violation
            below = np.maximum(mins - vals, 0.0)
            above = np.maximum(vals - maxs, 0.0)
            max_violation = float(np.max(np.maximum(below, above)))
            return -max_violation

        # Inside: find min distance to any edge
        dist_to_min = vals - mins
        dist_to_max = maxs - vals
        return float(np.min(np.minimum(dist_to_min, dist_to_max)))

    def volume(self) -> float:
        """Calculate the hypervolume of this shape."""
        if not self.dimensions:
            return 0.0
        ranges = np.array([d.max_value - d.min_value for d in self.dimensions.values()])
        return float(np.prod(ranges))

    def record_hit(self) -> None:
        """Record a containment hit (mutable counter only)."""
        self.hit_count += 1

    def to_dict(self) -> Dict:
        """Serialize for storage/anchoring."""
        return {
            "id": self.id,
            "dimensions": {
                name: {"min": d.min_value, "max": d.max_value}
                for name, d in self.dimensions.items()
            },
            "metadata": self.metadata,
            "anchor_id": self.anchor_id,
            "confidence": self.confidence,
            "hit_count": self.hit_count,
            "parent_id": self.parent_id,
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
            anchor_id=data.get("anchor_id"),
            confidence=data.get("confidence", 1.0),
            hit_count=data.get("hit_count", 0),
            parent_id=data.get("parent_id"),
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
