"""Doorway Memory — Geometric Memory Engine."""

from .shape import Dimension, Shape, extract_point
from .intersect import point_in_shape, find_containing_shapes, find_nearest_shapes, find_void
from .library import Library
from .memory import Memory
from .anchor import anchor_shape, verify_anchor, generate_receipt
from .emergence import Tier2Shape
from .scanner import Scanner, scan

__all__ = [
    "Memory", "Shape", "Dimension", "Library",
    "Tier2Shape", "Scanner", "scan",
    "extract_point", "point_in_shape",
    "find_containing_shapes", "find_nearest_shapes", "find_void",
    "anchor_shape", "verify_anchor", "generate_receipt",
]
