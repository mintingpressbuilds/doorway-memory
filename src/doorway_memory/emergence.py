"""Tier 2 detection, Geometric Coherence Score, and Interaction Strength.

Emergence detects when collections of shapes exhibit higher-order
structure — patterns that exist across shapes rather than within
any single shape. These "Tier 2" shapes represent emergent knowledge:
things the system knows that no individual shape captures.

Key metrics:
- GCS (Geometric Coherence Score): measures how tightly a cluster
  of shapes occupies a region of dimensional space.
- IS (Interaction Strength): measures how much shapes influence
  each other through overlap and proximity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .shape import Shape, Dimension
from .overlap import overlap_ratio, overlap_volume
from .merge import merge_shapes

# Minimum GCS to consider a cluster as emergent
GCS_THRESHOLD = 0.3

# Minimum interaction strength to consider shapes as interacting
IS_THRESHOLD = 0.1

# Minimum cluster size for Tier 2 detection
MIN_CLUSTER_SIZE = 2


@dataclass
class Tier2Shape:
    """An emergent shape detected from a cluster of base shapes."""
    shape: Shape
    source_ids: List[str]
    gcs: float
    interaction_strength: float
    metadata: Optional[Dict] = None


def geometric_coherence_score(shapes: List[Shape]) -> float:
    """
    Compute the Geometric Coherence Score for a set of shapes.

    GCS measures how tightly the shapes cluster relative to their
    combined bounding box. Higher values mean the shapes collectively
    cover more of their joint bounding region.

    GCS = sum(shape volumes) / bounding_box_volume

    Returns a value between 0.0 and len(shapes) (>1.0 means overlap).
    Normalized to [0, 1] by dividing by len(shapes).
    """
    if not shapes:
        return 0.0

    all_dims = set()
    for s in shapes:
        all_dims.update(s.dimensions.keys())

    if not all_dims:
        return 0.0

    bb_dims = {}
    for dim_name in all_dims:
        dim_mins = []
        dim_maxs = []
        for s in shapes:
            if dim_name in s.dimensions:
                dim_mins.append(s.dimensions[dim_name].min_value)
                dim_maxs.append(s.dimensions[dim_name].max_value)
        if dim_mins:
            bb_dims[dim_name] = (min(dim_mins), max(dim_maxs))

    bb_volume = 1.0
    for lo, hi in bb_dims.values():
        span = hi - lo
        if span == 0.0:
            bb_volume = 0.0
            break
        bb_volume *= span

    if bb_volume == 0.0:
        return 0.0

    total_shape_volume = sum(s.volume() for s in shapes)
    raw_gcs = total_shape_volume / bb_volume

    return min(raw_gcs / len(shapes), 1.0)


def interaction_strength(shape_a: Shape, shape_b: Shape) -> float:
    """
    Measure the interaction strength between two shapes.

    IS combines overlap ratio and boundary proximity. Two shapes
    interact strongly if they overlap significantly or if their
    boundaries are close.

    Returns a value between 0.0 (no interaction) and 1.0 (identical).
    """
    ratio = overlap_ratio(shape_a, shape_b)
    if ratio > 0:
        return ratio

    # No overlap — check proximity via distance
    shared = set(shape_a.dimensions.keys()) & set(shape_b.dimensions.keys())
    if not shared:
        return 0.0

    # Find minimum gap between shapes on any shared dimension
    min_gap = float('inf')
    max_span = 0.0
    for name in shared:
        da = shape_a.dimensions[name]
        db = shape_b.dimensions[name]
        gap = max(0.0, max(da.min_value, db.min_value) - min(da.max_value, db.max_value))
        span = max(da.max_value, db.max_value) - min(da.min_value, db.min_value)
        min_gap = min(min_gap, gap)
        max_span = max(max_span, span)

    if max_span == 0.0:
        return 0.0

    # Proximity decays with distance
    proximity = max(0.0, 1.0 - min_gap / max_span)
    return proximity * 0.5  # Scale to [0, 0.5] for proximity-only interaction


def cluster_interaction_matrix(shapes: List[Shape]) -> np.ndarray:
    """
    Build a pairwise interaction strength matrix for a list of shapes.

    Returns an NxN numpy array where entry [i,j] is the IS between
    shapes[i] and shapes[j].
    """
    n = len(shapes)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            strength = interaction_strength(shapes[i], shapes[j])
            matrix[i, j] = strength
            matrix[j, i] = strength
    return matrix


def detect_clusters(
    shapes: List[Shape],
    is_threshold: float = IS_THRESHOLD,
) -> List[List[int]]:
    """
    Find clusters of interacting shapes using connected components.

    Two shapes are in the same cluster if their interaction strength
    exceeds the threshold, either directly or transitively.

    Returns list of clusters, where each cluster is a list of shape indices.
    """
    n = len(shapes)
    if n == 0:
        return []

    matrix = cluster_interaction_matrix(shapes)
    visited = [False] * n
    clusters = []

    for start in range(n):
        if visited[start]:
            continue
        cluster = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            cluster.append(node)
            for neighbor in range(n):
                if not visited[neighbor] and matrix[node, neighbor] >= is_threshold:
                    stack.append(neighbor)
        clusters.append(sorted(cluster))

    return clusters


def detect_tier2(
    shapes: List[Shape],
    gcs_threshold: float = GCS_THRESHOLD,
    is_threshold: float = IS_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> List[Tier2Shape]:
    """
    Detect emergent Tier 2 shapes from a collection of base shapes.

    Process:
    1. Find clusters of interacting shapes
    2. For each cluster meeting minimum size, compute GCS
    3. If GCS exceeds threshold, create a Tier 2 shape from the
       merged cluster

    Returns list of Tier2Shape objects.
    """
    clusters = detect_clusters(shapes, is_threshold)
    tier2_shapes = []

    for cluster_indices in clusters:
        if len(cluster_indices) < min_cluster_size:
            continue

        cluster_shapes = [shapes[i] for i in cluster_indices]
        gcs = geometric_coherence_score(cluster_shapes)

        if gcs < gcs_threshold:
            continue

        # Merge all shapes in the cluster
        merged = cluster_shapes[0]
        for s in cluster_shapes[1:]:
            merged = merge_shapes(merged, s)

        # Compute mean interaction strength within cluster
        total_is = 0.0
        pair_count = 0
        for i in range(len(cluster_shapes)):
            for j in range(i + 1, len(cluster_shapes)):
                total_is += interaction_strength(cluster_shapes[i], cluster_shapes[j])
                pair_count += 1
        mean_is = total_is / pair_count if pair_count > 0 else 0.0

        tier2 = Tier2Shape(
            shape=merged,
            source_ids=[shapes[idx].id for idx in cluster_indices],
            gcs=gcs,
            interaction_strength=mean_is,
            metadata={"cluster_size": len(cluster_indices)},
        )
        tier2_shapes.append(tier2)

    return tier2_shapes
