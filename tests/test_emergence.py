"""Tests for emergence.py — Tier 2 detection, GCS, IS."""

import numpy as np

from doorway_memory.shape import Dimension, Shape
from doorway_memory.emergence import (
    Tier2Shape,
    geometric_coherence_score,
    interaction_strength,
    cluster_interaction_matrix,
    detect_clusters,
    detect_tier2,
    GCS_THRESHOLD,
    IS_THRESHOLD,
)


def _make_shape(xmin, xmax, name=None):
    return Shape(
        dimensions={"x": Dimension("x", xmin, xmax)},
        metadata={"name": name} if name else None,
    )


# --- GCS ---

def test_gcs_single_shape():
    s = _make_shape(0.0, 10.0)
    gcs = geometric_coherence_score([s])
    # Single shape fills its own bounding box perfectly: 10/10 / 1 = 1.0
    assert abs(gcs - 1.0) < 1e-10


def test_gcs_two_identical():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(0.0, 10.0)
    gcs = geometric_coherence_score([s1, s2])
    # Total vol = 20, bb_vol = 10, raw = 2.0, normalized = 2.0/2 = 1.0
    assert abs(gcs - 1.0) < 1e-10


def test_gcs_two_spread_out():
    s1 = _make_shape(0.0, 1.0)
    s2 = _make_shape(99.0, 100.0)
    gcs = geometric_coherence_score([s1, s2])
    # Total vol = 2, bb_vol = 100, raw = 0.02, normalized = 0.01
    assert gcs < 0.05


def test_gcs_partial_coverage():
    s1 = _make_shape(0.0, 5.0)
    s2 = _make_shape(5.0, 10.0)
    gcs = geometric_coherence_score([s1, s2])
    # Total vol = 10, bb_vol = 10, raw = 1.0, normalized = 0.5
    assert abs(gcs - 0.5) < 1e-10


def test_gcs_empty():
    assert geometric_coherence_score([]) == 0.0


def test_gcs_2d():
    s1 = Shape(dimensions={
        "x": Dimension("x", 0.0, 10.0),
        "y": Dimension("y", 0.0, 10.0),
    })
    s2 = Shape(dimensions={
        "x": Dimension("x", 5.0, 15.0),
        "y": Dimension("y", 5.0, 15.0),
    })
    gcs = geometric_coherence_score([s1, s2])
    # Total vol = 200, bb_vol = 15*15=225, raw = 200/225, normalized = raw/2
    expected = (200.0 / 225.0) / 2.0
    assert abs(gcs - expected) < 1e-10


# --- Interaction Strength ---

def test_is_overlapping():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    is_val = interaction_strength(s1, s2)
    # overlap_ratio = 5/10 = 0.5
    assert abs(is_val - 0.5) < 1e-10


def test_is_full_containment():
    s1 = _make_shape(0.0, 20.0)
    s2 = _make_shape(5.0, 10.0)
    is_val = interaction_strength(s1, s2)
    assert abs(is_val - 1.0) < 1e-10


def test_is_no_overlap_close():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(11.0, 20.0)
    is_val = interaction_strength(s1, s2)
    # Close but not overlapping — should have some proximity
    assert 0.0 < is_val < 0.5


def test_is_no_overlap_far():
    s1 = _make_shape(0.0, 1.0)
    s2 = _make_shape(100.0, 101.0)
    is_val = interaction_strength(s1, s2)
    assert is_val < 0.05


def test_is_no_shared_dims():
    s1 = Shape(dimensions={"x": Dimension("x", 0.0, 10.0)})
    s2 = Shape(dimensions={"y": Dimension("y", 0.0, 10.0)})
    assert interaction_strength(s1, s2) == 0.0


# --- Cluster Interaction Matrix ---

def test_cluster_matrix_shape():
    shapes = [_make_shape(0.0, 10.0), _make_shape(5.0, 15.0), _make_shape(50.0, 60.0)]
    matrix = cluster_interaction_matrix(shapes)
    assert matrix.shape == (3, 3)


def test_cluster_matrix_symmetric():
    shapes = [_make_shape(0.0, 10.0), _make_shape(5.0, 15.0)]
    matrix = cluster_interaction_matrix(shapes)
    assert matrix[0, 1] == matrix[1, 0]


def test_cluster_matrix_diagonal_zero():
    shapes = [_make_shape(0.0, 10.0)]
    matrix = cluster_interaction_matrix(shapes)
    assert matrix[0, 0] == 0.0


# --- Detect Clusters ---

def test_detect_clusters_two_groups():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    s3 = _make_shape(100.0, 110.0)
    s4 = _make_shape(105.0, 115.0)
    clusters = detect_clusters([s1, s2, s3, s4])
    assert len(clusters) == 2
    assert sorted(clusters[0]) in ([0, 1], [2, 3])
    assert sorted(clusters[1]) in ([0, 1], [2, 3])


def test_detect_clusters_all_separate():
    s1 = _make_shape(0.0, 1.0)
    s2 = _make_shape(100.0, 101.0)
    s3 = _make_shape(200.0, 201.0)
    clusters = detect_clusters([s1, s2, s3], is_threshold=0.4)
    assert len(clusters) == 3


def test_detect_clusters_all_connected():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    s3 = _make_shape(10.0, 20.0)
    clusters = detect_clusters([s1, s2, s3])
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2]


def test_detect_clusters_empty():
    assert detect_clusters([]) == []


def test_detect_clusters_single():
    s1 = _make_shape(0.0, 10.0)
    clusters = detect_clusters([s1])
    assert len(clusters) == 1
    assert clusters[0] == [0]


# --- Detect Tier 2 ---

def test_detect_tier2_basic():
    # Two overlapping shapes should form a Tier 2 shape
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    results = detect_tier2([s1, s2])
    assert len(results) == 1
    assert isinstance(results[0], Tier2Shape)
    assert len(results[0].source_ids) == 2


def test_detect_tier2_gcs_and_is():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    results = detect_tier2([s1, s2])
    t2 = results[0]
    assert t2.gcs > 0
    assert t2.interaction_strength > 0


def test_detect_tier2_merged_shape():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    results = detect_tier2([s1, s2])
    merged = results[0].shape
    assert merged.dimensions["x"].min_value == 0.0
    assert merged.dimensions["x"].max_value == 15.0


def test_detect_tier2_no_emergence():
    # Two far-apart shapes shouldn't form Tier 2
    s1 = _make_shape(0.0, 1.0)
    s2 = _make_shape(100.0, 101.0)
    results = detect_tier2([s1, s2], is_threshold=0.4)
    assert len(results) == 0


def test_detect_tier2_below_min_cluster():
    s1 = _make_shape(0.0, 10.0)
    results = detect_tier2([s1], min_cluster_size=2)
    assert len(results) == 0


def test_detect_tier2_multiple_clusters():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    s3 = _make_shape(100.0, 110.0)
    s4 = _make_shape(105.0, 115.0)
    results = detect_tier2([s1, s2, s3, s4])
    assert len(results) == 2


def test_detect_tier2_metadata():
    s1 = _make_shape(0.0, 10.0)
    s2 = _make_shape(5.0, 15.0)
    results = detect_tier2([s1, s2])
    assert results[0].metadata["cluster_size"] == 2


def test_detect_tier2_empty():
    assert detect_tier2([]) == []
