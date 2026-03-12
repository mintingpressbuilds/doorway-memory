# CLAUDE.md — doorway-memory

## What This Repo Is

Geometric memory engine. Standalone Python package. Store knowledge as shapes with dimensional boundaries. Retrieve by containment testing. Verify by cryptographic chain.

**This is not part of Doorway’s reasoning engine.** It’s a standalone package that anyone can use for any system. It does not import doorway, pruv, or any Doorway-specific code. xycore is an optional dependency for anchoring.

## Package Structure

```
doorway-memory/
├── pyproject.toml
├── BLUEPRINT.md              # Build specification — the contract
├── CLAUDE.md                 # This file — operating instructions
├── README.md
├── LICENSE                   # Apache 2.0
├── src/
│   └── doorway_memory/
│       ├── __init__.py       # Exports: Memory, Shape, Dimension, Library
│       ├── shape.py          # Shape + Dimension + confidence + extract_point
│       ├── intersect.py      # Geometric queries + void detection
│       ├── library.py        # Storage + retrieval + backends
│       ├── memory.py         # Public API: store, recall, verify, replay
│       ├── anchor.py         # xycore integration (optional)
│       ├── growth.py         # Shape expansion from near-misses
│       ├── overlap.py        # Pairwise cross-domain intersection
│       ├── merge.py          # Shape fusion when boundaries meet
│       ├── decay.py          # Knowledge fading + archive
│       ├── narrative.py      # Trajectories + prediction + common paths
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

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_shape.py -v

# Run with coverage
pytest tests/ --cov=doorway_memory -v

# Build package
python -m build

# Publish to PyPI
twine upload dist/*
```

## Dependencies

**Required:**

- numpy >= 1.20.0

**Optional:**

- xycore >= 0.1.0 (for anchoring/chain/replay)
- supabase >= 1.0.0 (for cloud persistence)

**Dev:**

- pytest
- pytest-cov

## Conventions

### Code Style

- Python 3.9+ compatible
- Type hints on all public functions
- Dataclasses for data objects (Shape, Dimension, VoidRegion, Trajectory, Tier2Shape)
- No external dependencies beyond numpy unless optional
- All optional imports wrapped in try/except with HAS_X flags

### Naming

- Files: lowercase with underscores
- Classes: PascalCase
- Functions: snake_case
- Constants: UPPER_SNAKE_CASE at module level
- All configurable thresholds as module-level constants, not hardcoded

### Testing

- Every file has a corresponding test file
- Tests must pass without xycore or supabase installed
- Tests that require optional deps should be skipped with `pytest.mark.skipif`
- Test names: `test_<what>_<condition>` (e.g., `test_contains_point_inside`)

### Error Handling

- `ImportError` with helpful message when optional dep is missing
- Never silently fail — if xycore isn’t installed and anchoring is requested, raise
- Return `None` or empty list for no-result queries, don’t raise

## What NOT To Do

- Do NOT import from doorway, pruv, or any Doorway package
- Do NOT add authentication or user management (that’s namespace.py, coming later)
- Do NOT use localStorage, databases, or external services in the base layer — file and memory backends only (supabase is optional)
- Do NOT modify pyproject.toml dependencies without checking this file
- Do NOT use print() for logging — use Python’s logging module if needed
- Do NOT hardcode thresholds inside functions — use module-level constants

## Build Order

Follow BLUEPRINT.md phases strictly. Do not skip ahead.

1. Phase 1: shape → intersect → library → anchor → memory
1. Phase 2: growth → overlap → decay → narrative → confidence
1. Phase 3: merge → void_map → emergence
1. Phase 4: scanner
1. Phase 5: integration + full test suite + README + publish

Each phase must have all tests passing before starting the next phase.

## Architecture Notes

### The Two Modes

doorway-memory provides geometric containment. doorway provides semantic similarity. Same dimensional space, different query type. They share feature extraction but diverge at the matching step.

### Optional xycore Integration

Every file that uses xycore must wrap the import:

```python
try:
    import xycore
    HAS_XYCORE = True
except ImportError:
    HAS_XYCORE = False
```

Functions that need xycore must check `HAS_XYCORE` and raise `ImportError` with install instructions if missing.

### Memory Class Is The Public API

Users interact with `Memory`. They should not need to import growth, decay, overlap, merge, void_map, or emergence directly. Memory orchestrates all mechanics internally. The `__init__.py` exports `Memory`, `Shape`, `Dimension`, and `Library` — that’s the public surface.

### Shapes Are Immutable After Creation

When growth expands a shape, it creates a new Shape with a new ID. The old shape is archived. When decay shrinks a shape, same thing. When merge fuses shapes, same thing. Shapes don’t mutate in place. The chain records the lineage.

## Repository

- **GitHub:** github.com/mintingpressbuilds/doorway-memory
- **PyPI:** doorway-memory
- **License:** Apache 2.0
- **Author:** Luke H
- **Part of:** Doorway stack (doorwayagi.com) — but standalone
