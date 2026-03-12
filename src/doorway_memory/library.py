"""Persistent storage and retrieval for geometric shapes."""

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
