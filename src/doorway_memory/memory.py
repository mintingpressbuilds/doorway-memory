"""High-level geometric memory API."""

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

    def scan_and_store(self, source: Any, name: str = "auto") -> int:
        """Scan a data source and store all extracted shapes. Returns count stored."""
        from .scanner import scan
        result = scan(source, name=name)
        for shape in result.shapes:
            self.store(shape)
        return len(result.shapes)

    def count(self) -> int:
        return self.library.count()
