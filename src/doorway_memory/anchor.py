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
