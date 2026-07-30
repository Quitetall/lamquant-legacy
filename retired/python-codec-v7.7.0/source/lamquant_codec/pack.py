"""ADR 0075 — the ONE definition of the LQTP1 pack index hash.

The builder (``dataset/build_pack.py``) stamps a pack with this hash; the trainer
(``student/lma_typed_adapter.py``) verifies it fail-closed. They MUST agree byte-for-byte,
or the check either always refuses the pack or — worse — accepts a mismatched one. This is
the single source of truth for that binding; both sides import it.
"""
import hashlib


def index_hash(index) -> bytes:
    """sha256 over the ORDERED base index. Each entry is
    ``(lma_path, stem, win_idx, ...)``; only the identity triple (in order) participates,
    so the hash changes iff the set/order of windows the pack must contain changes.
    """
    h = hashlib.sha256()
    for lma_path, stem, win_idx, *_ in index:
        h.update(f"{lma_path}|{stem}|{win_idx}\n".encode())
    return h.digest()
