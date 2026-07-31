"""
LamQuant lossless codec module (LML / Mode 3).

This module is now lossless-only. The neural codec wrappers (TernaryCodec
/ SubbandCodec, Modes 0-2) and the model definitions they wrap were moved
to the PRIVATE LamQuant-Neural package on 2026-05-29 (reverting the
ADR-0018 "codec owns the model classes" decision). The PUBLIC Lossless
package no longer imports torch or any neural class.

What lives here now:
  - The pure entropy-coder primitives (Golomb-Rice + rANS), re-exported
    from `lamquant_codec.ops.*` for backward compat with importers that
    did `from lamquant_codec.codec import _rans_encode_symbols` etc.
  - `LosslessCodec` (Mode 3, pure DSP, R=1.000), re-exported from
    `lamquant_codec.lossless`.

The neural wrappers now live at:
    from lamquant_neural.codec import SubbandCodec, TernaryCodec
which requires the LamQuant-Neural wheel.
"""

# Entropy coding primitives — the pure mechanism layer. Re-exported so
# existing `from lamquant_codec.codec import _rans_encode_symbols` (and the
# Golomb dense/detail helpers) keep resolving.
from lamquant_codec.ops.golomb import (  # noqa: F401
    BitWriter, BitReader,
    zigzag_encode as _zigzag_encode,
    zigzag_decode as _zigzag_decode,
    compute_adaptive_k as _compute_adaptive_k,
    encode_dense as _encode_dense_subband,
    decode_dense as _decode_dense_subband,
    encode_detail as _encode_detail_subband,
    decode_detail as _decode_detail_subband,
)
from lamquant_codec.ops.rans import (  # noqa: F401
    encode as _rans_encode_symbols,
    decode as _rans_decode_symbols,
)



# ============================================================
# Mode 3: Lossless DSP Codec
# ============================================================
# Moved to lamquant_codec/lossless.py — the wire format (LMQ v4) and
# all KLT / lifting helpers now live there. We re-export LosslessCodec
# here so existing `from lamquant_codec.codec import LosslessCodec`
# imports keep working.

from lamquant_codec.lossless import (  # noqa: E402, F401
    LosslessCodec,
    compute_klt,
    compute_lifting_rotations,
    apply_lifting_klt_forward,
    apply_lifting_klt_inverse,
    LIFT_PREC,
    LIFT_HALF,
)
