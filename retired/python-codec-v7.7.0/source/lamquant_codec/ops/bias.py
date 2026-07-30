"""Context-adaptive bias cancellation / restoration.

Subtracts (encode) or adds back (decode) the running mean of the last
`ctx_len` residual samples.  +6% compression ratio improvement.

JIT-compiled with numba when available; pure-Python fallback otherwise.
Both paths use a running accumulator with floor division -- bit-identical
across environments.
"""
import numpy as np

from lamquant_codec.ops.constants import BIAS_CTX_LEN

try:
    import numba as _numba

    @_numba.njit(cache=True, fastmath=False)
    def _floor_div(a, b):
        """Floor division matching Python's // semantics (toward -inf).

        AUDIT (2026-04-28): CRITICAL BUG FIX. The old comment claimed
        "Numba's // uses C truncation (toward zero) for int64" — this is
        FALSE. Numba's // for int64 already does floor division (toward -inf),
        matching Python's native // operator. The old code applied a
        correction for C-style truncation on top of an already-correct floor
        division, producing a DOUBLE CORRECTION that was off by 1 for all
        negative dividends where the division is not exact.

        Example: _floor_div(-100, 32) returned -5 (wrong), should be -4.
        This caused the bias cancellation to diverge from Rust's floor_div,
        producing ±1 errors on ~95% of samples in cross-language decode.

        Every .lml file encoded with numba available was affected. The Python
        fallback (line 80: running_sum // ctx_len) was correct — the bug was
        numba-only. This means the roundtrip worked on machines without numba
        but failed on machines with numba.
        """
        return a // b

    @_numba.njit('int64[:](int64[:], int64)', cache=True, fastmath=False)
    def _cancel_jit_inner(residual, ctx_len):
        out = residual.copy()
        buf = np.zeros(ctx_len, dtype=np.int64)
        running_sum = np.int64(0)
        for i in range(len(out)):
            bias = _floor_div(running_sum, ctx_len)
            val = residual[i]
            out[i] -= bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = val
            running_sum += val - old_val
        return out

    @_numba.njit('int64[:](int64[:], int64)', cache=True, fastmath=False)
    def _restore_jit_inner(corrected, ctx_len):
        out = corrected.copy()
        buf = np.zeros(ctx_len, dtype=np.int64)
        running_sum = np.int64(0)
        for i in range(len(out)):
            bias = _floor_div(running_sum, ctx_len)
            out[i] += bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = out[i]
            running_sum += out[i] - old_val
        return out

    def cancel_jit(residual, ctx_len):
        """Subtract running mean of last ctx_len residuals. Encode side."""
        ctx_len = int(ctx_len)
        if ctx_len < 1:
            raise ValueError(f"ctx_len must be >= 1, got {ctx_len}")
        return _cancel_jit_inner(residual, np.int64(ctx_len))

    def restore_jit(corrected, ctx_len):
        """Add back running mean. Decode side. Exact inverse of cancel."""
        ctx_len = int(ctx_len)
        if ctx_len < 1:
            raise ValueError(f"ctx_len must be >= 1, got {ctx_len}")
        return _restore_jit_inner(corrected, np.int64(ctx_len))

except ImportError:
    def cancel_jit(residual, ctx_len):
        """Subtract running mean (pure Python fallback)."""
        ctx_len = int(ctx_len)
        if ctx_len < 1:
            raise ValueError(f"ctx_len must be >= 1, got {ctx_len}")
        out = residual.copy()
        buf = np.zeros(ctx_len, dtype=np.int64)
        running_sum = np.int64(0)
        for i in range(len(out)):
            bias = running_sum // ctx_len
            val = residual[i]
            out[i] -= bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = val
            running_sum += val - old_val
        return out

    def restore_jit(corrected, ctx_len):
        """Add back running mean (pure Python fallback)."""
        ctx_len = int(ctx_len)
        if ctx_len < 1:
            raise ValueError(f"ctx_len must be >= 1, got {ctx_len}")
        out = corrected.copy()
        buf = np.zeros(ctx_len, dtype=np.int64)
        running_sum = np.int64(0)
        for i in range(len(out)):
            bias = running_sum // ctx_len
            out[i] += bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = out[i]
            running_sum += out[i] - old_val
        return out

__all__ = ['cancel_jit', 'restore_jit', 'BIAS_CTX_LEN']
