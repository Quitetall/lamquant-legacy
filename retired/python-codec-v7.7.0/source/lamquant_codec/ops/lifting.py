"""Le Gall 5/3 lifting wavelet transform — float and integer variants.

Pure mechanism — no opinions about levels, orders, or when to use.
The policy layer decides how many levels to decompose.

Public API (integer, vectorised):
    forward_1d_int, inverse_1d_int
    forward_3level_int, inverse_3level_int
    forward_nlevel_int, inverse_nlevel_int

Public API (float, vectorised):
    forward_1d, inverse_1d
    forward_3level, inverse_3level

JIT-accelerated (requires numba):
    forward_1d_int_jit, inverse_1d_int_jit

Reference (pure-Python loop, for testing only):
    _forward_1d_pyref, _inverse_1d_pyref
    _forward_1d_int_pyref, _inverse_1d_int_pyref

References:
    JPEG 2000 Part 2 (ISO 15444-2) — reversible 5/3 filter
    Le Gall & Tabatabai, "Sub-band coding of digital images using
    short kernel filters and arithmetic coding techniques", ICASSP 1988
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ============================================================
# Reference float lifting — the SPEC.
# ============================================================

def _forward_1d_pyref(signal):
    """Reference float forward lifting — pure Python loop."""
    N = len(signal)
    if N < 2:
        return signal.copy(), np.array([], dtype=signal.dtype)
    x = signal.copy()
    n_detail = N // 2
    n_approx = (N + 1) // 2

    for n in range(n_detail - 1):
        x[2*n + 1] -= (x[2*n] + x[2*n + 2]) / 2.0

    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] -= x[last_odd - 1]
        else:
            x[last_odd] -= (x[last_odd - 1] + x[last_odd + 1]) / 2.0

    x[0] += x[1] / 2.0
    for n in range(1, n_approx):
        left = 2*n - 1
        right = 2*n + 1
        if right < N:
            x[2*n] += (x[left] + x[right] + 0) / 4.0
        else:
            x[2*n] += x[left] / 2.0

    return x[0::2][:n_approx], x[1::2][:n_detail]


def _inverse_1d_pyref(approx, detail):
    """Reference float inverse lifting — pure Python loop."""
    n_approx = len(approx)
    n_detail = len(detail)
    N = n_approx + n_detail
    x = np.zeros(N, dtype=approx.dtype)
    x[0::2][:n_approx] = approx
    x[1::2][:n_detail] = detail

    for n in range(n_approx - 1, 0, -1):
        left = 2*n - 1
        right = 2*n + 1
        if right < N:
            x[2*n] -= (x[left] + x[right] + 0) / 4.0
        else:
            x[2*n] -= x[left] / 2.0

    x[0] -= x[1] / 2.0

    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] += x[last_odd - 1]
        else:
            x[last_odd] += (x[last_odd - 1] + x[last_odd + 1]) / 2.0

    for n in range(n_detail - 2, -1, -1):
        x[2*n + 1] += (x[2*n] + x[2*n + 2]) / 2.0

    return x


# ============================================================
# Reference integer lifting — the SPEC for bit-exact behaviour.
# Tests use these to assert the vectorised versions below produce
# byte-identical output. Never optimise these; if they ever change,
# the lossless wire format changes too.
# ============================================================

def _forward_1d_int_pyref(signal):
    """Reference forward lifting — pure Python loop. Slow but unambiguous."""
    N = len(signal)
    if N < 2:
        return signal.copy(), np.array([], dtype=signal.dtype)

    x = signal.astype(np.int64).copy()
    n_detail = N // 2
    n_approx = (N + 1) // 2

    for n in range(n_detail - 1):
        x[2*n + 1] -= (x[2*n] + x[2*n + 2]) >> 1

    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] -= x[last_odd - 1]
        else:
            x[last_odd] -= (x[last_odd - 1] + x[last_odd + 1]) >> 1

    x[0] += (x[1] + 1) >> 1

    for n in range(1, n_approx):
        left = 2*n - 1
        right = 2*n + 1
        if right < N:
            x[2*n] += (x[left] + x[right] + 2) >> 2
        else:
            x[2*n] += (x[left] + 1) >> 1

    approx = x[0::2][:n_approx].copy()
    detail = x[1::2][:n_detail].copy()
    return approx, detail


def _inverse_1d_int_pyref(approx, detail):
    """Reference inverse lifting — pure Python loop."""
    n_approx = len(approx)
    n_detail = len(detail)
    N = n_approx + n_detail

    x = np.zeros(N, dtype=np.int64)
    x[0::2][:n_approx] = approx
    x[1::2][:n_detail] = detail

    for n in range(n_approx - 1, 0, -1):
        left = 2*n - 1
        right = 2*n + 1
        if right < N:
            x[2*n] -= (x[left] + x[right] + 2) >> 2
        else:
            x[2*n] -= (x[left] + 1) >> 1

    x[0] -= (x[1] + 1) >> 1

    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] += x[last_odd - 1]
        else:
            x[last_odd] += (x[last_odd - 1] + x[last_odd + 1]) >> 1

    for n in range(n_detail - 2, -1, -1):
        x[2*n + 1] += (x[2*n] + x[2*n + 2]) >> 1

    return x


# ============================================================
# Vectorised float lifting — numpy slicing (parallel-per-step).
# Bit-identical to the reference because each output cell still
# computes the same float expression in the same order; we just do
# them all in parallel via slice arithmetic.
# ============================================================

def forward_1d(signal):
    """Single-level Le Gall 5/3 forward lifting (float).

    Vectorised; bit-identical to `_forward_1d_pyref`.
    Returns (approximation, detail).
    """
    N = len(signal)
    if N < 2:
        return signal.copy(), np.array([], dtype=signal.dtype)

    x = signal.copy()
    n_detail = N // 2
    n_approx = (N + 1) // 2

    # ---- Predict step (writes odd indices) ----
    n_predict = n_detail - 1
    if n_predict > 0:
        odd_end = 2 * n_predict + 1
        x[1:odd_end:2] -= (x[0:odd_end - 1:2] + x[2:odd_end + 1:2]) / 2.0

    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] -= x[last_odd - 1]
        else:
            x[last_odd] -= (x[last_odd - 1] + x[last_odd + 1]) / 2.0

    # ---- Update step (writes even indices) ----
    x[0] += x[1] / 2.0

    if N % 2 == 0:
        if n_approx > 1:
            x[2::2] += (x[1:-1:2] + x[3::2] + 0) / 4.0
    else:
        if n_approx > 1:
            general_end = 2 * (n_approx - 1)
            x[2:general_end:2] += (x[1:general_end - 1:2]
                                    + x[3:general_end + 1:2] + 0) / 4.0
            last_even = 2 * (n_approx - 1)
            x[last_even] += x[last_even - 1] / 2.0

    return x[0::2][:n_approx], x[1::2][:n_detail]


def inverse_1d(approx, detail):
    """Single-level Le Gall 5/3 inverse lifting (float).

    Vectorised; bit-identical to `_inverse_1d_pyref`.
    """
    n_approx = len(approx)
    n_detail = len(detail)
    N = n_approx + n_detail

    x = np.zeros(N, dtype=approx.dtype)
    x[0::2][:n_approx] = approx
    x[1::2][:n_detail] = detail

    # ---- Inverse update step ----
    if N % 2 == 0:
        if n_approx > 1:
            x[2::2] -= (x[1:-1:2] + x[3::2] + 0) / 4.0
    else:
        if n_approx > 1:
            general_end = 2 * (n_approx - 1)
            x[2:general_end:2] -= (x[1:general_end - 1:2]
                                    + x[3:general_end + 1:2] + 0) / 4.0
            last_even = 2 * (n_approx - 1)
            x[last_even] -= x[last_even - 1] / 2.0

    x[0] -= x[1] / 2.0

    # ---- Inverse predict step ----
    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] += x[last_odd - 1]
        else:
            x[last_odd] += (x[last_odd - 1] + x[last_odd + 1]) / 2.0

    n_predict = n_detail - 1
    if n_predict > 0:
        odd_end = 2 * n_predict + 1
        x[1:odd_end:2] += (x[0:odd_end - 1:2] + x[2:odd_end + 1:2]) / 2.0

    return x


# ============================================================
# Vectorised integer lifting — numpy slicing.
# Bit-identical to the reference because numpy's >> on int64 arrays
# uses the same C-level arithmetic shift as Python's int >>.
# ============================================================

def forward_1d_int(signal):
    """Integer Le Gall 5/3 forward lifting. Bit-exact roundtrip.

    Predict: d[n] = odd[n] - ((even[n] + even[n+1]) >> 1)
    Update:  s[n] = even[n] + ((d[n-1] + d[n] + 2) >> 2)

    Input must be integer-valued. Output is int64. Vectorised via numpy
    slice arithmetic; bit-identical to `_forward_1d_int_pyref`.
    """
    N = len(signal)
    if N < 2:
        return signal.copy(), np.array([], dtype=signal.dtype)

    x = signal.astype(np.int64).copy()
    n_detail = N // 2
    n_approx = (N + 1) // 2

    # ---- Predict step (writes odd indices) ----
    # For n in [0, n_detail - 1):  x[2n+1] -= (x[2n] + x[2n+2]) >> 1
    n_predict = n_detail - 1
    if n_predict > 0:
        odd_end = 2 * n_predict + 1   # exclusive end of the odd slice
        x[1:odd_end:2] -= (x[0:odd_end - 1:2] + x[2:odd_end + 1:2]) >> 1

    # Last-detail boundary (handled separately; reads modified evens).
    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] -= x[last_odd - 1]
        else:
            x[last_odd] -= (x[last_odd - 1] + x[last_odd + 1]) >> 1

    # ---- Update step (writes even indices, reads detail values) ----
    x[0] += (x[1] + 1) >> 1  # first even has only a right neighbour

    if N % 2 == 0:
        # All n in [1, n_approx) use the general formula.
        # Even targets: x[2], x[4], ..., x[N-2]  →  x[2::2]
        # Lefts:  x[1], x[3], ..., x[N-3]        →  x[1:-1:2]
        # Rights: x[3], x[5], ..., x[N-1]        →  x[3::2]
        if n_approx > 1:
            x[2::2] += (x[1:-1:2] + x[3::2] + 2) >> 2
    else:
        # N odd: last n = n_approx - 1 uses the right-edge formula;
        # the rest use the general formula.
        if n_approx > 1:
            general_end = 2 * (n_approx - 1)   # exclusive
            x[2:general_end:2] += (x[1:general_end - 1:2]
                                    + x[3:general_end + 1:2] + 2) >> 2
            last_even = 2 * (n_approx - 1)
            x[last_even] += (x[last_even - 1] + 1) >> 1

    approx = x[0::2][:n_approx].copy()
    detail = x[1::2][:n_detail].copy()
    return approx, detail


def inverse_1d_int(approx, detail):
    """Integer Le Gall 5/3 inverse lifting. Bit-exact inverse of forward."""
    n_approx = len(approx)
    n_detail = len(detail)
    N = n_approx + n_detail

    x = np.zeros(N, dtype=np.int64)
    x[0::2][:n_approx] = approx
    x[1::2][:n_detail] = detail

    # ---- Inverse update step (writes even indices) ----
    if N % 2 == 0:
        if n_approx > 1:
            x[2::2] -= (x[1:-1:2] + x[3::2] + 2) >> 2
    else:
        if n_approx > 1:
            general_end = 2 * (n_approx - 1)
            x[2:general_end:2] -= (x[1:general_end - 1:2]
                                    + x[3:general_end + 1:2] + 2) >> 2
            last_even = 2 * (n_approx - 1)
            x[last_even] -= (x[last_even - 1] + 1) >> 1

    x[0] -= (x[1] + 1) >> 1

    # ---- Inverse predict step (writes odd indices) ----
    if n_detail > 0:
        last_odd = 2 * (n_detail - 1) + 1
        if N % 2 == 0:
            x[last_odd] += x[last_odd - 1]
        else:
            x[last_odd] += (x[last_odd - 1] + x[last_odd + 1]) >> 1

    n_predict = n_detail - 1
    if n_predict > 0:
        odd_end = 2 * n_predict + 1
        x[1:odd_end:2] += (x[0:odd_end - 1:2] + x[2:odd_end + 1:2]) >> 1

    return x


# ============================================================
# Numba JIT integer lifting
# ============================================================

if HAS_NUMBA:
    @numba.njit(cache=True, fastmath=False, boundscheck=False)
    def forward_1d_int_jit(signal):
        """Numba-native Le Gall 5/3 forward lifting.
        Bit-identical to forward_1d_int (numpy version).
        Used by the fused pipeline orchestrator.
        """
        N = len(signal)
        if N < 2:
            return signal.copy(), np.empty(0, dtype=numba.int64)

        x = signal.astype(numba.int64).copy()
        n_detail = N // 2
        n_approx = (N + 1) // 2

        # Predict
        for n in range(n_detail - 1):
            x[2*n+1] -= (x[2*n] + x[2*n+2]) >> numba.int64(1)
        if n_detail > 0:
            lo = 2*(n_detail-1)+1
            if N % 2 == 0:
                x[lo] -= x[lo-1]
            else:
                x[lo] -= (x[lo-1] + x[lo+1]) >> numba.int64(1)

        # Update
        x[0] += (x[1] + numba.int64(1)) >> numba.int64(1)
        for n in range(1, n_approx):
            left = 2*n-1
            right = 2*n+1
            if right < N:
                x[2*n] += (x[left] + x[right] + numba.int64(2)) >> numba.int64(2)
            else:
                x[2*n] += (x[left] + numba.int64(1)) >> numba.int64(1)

        approx = np.empty(n_approx, dtype=numba.int64)
        detail = np.empty(n_detail, dtype=numba.int64)
        for i in range(n_approx):
            approx[i] = x[2*i]
        for i in range(n_detail):
            detail[i] = x[2*i+1]
        return approx, detail

    @numba.njit(cache=True, fastmath=False, boundscheck=False)
    def inverse_1d_int_jit(approx, detail):
        """Numba-native Le Gall 5/3 inverse lifting.
        Bit-identical to inverse_1d_int (numpy version).
        Used by the fused decompress orchestrator.
        """
        n_approx = len(approx)
        n_detail = len(detail)
        N = n_approx + n_detail

        x = np.zeros(N, dtype=numba.int64)
        for i in range(n_approx):
            x[2*i] = approx[i]
        for i in range(n_detail):
            x[2*i+1] = detail[i]

        # Inverse update
        if N % 2 == 0:
            if n_approx > 1:
                for n in range(1, n_approx):
                    left = 2*n-1
                    right = 2*n+1
                    if right < N:
                        x[2*n] -= (x[left] + x[right] + numba.int64(2)) >> numba.int64(2)
                    else:
                        x[2*n] -= (x[left] + numba.int64(1)) >> numba.int64(1)
        else:
            if n_approx > 1:
                for n in range(1, n_approx - 1):
                    left = 2*n-1
                    right = 2*n+1
                    x[2*n] -= (x[left] + x[right] + numba.int64(2)) >> numba.int64(2)
                last_even = 2 * (n_approx - 1)
                x[last_even] -= (x[last_even - 1] + numba.int64(1)) >> numba.int64(1)

        x[0] -= (x[1] + numba.int64(1)) >> numba.int64(1)

        # Inverse predict
        if n_detail > 0:
            last_odd = 2 * (n_detail - 1) + 1
            if N % 2 == 0:
                x[last_odd] += x[last_odd - 1]
            else:
                x[last_odd] += (x[last_odd - 1] + x[last_odd + 1]) >> numba.int64(1)

        for n in range(n_detail - 1):
            x[2*n+1] += (x[2*n] + x[2*n+2]) >> numba.int64(1)

        return x

else:
    # Fallback: use the pyref versions when numba is not available
    forward_1d_int_jit = _forward_1d_int_pyref
    inverse_1d_int_jit = _inverse_1d_int_pyref


# ============================================================
# 3-level composites (float)
# ============================================================

def forward_3level(signal):
    """3-level lifting DWT on a single channel.

    2500 → L1(1250 approx + 1250 detail)
         → L2(625 approx + 625 detail)
         → L3(313 approx + 312 detail)

    Returns dict with subband arrays.
    """
    # Level 1
    l1_approx, l1_detail = forward_1d(signal)

    # Level 2
    l2_approx, l2_detail = forward_1d(l1_approx)

    # Level 3 (625 samples → 313 approx + 312 detail, 625 is odd)
    l3_approx, l3_detail = forward_1d(l2_approx)

    return {
        'l3_approx': l3_approx,   # [313]
        'l3_detail': l3_detail,   # [312]
        'l2_detail': l2_detail,   # [625]
        'l1_detail': l1_detail,   # [1250]
    }


def inverse_3level(subbands):
    """Inverse 3-level lifting DWT on a single channel.
    Reconstructs the original signal from subbands.
    """
    # Level 3 inverse
    l2_approx = inverse_1d(subbands['l3_approx'], subbands['l3_detail'])

    # Level 2 inverse
    l1_approx = inverse_1d(l2_approx, subbands['l2_detail'])

    # Level 1 inverse
    signal = inverse_1d(l1_approx, subbands['l1_detail'])

    return signal


# ============================================================
# 3-level composites (integer)
# ============================================================

def forward_3level_int(signal):
    """3-level integer lifting on a single channel. Bit-exact."""
    signal_int = np.round(signal).astype(np.int64)
    l1_approx, l1_detail = forward_1d_int(signal_int)
    l2_approx, l2_detail = forward_1d_int(l1_approx)
    l3_approx, l3_detail = forward_1d_int(l2_approx)
    return {
        'l3_approx': l3_approx,
        'l3_detail': l3_detail,
        'l2_detail': l2_detail,
        'l1_detail': l1_detail,
    }


def inverse_3level_int(subbands):
    """3-level integer inverse lifting. Bit-exact inverse of forward_int."""
    l2_approx = inverse_1d_int(subbands['l3_approx'], subbands['l3_detail'])
    l1_approx = inverse_1d_int(l2_approx, subbands['l2_detail'])
    signal = inverse_1d_int(l1_approx, subbands['l1_detail'])
    return signal


# ============================================================
# N-level composites (integer)
# ============================================================

def forward_nlevel_int(signal_int, n_levels):
    """N-level integer lifting. Returns dict of subbands."""
    approx = signal_int
    details = []
    for _ in range(n_levels):
        approx, detail = forward_1d_int(approx)
        details.append(detail)
    result = {f'l{n_levels}_approx': approx}
    for i, d in enumerate(reversed(details)):
        level = n_levels - i
        result[f'l{level}_detail'] = d
    return result


def inverse_nlevel_int(subbands, n_levels):
    """N-level integer inverse lifting."""
    approx = subbands[f'l{n_levels}_approx']
    for level in range(n_levels, 0, -1):
        detail = subbands[f'l{level}_detail']
        approx = inverse_1d_int(approx, detail)
    return approx


__all__ = [
    'forward_1d', 'inverse_1d',
    'forward_1d_int', 'inverse_1d_int',
    'forward_1d_int_jit', 'inverse_1d_int_jit',
    'forward_3level', 'inverse_3level',
    'forward_3level_int', 'inverse_3level_int',
    'forward_nlevel_int', 'inverse_nlevel_int',
]
