"""Linear Predictive Coding (LPC) — analysis, synthesis, integer Q27 variants.

Public API (ops namespace convention):
  analyze_channel       - Levinson-Durbin + vectorised residual (single channel)
  synthesize_channel    - JIT float synthesis wrapper (single channel)
  analyze               - multi-channel analysis wrapper
  synthesize            - multi-channel synthesis wrapper
  analyze_int           - Q27 quantised integer analysis
  synthesize_int        - Q27 integer synthesis wrapper
  analyze_jit           - fused JIT analysis (autocorr + Levinson + residual + bias)
  synthesize_jit        - fused JIT synthesis (bias restore + IIR feedback)
  Q_LPC                 - Q27 fixed-point precision constant

Pyref (reference, slow):
  _analyze_channel_pyref
  _synthesize_channel_pyref
  _analyze_int_pyref
  _synthesize_int_pyref
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True

    @numba.njit(cache=True, fastmath=False)
    def _floor_div(a, b):
        """Floor division matching Python's // (toward -inf).

        AUDIT (2026-04-28): CRITICAL BUG FIX — same bug as bias.py.
        Numba's // for int64 already does floor division. The old
        correction was a double-correction that was off by 1.
        """
        return a // b

except ImportError:
    HAS_NUMBA = False

    def _floor_div(a, b):
        """Fallback floor division — Python's // is already correct."""
        return a // b

Q_LPC = 27   # Q27 fixed-point precision (matches the reference)


# ============================================================
# Reference float LPC implementations -- the SPEC.
# ============================================================

def _analyze_channel_pyref(signal, order=8, autocorr_len=256):
    """Reference float LPC analysis -- pure Python loop. Slow."""
    T = len(signal)
    seg = signal[:min(autocorr_len, T)]
    full_corr = np.correlate(seg, seg, 'full')
    R = full_corr[len(seg) - 1:len(seg) - 1 + order + 1]

    if abs(R[0]) <= 1e-12:
        return np.zeros(order), signal.copy()

    a = np.zeros(order)
    E = R[0]
    for m in range(order):
        lam = R[m + 1]
        for j in range(m):
            lam += a[j] * R[m - j]
        if abs(E) <= 1e-12:
            return np.zeros(order), signal.copy()
        k = -lam / E
        a_new = a.copy()
        a_new[m] = k
        for j in range(m):
            a_new[j] = a[j] + k * a[m - 1 - j]
        a = a_new
        E = E * (1 - k * k)
        if E <= 0:
            E = 1e-10

    pred_coeffs = -a
    residual = signal.copy()
    for n in range(order, T):
        pred = 0.0
        for k_idx in range(order):
            pred += pred_coeffs[k_idx] * signal[n - 1 - k_idx]
        residual[n] = signal[n] - pred
    return pred_coeffs, residual


def _synthesize_channel_pyref(residual, coeffs):
    """Reference float LPC synthesis -- pure Python loop."""
    order = len(coeffs)
    T = len(residual)
    signal = residual.copy()
    for n in range(order, T):
        pred = 0.0
        for k in range(order):
            pred += coeffs[k] * signal[n - 1 - k]
        signal[n] = residual[n] + pred
    return signal


# ============================================================
# Reference integer LPC implementations -- the SPEC.
# ============================================================

def _analyze_int_pyref(signal_int16, coeffs_float, order=8):
    """Reference integer LPC analysis -- pure Python loop. Slow."""
    Q = 27
    SCALE = 1 << Q
    coeffs_q27 = np.round(coeffs_float * SCALE).astype(np.int32)

    N = len(signal_int16)
    sig = signal_int16.astype(np.int64)
    residual = np.zeros(N, dtype=np.int64)

    for n in range(N):
        pred = np.int64(0)
        for k in range(min(order, n)):
            pred += np.int64(coeffs_q27[k]) * sig[n - 1 - k]
        pred = pred >> Q
        residual[n] = sig[n] - pred

    return coeffs_q27, residual


def _synthesize_int_pyref(residual_int, coeffs_q27, order=8):
    """Reference integer LPC synthesis -- pure Python loop. Slow but exact."""
    Q = 27
    N = len(residual_int)
    signal = np.zeros(N, dtype=np.int64)

    for n in range(N):
        pred = np.int64(0)
        for k in range(min(order, n)):
            pred += np.int64(coeffs_q27[k]) * signal[n - 1 - k]
        pred = pred >> Q
        signal[n] = residual_int[n] + pred

    return signal


# ============================================================
# Vectorised float LPC.
# - Levinson-Durbin recursion is O(order^2), tiny -- left as Python.
# - The residual computation is O(T*order) and the actual hot loop --
#   replaced with a sliding-window matmul (pure numpy, no float reordering
#   relative to the loop because each cell still computes the same sum).
# - Synthesis is IIR; numba JIT.
# ============================================================

def analyze_channel(signal, order=8, autocorr_len=256):
    """Compute LPC coefficients and prediction residual for one channel.

    Vectorised residual computation. Levinson-Durbin stays as a Python
    loop because it's O(order^2) and order is small (typically 1-8).

    Returns:
        coeffs: [order] prediction coefficients (negated LP coefficients)
        residual: [T] prediction residual
    """
    T = len(signal)
    seg = signal[:min(autocorr_len, T)]

    # Biased autocorrelation (standard for Levinson-Durbin).
    full_corr = np.correlate(seg, seg, 'full')
    R = full_corr[len(seg) - 1:len(seg) - 1 + order + 1]

    if abs(R[0]) <= 1e-12:
        return np.zeros(order), signal.copy()

    # Levinson-Durbin recursion (O(order^2), tiny).
    a = np.zeros(order)
    E = R[0]
    for m in range(order):
        lam = R[m + 1]
        for j in range(m):
            lam += a[j] * R[m - j]
        if abs(E) <= 1e-12:
            return np.zeros(order), signal.copy()
        k = -lam / E
        a_new = a.copy()
        a_new[m] = k
        for j in range(m):
            a_new[j] = a[j] + k * a[m - 1 - j]
        a = a_new
        E = E * (1 - k * k)
        if E <= 0:
            E = 1e-10

    pred_coeffs = -a
    residual = signal.copy()

    # Vectorised residual via sum of shifted arrays (avoids sliding_window_view overhead)
    if T > order and order > 0:
        pred = np.zeros(T - order, dtype=np.float64)
        for k in range(order):
            # pred_coeffs[k] * signal[order-1-k : T-1-k]
            pred += pred_coeffs[k] * signal[order - 1 - k:T - 1 - k]
        residual[order:] = signal[order:] - pred

    return pred_coeffs, residual


# JIT float synthesis
if HAS_NUMBA:
    @numba.njit('float64[:](float64[:], float64[:])',
                cache=True, fastmath=False, boundscheck=False)
    def _synthesize_channel_jit(residual, coeffs):
        """Numba JIT'd float LPC synthesis."""
        order = coeffs.shape[0]
        T = residual.shape[0]
        signal = residual.copy()
        for n in range(order, T):
            pred = 0.0
            for k in range(order):
                pred += coeffs[k] * signal[n - 1 - k]
            signal[n] = residual[n] + pred
        return signal
else:
    def _synthesize_channel_jit(residual, coeffs):
        """Fallback float LPC synthesis (no numba)."""
        return _synthesize_channel_pyref(residual, coeffs)


def synthesize_channel(residual, coeffs):
    """Reconstruct signal from LPC residual and coefficients.

    Inverse of `analyze_channel`'s prediction filter:
        x[n] = residual[n] + sum(coeffs[k] * x[n-1-k])

    First `order` samples are copied from residual as-is.
    Wraps the JIT primitive; coerces dtype to match the pinned signature.
    """
    return _synthesize_channel_jit(
        np.ascontiguousarray(residual, dtype=np.float64),
        np.ascontiguousarray(coeffs, dtype=np.float64),
    )


def analyze(signal, order=8, autocorr_len=256):
    """Multi-channel LPC analysis.
    Args:
        signal: [C, T] numpy array
        order: LPC order
    Returns:
        coeffs: [C, order] LPC coefficients
        residual: [C, T] prediction residuals
    """
    C, T = signal.shape
    coeffs = np.zeros((C, order), dtype=signal.dtype)
    residual = np.zeros_like(signal)
    for c in range(C):
        coeffs[c], residual[c] = analyze_channel(signal[c], order, autocorr_len)
    return coeffs, residual


def synthesize(residual, coeffs):
    """Multi-channel LPC synthesis (inverse of analyze).
    Args:
        residual: [C, T] numpy array
        coeffs: [C, order] LPC coefficients
    Returns:
        signal: [C, T] reconstructed signal
    """
    C, T = residual.shape
    signal = np.zeros_like(residual)
    for c in range(C):
        signal[c] = synthesize_channel(residual[c], coeffs[c])
    return signal


# ============================================================
# Vectorised integer LPC (Q27 fixed-point).
# ============================================================

def analyze_int(signal_int16, coeffs_float, order=8):
    """Integer LPC analysis. Returns (coeffs_q27, residual_int).

    Bit-exact vectorisation of `_analyze_int_pyref`. Verified by the
    fuzz tests in tests/test_subband_int_vectorize.py.
    """
    Q = Q_LPC
    SCALE = 1 << Q
    coeffs_q27 = np.round(coeffs_float * SCALE).astype(np.int32)

    sig = np.asarray(signal_int16, dtype=np.int64)
    N = sig.shape[0]
    if N == 0:
        return coeffs_q27, np.zeros(0, dtype=np.int64)

    # Pad `order` zeros at the front so windows that would index <0 are 0.
    # For each output n, the prediction window is sig[n-order .. n-1] which
    # equals sig_padded[n .. n+order-1].
    sig_padded = np.empty(N + order, dtype=np.int64)
    sig_padded[:order] = 0
    sig_padded[order:] = sig

    # sliding_window_view returns shape (N+1, order); we want the first N.
    windows = np.lib.stride_tricks.sliding_window_view(sig_padded, order)[:N]

    # Inside each window, the most-recent past sample is at the END.
    # The reference loop accumulates coeffs[0]*sig[n-1] + coeffs[1]*sig[n-2]
    # + ... + coeffs[order-1]*sig[n-order]. That's windows[n] @ coeffs[::-1].
    coeffs_rev = coeffs_q27[::-1].astype(np.int64)

    preds = (windows * coeffs_rev).sum(axis=1) >> Q
    residual = sig - preds
    return coeffs_q27, residual


# Integer LPC synthesis JIT
if HAS_NUMBA:
    @numba.njit('int64[:](int64[:], int32[:], int64)',
                cache=True, fastmath=False, boundscheck=False)
    def _synthesize_int_jit(residual_int, coeffs_q27, order):
        """Numba JIT'd LPC synthesis -- same integer semantics as _pyref."""
        Q = 27
        N = residual_int.shape[0]
        signal = np.zeros(N, dtype=np.int64)
        for n in range(N):
            k_max = order if order < n else n
            pred = np.int64(0)
            for k in range(k_max):
                pred += np.int64(coeffs_q27[k]) * signal[n - 1 - k]
            signal[n] = residual_int[n] + (pred >> Q)
        return signal
else:
    def _synthesize_int_jit(residual_int, coeffs_q27, order):
        """Fallback integer LPC synthesis (no numba)."""
        return _synthesize_int_pyref(
            residual_int, coeffs_q27, order=int(order),
        )


def synthesize_int(residual_int, coeffs_q27, order=8):
    """Integer LPC synthesis. Bit-exact inverse of analyze_int.

    Wraps the JIT primitive. Coerces dtypes to match the pinned signature
    so the same compiled binary handles every caller (no recompilation).
    """
    return _synthesize_int_jit(
        np.ascontiguousarray(residual_int, dtype=np.int64),
        np.ascontiguousarray(coeffs_q27, dtype=np.int32),
        np.int64(order),
    )


# ============================================================
# Fused JIT implementations (used by fused_lml.py).
# ============================================================

if HAS_NUMBA:
    @numba.njit(cache=True, fastmath=False)
    def analyze_jit(subband, order, ctx_len):
        """Numba-native LPC analysis + bias cancellation.
        Same algorithm as analyze_channel + analyze_int + _bias_cancel_jit.
        Single source of truth for the fused pipeline.
        Returns (coeffs_q27 int32[], corrected_residual int64[]).
        """
        T = len(subband)

        if T <= order or T < 3 or order == 0:
            residual = subband.copy()
            buf = np.zeros(ctx_len, dtype=numba.int64)
            running_sum = numba.int64(0)
            for i in range(T):
                bias = _floor_div(running_sum, ctx_len)
                val = residual[i]
                residual[i] -= bias
                old_val = buf[i % ctx_len]
                buf[i % ctx_len] = val
                running_sum += val - old_val
            return np.zeros(max(order, 0), dtype=numba.int32), residual

        # Autocorrelation
        seg_len = min(256, T // 2)
        if seg_len < 1:
            seg_len = 1
        R = np.zeros(order + 1, dtype=numba.float64)
        for lag in range(order + 1):
            s = 0.0
            for i in range(seg_len - lag):
                s += numba.float64(subband[i]) * numba.float64(subband[i + lag])
            R[lag] = s

        if abs(R[0]) <= 1e-12:
            residual = subband.copy()
            buf = np.zeros(ctx_len, dtype=numba.int64)
            running_sum = numba.int64(0)
            for i in range(T):
                bias = _floor_div(running_sum, ctx_len)
                val = residual[i]
                residual[i] -= bias
                old_val = buf[i % ctx_len]
                buf[i % ctx_len] = val
                running_sum += val - old_val
            return np.zeros(order, dtype=numba.int32), residual

        # Levinson-Durbin
        a = np.zeros(order, dtype=numba.float64)
        E = R[0]
        for m in range(order):
            lam = R[m + 1]
            for j in range(m):
                lam += a[j] * R[m - j]
            if abs(E) <= 1e-12:
                break
            k = -lam / E
            a_new = np.zeros(order, dtype=numba.float64)
            a_new[m] = k
            for j in range(m):
                a_new[j] = a[j] + k * a[m - 1 - j]
            for j in range(order):
                a[j] = a_new[j]
            E *= (1 - k * k)
            if E <= 0:
                E = 1e-10

        # Q27 coefficients
        Q27 = numba.int64(1) << numba.int64(27)
        coeffs_q27 = np.empty(order, dtype=numba.int32)
        for i in range(order):
            coeffs_q27[i] = numba.int32(round((-a[i]) * Q27))

        # Integer LPC forward with zero-padded history
        residual = subband.copy()
        for n in range(T):
            pred = numba.int64(0)
            for k_idx in range(order):
                idx = n - 1 - k_idx
                if idx >= 0:
                    pred += numba.int64(coeffs_q27[k_idx]) * subband[idx]
            residual[n] = subband[n] - (pred >> numba.int64(27))

        # Bias cancellation -- running accumulator, O(T) not O(T * ctx_len)
        buf = np.zeros(ctx_len, dtype=numba.int64)
        running_sum = numba.int64(0)
        for i in range(T):
            bias = _floor_div(running_sum, ctx_len)
            val = residual[i]
            residual[i] -= bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = val
            running_sum += val - old_val

        return coeffs_q27, residual

    @numba.njit(cache=True, fastmath=False)
    def synthesize_jit(residual, coeffs_q27, order, ctx_len):
        """Numba-native LPC synthesis + bias restoration.
        Exact inverse of analyze_jit.
        Used by the fused decompress orchestrator.
        """
        T = len(residual)

        # Bias restoration (inverse of bias cancellation)
        # Running accumulator: O(T) instead of O(T * ctx_len)
        restored = residual.copy()
        buf = np.zeros(ctx_len, dtype=numba.int64)
        running_sum = numba.int64(0)
        for i in range(T):
            bias = _floor_div(running_sum, ctx_len)
            restored[i] += bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = restored[i]
            running_sum += restored[i] - old_val

        if T <= order or T < 3 or order == 0:
            return restored

        # LPC synthesis (IIR feedback)
        signal = np.zeros(T, dtype=numba.int64)
        for n in range(T):
            pred = numba.int64(0)
            for k in range(order):
                if k < n:
                    pred += numba.int64(coeffs_q27[k]) * signal[n - 1 - k]
            signal[n] = restored[n] + (pred >> numba.int64(27))

        return signal

else:
    def analyze_jit(subband, order, ctx_len):
        """Fallback LPC analysis + bias cancellation (no numba)."""
        T = len(subband)
        subband = np.asarray(subband, dtype=np.int64)

        if T <= order or T < 3 or order == 0:
            residual = subband.copy()
            buf = np.zeros(ctx_len, dtype=np.int64)
            running_sum = np.int64(0)
            for i in range(T):
                bias = _floor_div(running_sum, ctx_len)
                val = residual[i]
                residual[i] -= bias
                old_val = buf[i % ctx_len]
                buf[i % ctx_len] = val
                running_sum += val - old_val
            return np.zeros(max(order, 0), dtype=np.int32), residual

        # Autocorrelation
        seg_len = min(256, T // 2)
        if seg_len < 1:
            seg_len = 1
        R = np.zeros(order + 1, dtype=np.float64)
        for lag in range(order + 1):
            s = 0.0
            for i in range(seg_len - lag):
                s += float(subband[i]) * float(subband[i + lag])
            R[lag] = s

        if abs(R[0]) <= 1e-12:
            residual = subband.copy()
            buf = np.zeros(ctx_len, dtype=np.int64)
            running_sum = np.int64(0)
            for i in range(T):
                bias = _floor_div(running_sum, ctx_len)
                val = residual[i]
                residual[i] -= bias
                old_val = buf[i % ctx_len]
                buf[i % ctx_len] = val
                running_sum += val - old_val
            return np.zeros(order, dtype=np.int32), residual

        # Levinson-Durbin
        a = np.zeros(order, dtype=np.float64)
        E = R[0]
        for m in range(order):
            lam = R[m + 1]
            for j in range(m):
                lam += a[j] * R[m - j]
            if abs(E) <= 1e-12:
                break
            k = -lam / E
            a_new = np.zeros(order, dtype=np.float64)
            a_new[m] = k
            for j in range(m):
                a_new[j] = a[j] + k * a[m - 1 - j]
            for j in range(order):
                a[j] = a_new[j]
            E *= (1 - k * k)
            if E <= 0:
                E = 1e-10

        # Q27 coefficients
        Q27 = np.int64(1) << np.int64(27)
        coeffs_q27 = np.empty(order, dtype=np.int32)
        for i in range(order):
            coeffs_q27[i] = np.int32(round((-a[i]) * Q27))

        # Integer LPC forward with zero-padded history
        residual = subband.copy()
        for n in range(T):
            pred = np.int64(0)
            for k_idx in range(order):
                idx = n - 1 - k_idx
                if idx >= 0:
                    pred += np.int64(coeffs_q27[k_idx]) * subband[idx]
            residual[n] = subband[n] - (pred >> np.int64(27))

        # Bias cancellation
        buf = np.zeros(ctx_len, dtype=np.int64)
        running_sum = np.int64(0)
        for i in range(T):
            bias = _floor_div(running_sum, ctx_len)
            val = residual[i]
            residual[i] -= bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = val
            running_sum += val - old_val

        return coeffs_q27, residual

    def synthesize_jit(residual, coeffs_q27, order, ctx_len):
        """Fallback LPC synthesis + bias restoration (no numba)."""
        T = len(residual)
        residual = np.asarray(residual, dtype=np.int64)

        # Bias restoration
        restored = residual.copy()
        buf = np.zeros(ctx_len, dtype=np.int64)
        running_sum = np.int64(0)
        for i in range(T):
            bias = _floor_div(running_sum, ctx_len)
            restored[i] += bias
            old_val = buf[i % ctx_len]
            buf[i % ctx_len] = restored[i]
            running_sum += restored[i] - old_val

        if T <= order or T < 3 or order == 0:
            return restored

        # LPC synthesis (IIR feedback)
        signal = np.zeros(T, dtype=np.int64)
        for n in range(T):
            pred = np.int64(0)
            for k in range(order):
                if k < n:
                    pred += np.int64(coeffs_q27[k]) * signal[n - 1 - k]
            signal[n] = restored[n] + (pred >> np.int64(27))

        return signal


__all__ = [
    'analyze_channel', 'synthesize_channel',
    'analyze', 'synthesize',
    'analyze_int', 'synthesize_int',
    'analyze_jit', 'synthesize_jit',
    'Q_LPC',
]
