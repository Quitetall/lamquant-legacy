"""rANS (range Asymmetric Numeral Systems) entropy coding.

Pure mechanism — encodes/decodes symbol streams given a frequency table.
No opinions about what symbols mean or how the frequency table is built.

Reference: Duda 2014, "Asymmetric numeral systems: entropy coding
combining speed of Huffman coding with compression rate of arithmetic coding"

Based on ryg's reference rANS implementation.
State invariant: RANS_L <= state < RANS_L * 256 after each symbol.
RANS_L = 256 * total_freq to ensure sufficient precision.

Public API
----------
    compute_freq(symbols, n_sym, total_freq) -> freq
    encode_with_freq(symbols, freq, total_freq) -> rans_bytes
    encode(symbols, total_freq) -> (rans_bytes, freq)
    decode(rans_bytes, freq, n_symbols, total_freq) -> symbols

Uses Rust (lamquant_core) when available, numba fallback otherwise.
"""

import numpy as np

try:
    import lamquant_core as _rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def compute_freq(symbols, n_sym=None, total_freq=4096):
    """Build a normalized frequency table for an integer symbol stream.

    Returns int32 freq array of length max(n_sym, max(symbols)+1) whose
    elements are >= 1 and sum to exactly total_freq.

    Args:
        symbols: [N] non-negative integers.
        n_sym:   Output alphabet size. Defaults to max(symbols)+1. Pass
                 explicitly when the alphabet is fixed (e.g., FSQ levels)
                 so unused symbols still get a frequency-1 slot.
        total_freq: Sum-to value (default 4096).
    """
    symbols = np.asarray(symbols, dtype=np.int64)
    if n_sym is None:
        n_sym = int(symbols.max()) + 1 if len(symbols) > 0 else 1
    # M must be >= 2*n_sym so every symbol gets at least 1 slot
    # with room for frequent symbols to get proportional counts.
    M = max(total_freq, n_sym * 2)
    counts = np.bincount(symbols, minlength=n_sym)[:n_sym]
    if counts.sum() == 0:
        freq = np.full(n_sym, M // n_sym, dtype=np.int32)
        freq[0] += M - freq.sum()
        return freq
    # Two-pass frequency assignment:
    # Pass 1: give unseen symbols freq=1, observed symbols proportional share
    n_observed = (counts > 0).sum()
    n_unseen = n_sym - n_observed
    budget_for_observed = M - n_unseen  # reserve 1 per unseen
    freq = np.ones(n_sym, dtype=np.int32)  # floor of 1 for all
    if budget_for_observed > n_observed and counts.sum() > 0:
        # Distribute remaining budget proportional to counts
        observed_mask = counts > 0
        proportional = (counts[observed_mask].astype(np.float64)
                        / counts[observed_mask].sum()
                        * budget_for_observed).astype(np.int32)
        proportional = np.maximum(1, proportional)
        freq[observed_mask] = proportional
    # Adjust to exact sum
    diff = M - freq.sum()
    # Add/subtract from the most frequent observed symbol
    most_freq_idx = int(np.argmax(counts))
    freq[most_freq_idx] += diff
    # Safety: ensure no freq <= 0
    if freq[most_freq_idx] <= 0:
        freq[most_freq_idx] = 1
        # Redistribute by stealing from next-most-frequent
        remaining = M - freq.sum()
        sorted_idx = np.argsort(freq)[::-1]
        for idx in sorted_idx:
            if idx == most_freq_idx:
                continue
            steal = min(freq[idx] - 1, -remaining) if remaining < 0 else 0
            freq[idx] -= steal
            remaining += steal
            if remaining >= 0:
                break
        freq[most_freq_idx] += remaining
    return freq


# ============================================================
# Reference implementations — the SPEC. Slow but unambiguous.
# ============================================================

def _encode_with_freq_pyref(symbols, freq, total_freq=4096):
    """Reference rANS encoder — pure Python."""
    symbols = np.asarray(symbols, dtype=np.int64)
    freq = np.asarray(freq, dtype=np.int32)
    n_sym = len(freq)
    M = total_freq

    start = np.cumsum(freq) - freq

    fs = freq[symbols].tolist()
    ss = start[symbols].tolist()
    n = len(fs)

    RANS_L = 256 * M
    rl_div_m = RANS_L // M
    state = RANS_L
    output = bytearray()

    for i in range(n - 1, -1, -1):
        f = fs[i]
        s = ss[i]
        threshold = (rl_div_m * f) << 8
        while state >= threshold:
            output.append(state & 0xFF)
            state >>= 8
        state = (state // f) * M + (state % f) + s

    for _ in range(4):
        output.append(state & 0xFF)
        state >>= 8

    return bytes(output)


def _decode_pyref(rans_bytes, freq, n_symbols, total_freq=4096):
    """Reference rANS decoder — pure Python."""
    freq = np.asarray(freq, dtype=np.int32)
    M = total_freq
    n_sym = len(freq)
    start = np.cumsum(freq) - freq

    cum2sym = np.zeros(M, dtype=np.int32)
    for s in range(n_sym):
        cum2sym[start[s]:start[s] + freq[s]] = s

    rans_payload = bytes(rans_bytes)
    RANS_L = 256 * M
    byte_idx = len(rans_payload) - 1
    state = 0
    for _ in range(4):
        if byte_idx >= 0:
            state = (state << 8) | rans_payload[byte_idx]
            byte_idx -= 1

    cum2sym_list = cum2sym.tolist()
    freq_list = freq.tolist()
    start_list = start.tolist()
    symbols = [0] * n_symbols

    for i in range(n_symbols):
        slot = state % M
        sym = cum2sym_list[slot]
        f = freq_list[sym]
        s_val = start_list[sym]
        state = f * (state // M) + slot - s_val
        while state < RANS_L and byte_idx >= 0:
            state = (state << 8) | rans_payload[byte_idx]
            byte_idx -= 1
        symbols[i] = sym

    return np.array(symbols, dtype=np.int64)


# ============================================================
# JIT'd rANS — numba @njit. ~20× faster than the Python reference.
# Same byte format, byte-identical output.
# Used only when lamquant_core (Rust) is not available.
# ============================================================

if not _HAS_NUMBA:
    def _encode_rans_jit(*a): raise RuntimeError("requires numba or lamquant_core")
    def _decode_rans_jit(*a): raise RuntimeError("requires numba or lamquant_core")

if _HAS_NUMBA:

    @numba.njit(numba.int64(numba.int64[:], numba.int32[:], numba.int32[:],
                            numba.int64, numba.uint8[:]),
                cache=True, boundscheck=False)
    def _encode_rans_jit(symbols, freq, start, M, out):
        n = symbols.shape[0]
        M_u = numba.uint64(M)
        RANS_L = numba.uint64(256) * M_u
        rl_div_m = RANS_L // M_u
        state = RANS_L
        out_pos = numba.int64(0)

        for i in range(n - 1, -1, -1):
            sym = symbols[i]
            f = numba.uint64(freq[sym])
            s = numba.uint64(start[sym])
            threshold = (rl_div_m * f) << numba.uint64(8)
            while state >= threshold:
                out[out_pos] = numba.uint8(state & numba.uint64(0xFF))
                out_pos += 1
                state = state >> numba.uint64(8)
            state = (state // f) * M_u + (state % f) + s

        for _ in range(4):
            out[out_pos] = numba.uint8(state & numba.uint64(0xFF))
            out_pos += 1
            state = state >> numba.uint64(8)

        return out_pos

    @numba.njit(numba.void(numba.uint8[:], numba.int32[:], numba.int32[:],
                           numba.int32[:], numba.int64, numba.int64,
                           numba.int64[:]),
                cache=True, boundscheck=False)
    def _decode_rans_jit(rans_bytes, freq, start, cum2sym, M, n_symbols, out):
        n_data = rans_bytes.shape[0]
        M_u = numba.uint64(M)
        RANS_L = numba.uint64(256) * M_u
        byte_idx = numba.int64(n_data - 1)

        state = numba.uint64(0)
        for _ in range(4):
            if byte_idx >= 0:
                state = (state << numba.uint64(8)) | numba.uint64(rans_bytes[byte_idx])
                byte_idx -= 1

        for i in range(n_symbols):
            slot_u = state % M_u
            slot = numba.int32(slot_u)
            sym = cum2sym[slot]
            f = numba.uint64(freq[sym])
            s_val = numba.uint64(start[sym])
            state = f * (state // M_u) + slot_u - s_val
            while state < RANS_L and byte_idx >= 0:
                state = (state << numba.uint64(8)) | numba.uint64(rans_bytes[byte_idx])
                byte_idx -= 1
            out[i] = numba.int64(sym)


# ============================================================
# Public wrappers — Rust first, numba fallback.
# ============================================================

def encode_with_freq(symbols, freq, total_freq=None):
    """Encode symbols using a pre-built frequency table."""
    symbols = np.ascontiguousarray(symbols, dtype=np.int64)
    freq = np.ascontiguousarray(freq, dtype=np.int32)
    M = int(freq.sum()) if total_freq is None else total_freq
    actual_sum = int(freq.sum())
    if M != actual_sum:
        raise ValueError(f"total_freq={M} != sum(freq)={actual_sum}")
    start = (np.cumsum(freq) - freq).astype(np.int32)

    if _HAS_RUST:
        return bytes(_rs.rans_encode(symbols, freq, start, M))

    n = symbols.shape[0]
    out = np.empty(max(256, n * 8 + 64), dtype=np.uint8)
    n_bytes = _encode_rans_jit(symbols, freq, start, np.int64(M), out)
    return bytes(out[:n_bytes].tobytes())


def encode(symbols, total_freq=4096):
    """Build a freq table and encode in one call. Returns (bytes, freq, M)."""
    symbols = np.asarray(symbols, dtype=np.int64)
    freq = compute_freq(symbols, total_freq=total_freq)
    M = int(freq.sum())
    rans_bytes = encode_with_freq(symbols, freq, total_freq=M)
    return rans_bytes, freq, M


def decode(rans_bytes, freq, n_symbols, total_freq=None):
    """rANS decode symbols from compressed bytes. Exact inverse of encode_with_freq."""
    freq = np.ascontiguousarray(freq, dtype=np.int32)
    M = int(freq.sum()) if total_freq is None else total_freq
    if M <= 0:
        raise ValueError(f"Invalid total_freq={M}. Frequency table is corrupt.")
    if M > 1_000_000:
        raise ValueError(
            f"total_freq={M} exceeds maximum (1M). Frequency table is corrupt.")
    n_sym = len(freq)
    start = (np.cumsum(freq) - freq).astype(np.int32)

    cum2sym = np.zeros(M, dtype=np.int32)
    for s in range(n_sym):
        cum2sym[int(start[s]):int(start[s]) + int(freq[s])] = s

    if _HAS_RUST:
        return np.asarray(_rs.rans_decode(
            rans_bytes, freq, start, cum2sym, M, n_symbols), dtype=np.int64)

    if isinstance(rans_bytes, (bytes, bytearray, memoryview)):
        data = np.frombuffer(bytes(rans_bytes), dtype=np.uint8).copy()
    else:
        data = np.array(rans_bytes, dtype=np.uint8, copy=True)

    out = np.empty(n_symbols, dtype=np.int64)
    _decode_rans_jit(data, freq, start, cum2sym, np.int64(M), np.int64(n_symbols), out)
    return out


__all__ = ['compute_freq', 'encode_with_freq', 'encode', 'decode']
