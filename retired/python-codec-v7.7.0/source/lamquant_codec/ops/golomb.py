"""Golomb-Rice entropy coding for integer sequences.

Pure mechanism — bit-level encode/decode for zigzag-mapped signed integers.
Optimal for geometric/Laplacian distributed data (lifting DWT details, LPC
residuals). No knowledge of what the values represent.

Primitives provided:
    BitWriter / BitReader          — bit-level stream I/O (Python)
    zigzag_encode / zigzag_decode  — signed ↔ unsigned mapping
    compute_adaptive_k             — pick k parameter from data
    encode_dense / decode_dense    — Rust (lamquant_core) or numba fallback
    encode_detail / decode_detail  — run-length + GR for sparse arrays

The dense GR hot path uses Rust via lamquant_core when available (1.2-2.1×
faster than numba, zero JIT warmup). Falls back to numba if the Rust
extension is not installed.
"""
import struct
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


class BitWriter:
    """Bitstream writer using a byte-aligned bit buffer.

    Avoids the per-bit Python list overhead of a naive implementation.
    `bitbuf` accumulates pending bits (MSB-first). When `bitpos >= 8`,
    full bytes are flushed to `buf`.
    """
    def __init__(self):
        self.buf = bytearray()
        self.bitbuf = 0    # accumulated bits, big-endian within buffer
        self.bitpos = 0    # number of bits currently in bitbuf
        self._n_bits = 0   # total bits written (for bit_count)

    def write_bits(self, value, n):
        """Append the low `n` bits of `value`, MSB-first."""
        if n <= 0:
            return
        mask = (1 << n) - 1
        self.bitbuf = (self.bitbuf << n) | (value & mask)
        self.bitpos += n
        self._n_bits += n
        # Flush whole bytes as soon as we have any.
        while self.bitpos >= 8:
            self.bitpos -= 8
            self.buf.append((self.bitbuf >> self.bitpos) & 0xFF)
        # Drop the bits we just emitted from the buffer.
        if self.bitpos:
            self.bitbuf &= (1 << self.bitpos) - 1
        else:
            self.bitbuf = 0

    def write_unary(self, value):
        """Write `value` zeros followed by a single 1 bit.

        Equivalent to write_bits(1, value + 1) but split into chunks so
        Python big-int operations stay cheap for large `value` (rare large
        values in Laplacian-distributed data → long unary codes).
        """
        # Emit zeros 56 bits at a time to keep the int small.
        while value >= 56:
            self.write_bits(0, 56)
            value -= 56
        self.write_bits(1, value + 1)

    def write_golomb_rice(self, value, k):
        """Encode unsigned value with Golomb-Rice parameter k."""
        q = value >> k
        self.write_unary(q)
        if k > 0:
            self.write_bits(value & ((1 << k) - 1), k)

    def to_bytes(self):
        """Flush remaining bits (left-aligned) and return bytes."""
        if self.bitpos:
            self.buf.append((self.bitbuf << (8 - self.bitpos)) & 0xFF)
            self.bitbuf = 0
            self.bitpos = 0
        return bytes(self.buf)

    def bit_count(self):
        return self._n_bits


class BitReader:
    """Bitstream reader using a 64-bit refill buffer.

    No per-bit Python list. `bitbuf` holds up to 56+ buffered bits; bits
    are dispensed MSB-first via shift-and-mask. Refill pulls bytes one at
    a time when the buffer drains.
    """
    def __init__(self, data):
        # `data` is bytes / bytearray / memoryview — keep as bytes for
        # constant-time indexing.
        self.data = bytes(data) if not isinstance(data, bytes) else data
        self.bytepos = 0
        self.bitbuf = 0
        self.bitpos = 0     # bits currently held in bitbuf (0..63)
        self.pos = 0        # cumulative bits consumed (for bytes_consumed)

    def _refill(self, n):
        """Ensure at least `n` bits are buffered."""
        while self.bitpos < n and self.bytepos < len(self.data):
            self.bitbuf = (self.bitbuf << 8) | self.data[self.bytepos]
            self.bytepos += 1
            self.bitpos += 8

    def read_bit(self):
        if self.bitpos == 0:
            self._refill(1)
            if self.bitpos == 0:
                return 0   # end of stream
        self.bitpos -= 1
        b = (self.bitbuf >> self.bitpos) & 1
        if self.bitpos:
            self.bitbuf &= (1 << self.bitpos) - 1
        else:
            self.bitbuf = 0
        self.pos += 1
        return b

    def read_bits(self, n):
        if n <= 0:
            return 0
        self._refill(n)
        n_avail = min(n, self.bitpos)
        self.bitpos -= n_avail
        value = self.bitbuf >> self.bitpos
        if self.bitpos:
            self.bitbuf &= (1 << self.bitpos) - 1
        else:
            self.bitbuf = 0
        self.pos += n_avail
        return value

    def read_unary(self):
        """Count leading zero bits, then consume the terminating 1."""
        count = 0
        while True:
            self._refill(56)
            if self.bitpos == 0:
                return count   # exhausted
            if self.bitbuf == 0:
                # All buffered bits are zero — count them all and refill.
                count += self.bitpos
                self.pos += self.bitpos
                self.bitpos = 0
                continue
            # Position of the highest set bit (1-indexed from LSB).
            bl = self.bitbuf.bit_length()
            zeros_before = self.bitpos - bl   # zeros above the highest 1
            count += zeros_before
            # Consume zeros + the terminating 1.
            self.bitpos = bl - 1
            if self.bitpos:
                self.bitbuf &= (1 << self.bitpos) - 1
            else:
                self.bitbuf = 0
            self.pos += zeros_before + 1
            return count

    def read_golomb_rice(self, k):
        q = self.read_unary()
        r = self.read_bits(k) if k > 0 else 0
        return (q << k) | r


def zigzag_encode(v):
    """Map signed int to unsigned: 0→0, -1→1, 1→2, -2→3, ...

    Works for arbitrary-precision Python ints (int32 and int64 safe).
    """
    v = int(v)
    return (v << 1) ^ (v >> 63)


def zigzag_decode(v):
    """Inverse of zigzag_encode."""
    v = int(v)
    return (v >> 1) ^ -(v & 1)


def compute_adaptive_k(values):
    """Compute optimal Golomb-Rice k parameter from data.

    k = max(0, floor(log2(mean(abs(values))))) for Laplacian-distributed data.
    """
    abs_vals = np.abs(values[values != 0]) if np.any(values != 0) else np.array([1])
    mean_abs = float(np.mean(abs_vals)) if len(abs_vals) > 0 else 1.0
    if mean_abs < 1:
        return 0
    return max(0, int(np.floor(np.log2(mean_abs))))


# ============================================================
# Reference (pure Python) implementations — the SPEC for tests.
# Slow; never optimised. The JIT'd versions below must produce
# byte-identical output for every input.
# ============================================================

def _encode_dense_pyref(coeffs):
    """Reference dense GR encoder — pure Python loop. Slow but unambiguous."""
    coeffs = np.asarray(coeffs).flatten()
    n_total = len(coeffs)
    if n_total == 0:
        return bytearray(struct.pack('<BH', 0, 0))

    coeffs_int = np.round(coeffs).astype(np.int64)
    zz_arr = (coeffs_int << 1) ^ (coeffs_int >> 63)

    nz_mask = zz_arr != 0
    mean_abs = float(np.abs(zz_arr[nz_mask]).mean()) if nz_mask.any() else 1.0
    k = max(0, int(np.floor(np.log2(mean_abs)))) if mean_abs >= 1 else 0
    k = min(k, 31)
    k_mask = (1 << k) - 1 if k else 0

    buf = bytearray()
    bitbuf = 0
    bitpos = 0
    for v in zz_arr.tolist():
        q = v >> k
        while q >= 56:
            bitbuf <<= 56
            bitpos += 56
            while bitpos >= 8:
                bitpos -= 8
                buf.append((bitbuf >> bitpos) & 0xFF)
            bitbuf = bitbuf & ((1 << bitpos) - 1) if bitpos else 0
            q -= 56
        n_unary = q + 1
        bitbuf = (bitbuf << n_unary) | 1
        bitpos += n_unary
        while bitpos >= 8:
            bitpos -= 8
            buf.append((bitbuf >> bitpos) & 0xFF)
        bitbuf = bitbuf & ((1 << bitpos) - 1) if bitpos else 0
        if k:
            r = v & k_mask
            bitbuf = (bitbuf << k) | r
            bitpos += k
            while bitpos >= 8:
                bitpos -= 8
                buf.append((bitbuf >> bitpos) & 0xFF)
            bitbuf = bitbuf & ((1 << bitpos) - 1) if bitpos else 0
    if bitpos:
        buf.append((bitbuf << (8 - bitpos)) & 0xFF)
    return bytearray(struct.pack('<BH', k, n_total)) + buf


def _decode_dense_pyref(data, offset=0):
    """Reference dense GR decoder — pure Python loop."""
    if len(data) - offset < 3:
        raise ValueError(
            f"Truncated Golomb-Rice data at offset {offset}: "
            f"need at least 3 bytes for header, got {len(data) - offset}")
    k, n_total = struct.unpack('<BH', data[offset:offset + 3])
    payload_start = offset + 3
    if n_total == 0:
        return np.array([], dtype=np.int64), 3

    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    n_data = len(data)

    values = [0] * n_total
    bytepos = payload_start
    bitbuf = 0
    bitpos = 0
    k_mask = (1 << k) - 1 if k else 0

    for i in range(n_total):
        q = 0
        while True:
            if bitpos == 0:
                while bitpos < 56 and bytepos < n_data:
                    bitbuf = (bitbuf << 8) | data[bytepos]
                    bytepos += 1
                    bitpos += 8
                if bitpos == 0:
                    break
            if bitbuf == 0:
                q += bitpos
                bitpos = 0
                continue
            bl = bitbuf.bit_length()
            q += bitpos - bl
            bitpos = bl - 1
            bitbuf = bitbuf & ((1 << bitpos) - 1) if bitpos else 0
            break
        if k:
            while bitpos < k and bytepos < n_data:
                bitbuf = (bitbuf << 8) | data[bytepos]
                bytepos += 1
                bitpos += 8
            if bitpos >= k:
                bitpos -= k
                r = (bitbuf >> bitpos) & k_mask
                bitbuf = bitbuf & ((1 << bitpos) - 1) if bitpos else 0
            else:
                r = bitbuf << (k - bitpos)
                bitbuf = 0
                bitpos = 0
        else:
            r = 0
        zz = (q << k) | r
        values[i] = (zz >> 1) ^ -(zz & 1)

    bits_consumed = (bytepos - payload_start) * 8 - bitpos
    bytes_consumed = 3 + (bits_consumed + 7) // 8
    return np.array(values, dtype=np.int64), bytes_consumed


# ============================================================
# JIT'd dense Golomb-Rice — numba @njit fallback.
# Used only when lamquant_core (Rust) is not available.
# When _HAS_NUMBA is False, stubs raise RuntimeError (Rust handles everything).
# ============================================================

if not _HAS_NUMBA:
    def _clz_u64(x): raise RuntimeError("requires numba or lamquant_core")
    def _encode_dense_payload_jit(*a): raise RuntimeError("requires numba or lamquant_core")
    def _decode_dense_payload_jit(*a): raise RuntimeError("requires numba or lamquant_core")
    def _encode_dense_full_jit(*a): raise RuntimeError("requires numba or lamquant_core")

if _HAS_NUMBA:

    @numba.njit(numba.int64(numba.uint64), cache=True, inline='always')
    def _clz_u64(x):
        """Count leading zeros of a uint64 (returns 64 for x==0)."""
        if x == 0:
            return numba.int64(64)
        n = numba.int64(0)
        if (x >> numba.uint64(32)) == numba.uint64(0):
            n += 32; x = x << numba.uint64(32)
        if (x >> numba.uint64(48)) == numba.uint64(0):
            n += 16; x = x << numba.uint64(16)
        if (x >> numba.uint64(56)) == numba.uint64(0):
            n += 8; x = x << numba.uint64(8)
        if (x >> numba.uint64(60)) == numba.uint64(0):
            n += 4; x = x << numba.uint64(4)
        if (x >> numba.uint64(62)) == numba.uint64(0):
            n += 2; x = x << numba.uint64(2)
        if (x >> numba.uint64(63)) == numba.uint64(0):
            n += 1
        return n

    @numba.njit(numba.types.UniTuple(numba.int64, 2)(
                    numba.int64[:], numba.int64, numba.uint8[:]),
                cache=True, boundscheck=False)
    def _encode_dense_payload_jit(zz_arr, k, out):
        """Inner JIT encoder. Writes the bitstream into `out` (pre-allocated).
        Returns (k_used, n_bytes_written).
        """
        n_total = zz_arr.shape[0]
        bitbuf = numba.uint64(0)
        bitpos = numba.int64(0)
        out_pos = numba.int64(0)
        k_u = numba.uint64(k)
        k_mask = (numba.uint64(1) << k_u) - numba.uint64(1) if k > 0 else numba.uint64(0)

        for i in range(n_total):
            v = numba.uint64(zz_arr[i])
            q = numba.int64(v >> k_u)
            r = v & k_mask

            while q >= 56:
                bitbuf = bitbuf << numba.uint64(56)
                bitpos += 56
                while bitpos >= 8:
                    bitpos -= 8
                    out[out_pos] = numba.uint8((bitbuf >> numba.uint64(bitpos)) & numba.uint64(0xFF))
                    out_pos += 1
                if bitpos > 0:
                    bitbuf = bitbuf & ((numba.uint64(1) << numba.uint64(bitpos)) - numba.uint64(1))
                else:
                    bitbuf = numba.uint64(0)
                q -= 56

            n_unary = q + 1
            bitbuf = (bitbuf << numba.uint64(n_unary)) | numba.uint64(1)
            bitpos += n_unary
            while bitpos >= 8:
                bitpos -= 8
                out[out_pos] = numba.uint8((bitbuf >> numba.uint64(bitpos)) & numba.uint64(0xFF))
                out_pos += 1
            if bitpos > 0:
                bitbuf = bitbuf & ((numba.uint64(1) << numba.uint64(bitpos)) - numba.uint64(1))
            else:
                bitbuf = numba.uint64(0)

            if k > 0:
                bitbuf = (bitbuf << k_u) | r
                bitpos += k
                while bitpos >= 8:
                    bitpos -= 8
                    out[out_pos] = numba.uint8((bitbuf >> numba.uint64(bitpos)) & numba.uint64(0xFF))
                    out_pos += 1
                if bitpos > 0:
                    bitbuf = bitbuf & ((numba.uint64(1) << numba.uint64(bitpos)) - numba.uint64(1))
                else:
                    bitbuf = numba.uint64(0)

        if bitpos > 0:
            out[out_pos] = numba.uint8((bitbuf << numba.uint64(8 - bitpos)) & numba.uint64(0xFF))
            out_pos += 1
        return k, out_pos

    @numba.njit(numba.types.UniTuple(numba.int64, 2)(
                    numba.uint8[:], numba.int64, numba.int64, numba.int64,
                    numba.int64[:]),
                cache=True, boundscheck=False)
    def _decode_dense_payload_jit(data, payload_start, k, n_total, out):
        n_data = data.shape[0]
        bitbuf = numba.uint64(0)
        bitpos = numba.int64(0)
        bytepos = payload_start
        k_u = numba.uint64(k)
        k_mask = (numba.uint64(1) << k_u) - numba.uint64(1) if k > 0 else numba.uint64(0)

        for i in range(n_total):
            q = numba.int64(0)
            while True:
                if bitpos == 0:
                    while bitpos < 56 and bytepos < n_data:
                        bitbuf = (bitbuf << numba.uint64(8)) | numba.uint64(data[bytepos])
                        bytepos += 1
                        bitpos += 8
                    if bitpos == 0:
                        break
                if bitbuf == numba.uint64(0):
                    q += bitpos
                    bitpos = 0
                    continue
                bl = numba.int64(64) - _clz_u64(bitbuf)
                q += bitpos - bl
                bitpos = bl - 1
                if bitpos > 0:
                    bitbuf = bitbuf & ((numba.uint64(1) << numba.uint64(bitpos)) - numba.uint64(1))
                else:
                    bitbuf = numba.uint64(0)
                break

            if k > 0:
                while bitpos < k and bytepos < n_data:
                    bitbuf = (bitbuf << numba.uint64(8)) | numba.uint64(data[bytepos])
                    bytepos += 1
                    bitpos += 8
                if bitpos >= k:
                    bitpos -= k
                    r = (bitbuf >> numba.uint64(bitpos)) & k_mask
                    if bitpos > 0:
                        bitbuf = bitbuf & ((numba.uint64(1) << numba.uint64(bitpos)) - numba.uint64(1))
                    else:
                        bitbuf = numba.uint64(0)
                else:
                    r = bitbuf << numba.uint64(k - bitpos)
                    bitbuf = numba.uint64(0)
                    bitpos = 0
            else:
                r = numba.uint64(0)

            zz = (numba.uint64(q) << k_u) | r
            zz_i = numba.int64(zz)
            out[i] = (zz_i >> numba.int64(1)) ^ (-(zz_i & numba.int64(1)))

        return bytepos, bitpos

    @numba.njit(numba.types.Tuple((numba.int64, numba.int64))(
                    numba.int64[:], numba.uint8[:]),
                cache=True, boundscheck=False)
    def _encode_dense_full_jit(coeffs_int, out):
        n_total = coeffs_int.shape[0]

        zz_sum = numba.int64(0)
        zz_count = numba.int64(0)
        for i in range(n_total):
            v = coeffs_int[i]
            zz = (v << numba.int64(1)) ^ (v >> numba.int64(63))
            coeffs_int[i] = zz
            if zz != 0:
                zz_sum += zz if zz > 0 else -zz
                zz_count += 1

        if zz_count > 0:
            mean_abs = numba.float64(zz_sum) / numba.float64(zz_count)
        else:
            mean_abs = 1.0
        k = numba.int64(0)
        if mean_abs >= 1.0:
            tmp = mean_abs
            while tmp >= 2.0 and k < 31:
                tmp *= 0.5
                k += 1

        out[0] = numba.uint8(k)
        out[1] = numba.uint8(n_total & 0xFF)
        out[2] = numba.uint8((n_total >> 8) & 0xFF)

        k_used, n_payload = _encode_dense_payload_jit(coeffs_int, k, out[3:])
        return k_used, n_payload + 3


def encode_dense(coeffs):
    """Encode a DENSE subband with straight Golomb-Rice (no run-length).

    Format:
      [1 byte]  k parameter (adaptive)
      [2 bytes] n_total (uint16)
      [N bytes] bitstream: GR(zigzag(value)) for each coefficient

    Uses Rust (lamquant_core) when available, numba fallback otherwise.
    """
    coeffs = np.asarray(coeffs).ravel()
    n_total = len(coeffs)
    if n_total == 0:
        return bytearray(struct.pack('<BH', 0, 0))

    coeffs_int = np.ascontiguousarray(np.round(coeffs).astype(np.int64))

    if _HAS_RUST:
        return bytes(_rs.golomb_encode_dense(coeffs_int))

    buf = np.empty(max(64, n_total * 32 + 3), dtype=np.uint8)
    _, n_bytes = _encode_dense_full_jit(coeffs_int, buf)
    return bytes(buf[:n_bytes])


def decode_dense(data, offset=0):
    """Decode a dense Golomb-Rice encoded subband.

    Returns: (int64 array, bytes_consumed)

    Uses Rust (lamquant_core) when available, numba fallback otherwise.
    """
    if len(data) - offset < 3:
        raise ValueError(
            f"Truncated Golomb-Rice data at offset {offset}: "
            f"need at least 3 bytes for header, got {len(data) - offset}")

    if _HAS_RUST:
        if isinstance(data, np.ndarray):
            data = bytes(data)
        arr, consumed = _rs.golomb_decode_dense(data, offset)
        return np.asarray(arr, dtype=np.int64), consumed

    # Parse header manually (avoid struct.unpack overhead)
    if isinstance(data, (bytes, bytearray, memoryview)):
        k = data[offset]
        n_total = data[offset + 1] | (data[offset + 2] << 8)
    else:
        k = int(data[offset])
        n_total = int(data[offset + 1]) | (int(data[offset + 2]) << 8)

    if n_total == 0:
        return np.array([], dtype=np.int64), 3

    if isinstance(data, np.ndarray) and data.dtype == np.uint8:
        data_arr = data
    elif isinstance(data, (bytes, bytearray, memoryview)):
        data_arr = np.frombuffer(data, dtype=np.uint8).copy()
    else:
        data_arr = np.asarray(data, dtype=np.uint8)

    payload_start = offset + 3
    out = np.empty(n_total, dtype=np.int64)
    bytepos, bitpos = _decode_dense_payload_jit(
        data_arr, np.int64(payload_start), np.int64(k), np.int64(n_total), out)

    bits_consumed = (bytepos - payload_start) * 8 - bitpos
    bytes_consumed = 3 + (bits_consumed + 7) // 8
    return out, bytes_consumed


def encode_detail(coeffs):
    """Encode a sparse subband with run-length + adaptive Golomb-Rice.

    Format:
      [1 byte]  k_run:  GR parameter for run lengths
      [1 byte]  k_val:  GR parameter for coefficient values
      [2 bytes] n_nz:   number of non-zero coefficients (uint16)
      [2 bytes] n_total: total number of coefficients (uint16)
      [N bytes] bitstream: interleaved GR(run) + GR(zigzag(value))

    Returns: bytearray
    """
    coeffs = np.asarray(coeffs).flatten()
    n_total = len(coeffs)

    if n_total == 0:
        return bytearray(struct.pack('<BBHH', 0, 0, 0, 0))

    # Convert to int32 (lifting coefficients are integer-valued)
    coeffs_int = np.round(coeffs).astype(np.int32)

    # Find non-zero positions and values
    nz_mask = coeffs_int != 0
    nz_indices = np.where(nz_mask)[0]
    nz_values = coeffs_int[nz_mask]
    n_nz = len(nz_values)

    if n_nz == 0:
        return bytearray(struct.pack('<BBHH', 0, 0, 0, n_total))

    # Compute run lengths (gaps between non-zero positions)
    runs = np.diff(np.concatenate([[-1], nz_indices])) - 1

    # Zigzag-encode values (vectorized)
    nz_i64 = nz_values.astype(np.int64)
    zz_values = ((nz_i64 << 1) ^ (nz_i64 >> 63)).astype(np.uint32)

    # Adaptive k for runs and values
    k_run = compute_adaptive_k(runs) if len(runs) > 0 else 0
    k_val = compute_adaptive_k(zz_values.astype(np.int32)) if len(zz_values) > 0 else 0
    k_run = min(k_run, 15)  # cap at 4 bits
    k_val = min(k_val, 15)

    # Encode bitstream
    writer = BitWriter()
    for i in range(n_nz):
        writer.write_golomb_rice(int(runs[i]), k_run)
        writer.write_golomb_rice(int(zz_values[i]), k_val)

    bitstream = writer.to_bytes()

    # Pack header + bitstream
    header = struct.pack('<BBHH', k_run, k_val, n_nz, n_total)
    return bytearray(header) + bytearray(bitstream)


def decode_detail(data, offset=0):
    """Decode a run-length + Golomb-Rice bitstream.

    Returns: (float32 coeffs array, bytes_consumed)
    """
    if len(data) - offset < 6:
        return np.array([]), 0

    k_run, k_val, n_nz, n_total = struct.unpack('<BBHH', data[offset:offset + 6])
    offset += 6

    if n_total == 0 or n_nz == 0:
        return np.zeros(n_total, dtype=np.float32), 6

    # Read bitstream
    bitstream_data = data[offset:]
    reader = BitReader(bitstream_data)

    coeffs = np.zeros(n_total, dtype=np.float32)
    pos = -1  # position tracker

    for i in range(n_nz):
        run = reader.read_golomb_rice(k_run)
        zz_val = reader.read_golomb_rice(k_val)
        value = zigzag_decode(zz_val)
        pos += run + 1
        if pos < n_total:
            coeffs[pos] = float(value)

    # Bytes consumed: header + ceil(bits_read / 8)
    bits_consumed = reader.pos
    bytes_consumed = 6 + (bits_consumed + 7) // 8

    return coeffs, bytes_consumed


# Compatibility aliases for the old codec.py private names.
encode = encode_dense  # most common use
decode = decode_dense


__all__ = [
    'BitWriter', 'BitReader',
    'zigzag_encode', 'zigzag_decode', 'compute_adaptive_k',
    'encode_dense', 'decode_dense',
    'encode_detail', 'decode_detail',
    'encode', 'decode',
]
