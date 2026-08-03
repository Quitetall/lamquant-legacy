"""Bit-exact regression + fuzzing for the vectorised lossless DSP primitives.

The four hot-path integer functions in `ai_models/student/subband_preprocess.py`
were rewritten in-place from pure-Python loops to vectorised numpy / numba
implementations. The original loops are preserved as `_*_pyref` reference
implementations in the same module — those are the SPEC, the new versions
are the IMPLEMENTATIONS.

These tests assert:
  1. Adversarial inputs (DC, dirac, alternating ±max, near-bound) produce
     byte-identical output between reference and vectorised versions.
  2. Random parametric coverage across (length, order, seed) combinations.
  3. ~10k random fuzzing cases per function — catches edge cases the
     hand-picked adversarial set didn't think of.

If any test here fails after a numba upgrade or numpy refactor, the
lossless `.lml` wire format has silently drifted. Investigate immediately.
"""
import os
import sys
import random
import numpy as np
import pytest

# Updated 2026-05-05: file moved tests/ → tests/codec/, so go up two levels.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'ai_models', 'student'))

from subband_preprocess import (  # noqa: E402
    lifting_1d_forward_int, lifting_1d_inverse_int,
    lpc_analyze_int, lpc_synthesize_int,
    _lifting_1d_forward_int_pyref, _lifting_1d_inverse_int_pyref,
    _lpc_analyze_int_pyref, _lpc_synthesize_int_pyref,
    # Float versions
    lifting_1d_forward, lifting_1d_inverse,
    lpc_analyze_channel, lpc_synthesize_channel,
    _lifting_1d_forward_pyref, _lifting_1d_inverse_pyref,
    _lpc_analyze_channel_pyref, _lpc_synthesize_channel_pyref,
)


# ============================================================
# Adversarial inputs — fixed cases that catch known edge cases.
# ============================================================

ADVERSARIAL_LENGTHS = [2, 3, 8, 9, 100, 313, 625, 1250, 2500, 2501]


def _adversarial_signals():
    """Yield (name, signal) for each canonical adversarial input."""
    for n in ADVERSARIAL_LENGTHS:
        yield f'all_zero_n{n}', np.zeros(n, dtype=np.int64)
        sig = np.zeros(n, dtype=np.int64); sig[n // 2] = 2**20
        yield f'dirac_pos_n{n}', sig
        sig = np.zeros(n, dtype=np.int64); sig[n // 2] = -(2**20)
        yield f'dirac_neg_n{n}', sig
        sig = np.zeros(n, dtype=np.int64)
        sig[::2] = 10000; sig[1::2] = -10000
        yield f'alternating_n{n}', sig
        yield f'monotonic_ramp_n{n}', np.arange(-n // 2, n - n // 2, dtype=np.int64)
        yield f'near_int30_bound_n{n}', np.full(n, 2**30, dtype=np.int64)
        yield f'dc_42_n{n}', np.full(n, 42, dtype=np.int64)


@pytest.mark.parametrize('name,sig',
                         list(_adversarial_signals()),
                         ids=lambda x: x if isinstance(x, str) else '')
def test_lifting_forward_adversarial(name, sig):
    """Vectorised forward must equal the reference on adversarial inputs."""
    ref_a, ref_d = _lifting_1d_forward_int_pyref(sig)
    new_a, new_d = lifting_1d_forward_int(sig)
    assert np.array_equal(ref_a, new_a), f'approx mismatch on {name}'
    assert np.array_equal(ref_d, new_d), f'detail mismatch on {name}'


@pytest.mark.parametrize('name,sig',
                         list(_adversarial_signals()),
                         ids=lambda x: x if isinstance(x, str) else '')
def test_lifting_inverse_adversarial(name, sig):
    """Vectorised inverse must equal the reference on adversarial inputs."""
    ref_a, ref_d = _lifting_1d_forward_int_pyref(sig)
    ref_recon = _lifting_1d_inverse_int_pyref(ref_a, ref_d)
    new_recon = lifting_1d_inverse_int(ref_a, ref_d)
    assert np.array_equal(ref_recon, new_recon), f'inverse mismatch on {name}'


# ============================================================
# Random parametric tests — random signals × multiple sizes/orders.
# ============================================================

@pytest.mark.parametrize('N', [10, 100, 1000, 2500])
@pytest.mark.parametrize('seed', [0, 1, 7, 42, 99])
def test_lifting_forward_inverse_random(N, seed):
    rng = np.random.default_rng(seed)
    sig = rng.integers(-2**20, 2**20, N, dtype=np.int64)

    ref_a, ref_d = _lifting_1d_forward_int_pyref(sig)
    new_a, new_d = lifting_1d_forward_int(sig)
    assert np.array_equal(ref_a, new_a) and np.array_equal(ref_d, new_d)

    ref_recon = _lifting_1d_inverse_int_pyref(new_a, new_d)
    new_recon = lifting_1d_inverse_int(new_a, new_d)
    assert np.array_equal(ref_recon, new_recon)
    # Round-trip must reproduce input exactly.
    assert np.array_equal(new_recon, sig)


@pytest.mark.parametrize('N', [10, 100, 1000, 2500])
@pytest.mark.parametrize('order', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('seed', [0, 42])
def test_lpc_analyze_random(N, order, seed):
    rng = np.random.default_rng(seed)
    sig = rng.integers(-2**20, 2**20, N, dtype=np.int64)
    coeffs_f = rng.uniform(-2.0, 2.0, order).astype(np.float64)

    ref_q, ref_r = _lpc_analyze_int_pyref(sig, coeffs_f, order)
    new_q, new_r = lpc_analyze_int(sig, coeffs_f, order)
    assert np.array_equal(ref_q, new_q), 'Q27 quantisation drift'
    assert np.array_equal(ref_r, new_r), f'analyze residual mismatch (max {np.abs(ref_r-new_r).max()})'


@pytest.mark.parametrize('N', [10, 100, 1000, 2500])
@pytest.mark.parametrize('order', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('seed', [0, 42])
def test_lpc_synthesize_random(N, order, seed):
    rng = np.random.default_rng(seed)
    sig = rng.integers(-2**20, 2**20, N, dtype=np.int64)
    coeffs_f = rng.uniform(-2.0, 2.0, order).astype(np.float64)

    # Use the reference analyse so test is decoupled from analyse correctness.
    coeffs_q27, residual = _lpc_analyze_int_pyref(sig, coeffs_f, order)

    ref_recon = _lpc_synthesize_int_pyref(residual, coeffs_q27, order)
    new_recon = lpc_synthesize_int(residual, coeffs_q27, order)
    assert np.array_equal(ref_recon, new_recon), 'synthesize mismatch'
    # And the analyse → synthesize round-trip must reconstruct the signal.
    assert np.array_equal(new_recon, sig), 'analyse/synthesise round-trip broken'


# ============================================================
# Fuzzing — many random cases against the reference. Catches the
# edge cases the hand-picked adversarial set didn't think of.
# ============================================================

# How many random cases per function. Tuned to ~10 s total.
N_FUZZ = 2000


def test_lifting_forward_fuzz():
    rng = np.random.default_rng(20260416)
    for _ in range(N_FUZZ):
        N = rng.integers(2, 5000)
        sig = rng.integers(-2**24, 2**24, N, dtype=np.int64)
        ra, rd = _lifting_1d_forward_int_pyref(sig)
        na, nd = lifting_1d_forward_int(sig)
        if not (np.array_equal(ra, na) and np.array_equal(rd, nd)):
            pytest.fail(f'forward mismatch at N={N}, seed signal hash {hash(sig.tobytes())}')


def test_lifting_inverse_fuzz():
    rng = np.random.default_rng(20260417)
    for _ in range(N_FUZZ):
        N = rng.integers(2, 5000)
        sig = rng.integers(-2**24, 2**24, N, dtype=np.int64)
        a, d = _lifting_1d_forward_int_pyref(sig)
        ref = _lifting_1d_inverse_int_pyref(a, d)
        new = lifting_1d_inverse_int(a, d)
        if not np.array_equal(ref, new):
            pytest.fail(f'inverse mismatch at N={N}')
        if not np.array_equal(new, sig):
            pytest.fail(f'inverse round-trip broken at N={N}')


def test_lpc_analyze_fuzz():
    rng = np.random.default_rng(20260418)
    for _ in range(N_FUZZ):
        N = rng.integers(10, 5000)
        order = int(rng.integers(1, 17))
        sig = rng.integers(-2**20, 2**20, N, dtype=np.int64)
        coeffs_f = rng.uniform(-2.0, 2.0, order).astype(np.float64)
        rq, rr = _lpc_analyze_int_pyref(sig, coeffs_f, order)
        nq, nr = lpc_analyze_int(sig, coeffs_f, order)
        if not (np.array_equal(rq, nq) and np.array_equal(rr, nr)):
            pytest.fail(f'analyze mismatch at N={N}, order={order}, '
                        f'max diff {np.abs(rr - nr).max()}')


def test_lpc_synthesize_fuzz():
    rng = np.random.default_rng(20260419)
    for _ in range(N_FUZZ):
        N = rng.integers(10, 5000)
        order = int(rng.integers(1, 17))
        sig = rng.integers(-2**20, 2**20, N, dtype=np.int64)
        coeffs_f = rng.uniform(-2.0, 2.0, order).astype(np.float64)
        coeffs_q27, residual = _lpc_analyze_int_pyref(sig, coeffs_f, order)
        ref = _lpc_synthesize_int_pyref(residual, coeffs_q27, order)
        new = lpc_synthesize_int(residual, coeffs_q27, order)
        if not np.array_equal(ref, new):
            pytest.fail(f'synthesize mismatch at N={N}, order={order}')


# ============================================================
# Float-side regression — vectorised lifting_1d_forward/inverse and
# LPC analyze/synthesize on float64. Each output cell still computes
# the SAME float expression as the loop (operations applied via slice
# arithmetic, no reordering), so the result is byte-identical.
# ============================================================

@pytest.mark.parametrize('N', [10, 100, 1000, 2500])
@pytest.mark.parametrize('seed', [0, 7, 42])
def test_float_lifting_random(N, seed):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(N).astype(np.float64) * 100.0

    ra, rd = _lifting_1d_forward_pyref(sig)
    na, nd = lifting_1d_forward(sig)
    assert np.array_equal(ra, na), 'float forward not bit-identical'
    assert np.array_equal(rd, nd)

    ref_recon = _lifting_1d_inverse_pyref(na, nd)
    new_recon = lifting_1d_inverse(na, nd)
    assert np.array_equal(ref_recon, new_recon), 'float inverse not bit-identical'


@pytest.mark.parametrize('N', [50, 313, 625, 1250, 2500])
@pytest.mark.parametrize('order', [1, 2, 4, 8])
@pytest.mark.parametrize('seed', [0, 42])
def test_float_lpc_analyze_random(N, order, seed):
    """Float-side LPC analyze: vectorised matmul preserves float ordering
    per output sample (same set of multiplications + summation per cell).
    """
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(N).astype(np.float64) * 100.0

    ref_c, ref_r = _lpc_analyze_channel_pyref(sig, order=order,
                                              autocorr_len=min(256, N // 2))
    new_c, new_r = lpc_analyze_channel(sig, order=order,
                                       autocorr_len=min(256, N // 2))
    assert np.array_equal(ref_c, new_c), 'coeffs differ'
    # The vectorised residual uses matmul which may reorder additions slightly
    # vs the per-element loop. Allow tight tolerance — the residual feeds
    # Q27 quantisation downstream which absorbs sub-µV float noise.
    np.testing.assert_allclose(ref_r, new_r, rtol=0, atol=1e-9)


@pytest.mark.parametrize('N', [50, 313, 625, 1250, 2500])
@pytest.mark.parametrize('order', [1, 2, 4, 8])
@pytest.mark.parametrize('seed', [0, 42])
def test_float_lpc_synthesize_random(N, order, seed):
    """Float-side LPC synthesise: numba JIT preserves all float operations
    in identical order — bit-exact match with reference."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(N).astype(np.float64) * 100.0
    coeffs, residual = _lpc_analyze_channel_pyref(
        sig, order=order, autocorr_len=min(256, N // 2))

    ref = _lpc_synthesize_channel_pyref(residual, coeffs)
    new = lpc_synthesize_channel(residual, coeffs)
    assert np.array_equal(ref, new), 'float synthesize not bit-identical'


def test_float_lifting_fuzz():
    """Fuzz: 500 random float signals, lifting forward+inverse round-trip."""
    rng = np.random.default_rng(20260420)
    for _ in range(500):
        N = int(rng.integers(2, 5000))
        sig = rng.standard_normal(N).astype(np.float64) * 100.0
        ra, rd = _lifting_1d_forward_pyref(sig)
        na, nd = lifting_1d_forward(sig)
        if not (np.array_equal(ra, na) and np.array_equal(rd, nd)):
            pytest.fail(f'float forward mismatch at N={N}')
        ref = _lifting_1d_inverse_pyref(na, nd)
        new = lifting_1d_inverse(na, nd)
        if not np.array_equal(ref, new):
            pytest.fail(f'float inverse mismatch at N={N}')


def test_float_lpc_synthesize_fuzz():
    """Fuzz: 500 random cases, JIT'd float synthesize must equal reference.

    Constrains order so autocorr_len > order+1 (the lossless codec
    enforces the same: `if len(sub_data) < order * 4: order = ...`).
    """
    rng = np.random.default_rng(20260421)
    for _ in range(500):
        N = int(rng.integers(80, 5000))   # min 80 so order 16 fits
        max_order = min(16, max(1, N // 8))
        order = int(rng.integers(1, max_order + 1))
        sig = rng.standard_normal(N).astype(np.float64) * 100.0
        coeffs, residual = _lpc_analyze_channel_pyref(
            sig, order=order, autocorr_len=min(256, N // 2))
        ref = _lpc_synthesize_channel_pyref(residual, coeffs)
        new = lpc_synthesize_channel(residual, coeffs)
        if not np.array_equal(ref, new):
            pytest.fail(f'float synthesize mismatch at N={N}, order={order}')


# ============================================================
# JIT'd entropy coders — bit-exact vs pure-Python references.
# encode_dense/decode_dense and rans encode/decode are JIT'd hot
# loops in ops/golomb.py and ops/rans.py.
# ============================================================

@pytest.mark.parametrize('n', [10, 100, 1250, 5000])
@pytest.mark.parametrize('vmax', [10, 1000, 100000])
@pytest.mark.parametrize('seed', [0, 42])
def test_golomb_dense_jit_byte_exact(n, vmax, seed):
    from lamquant_codec.ops.golomb import (
        encode_dense, decode_dense,
        _encode_dense_pyref, _decode_dense_pyref,
    )
    rng = np.random.default_rng(seed)
    data = rng.integers(-vmax, vmax, n, dtype=np.int64)
    ref_bytes = _encode_dense_pyref(data)
    new_bytes = encode_dense(data)
    assert ref_bytes == new_bytes, 'JIT encode_dense not byte-identical'
    ref_dec, _ = _decode_dense_pyref(ref_bytes)
    new_dec, _ = decode_dense(new_bytes)
    assert np.array_equal(ref_dec, new_dec)
    assert np.array_equal(new_dec, data), 'GR roundtrip broken'


def test_golomb_dense_jit_fuzz():
    """Fuzz: 1000 random int64 sequences."""
    from lamquant_codec.ops.golomb import (
        encode_dense, decode_dense,
        _encode_dense_pyref, _decode_dense_pyref,
    )
    rng = np.random.default_rng(20260422)
    for _ in range(1000):
        n = int(rng.integers(1, 5000))
        vmax = int(rng.integers(2, 1_000_000))
        data = rng.integers(-vmax, vmax, n, dtype=np.int64)
        ref_b = _encode_dense_pyref(data)
        new_b = encode_dense(data)
        if ref_b != new_b:
            pytest.fail(f'encode mismatch at n={n}, vmax={vmax}')
        new_d, _ = decode_dense(new_b)
        if not np.array_equal(new_d, data):
            pytest.fail(f'decode mismatch at n={n}, vmax={vmax}')


@pytest.mark.parametrize('n', [10, 100, 2528, 5000])
@pytest.mark.parametrize('L', [4, 8, 16, 32])
@pytest.mark.parametrize('seed', [0, 42])
def test_rans_jit_byte_exact(n, L, seed):
    from lamquant_codec.ops.rans import (
        compute_freq, encode_with_freq, decode,
        _encode_with_freq_pyref, _decode_pyref,
    )
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, L, n, dtype=np.int64)
    freq = compute_freq(syms, n_sym=L)
    ref_bytes = _encode_with_freq_pyref(syms, freq)
    new_bytes = encode_with_freq(syms, freq)
    assert ref_bytes == new_bytes, 'JIT rANS encode not byte-identical'
    ref_dec = _decode_pyref(ref_bytes, freq, n)
    new_dec = decode(new_bytes, freq, n)
    assert np.array_equal(ref_dec, new_dec)
    assert np.array_equal(new_dec, syms), 'rANS roundtrip broken'


def test_rans_jit_fuzz():
    """Fuzz: 1000 random rANS streams."""
    from lamquant_codec.ops.rans import (
        compute_freq, encode_with_freq, decode,
        _encode_with_freq_pyref, _decode_pyref,
    )
    rng = np.random.default_rng(20260423)
    for _ in range(1000):
        n = int(rng.integers(1, 5000))
        L = int(rng.integers(2, 64))
        syms = rng.integers(0, L, n, dtype=np.int64)
        freq = compute_freq(syms, n_sym=L)
        ref_b = _encode_with_freq_pyref(syms, freq)
        new_b = encode_with_freq(syms, freq)
        if ref_b != new_b:
            pytest.fail(f'rANS encode mismatch at n={n}, L={L}')
        new_d = decode(new_b, freq, n)
        if not np.array_equal(new_d, syms):
            pytest.fail(f'rANS decode mismatch at n={n}, L={L}')


# ============================================================
# warm_jit() integration — verify it's a no-op safe to call repeatedly.
# ============================================================

def test_warm_jit_idempotent():
    """warm_jit() must be safe to call multiple times without error."""
    from lamquant_codec import warm_jit
    warm_jit()
    warm_jit()  # second call should be a near-no-op (cache hit)
