"""LMQ v5 byte-exact roundtrip regression tests.

Catches the rANS bug class that lived undetected in the LMQ v2 path
(`SubbandCodec.compress`) for many months: encoder threshold off by 8
bits, encoder compressed bytes verbatim, decoder recovered ~12% of
symbols. The basic compress→decompress path was never exercised end-to-end.

These tests assert symbol-perfect roundtrip, not just "no exception". They
are L0 (pure math, no model, no checkpoint) so they always run.
"""
import numpy as np
import pytest

from lamquant_codec.compress import _compress_bytes, MAGIC as MAGIC_COMPRESS
from lamquant_codec.decompress import _decompress_bytes, MAGIC as MAGIC_DECOMPRESS


def _expected_symbols(latent: np.ndarray, L: int) -> np.ndarray:
    """Compute what symbols the encoder *should* emit for a given latent."""
    v = latent.flatten().astype(np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    span = vmax - vmin + 1e-8
    return np.clip(((v - vmin) / span * L).astype(np.int32), 0, L - 1)


def _recovered_symbols(decoded_latent: np.ndarray, ref_vmin: float,
                       ref_vmax: float, L: int) -> np.ndarray:
    """Recover symbols from a dequantized latent (inverts the FSQ midpoint)."""
    span = ref_vmax - ref_vmin + 1e-8
    return np.round((decoded_latent.flatten() - ref_vmin) / span * L - 0.5).astype(np.int32)


def test_magic_is_lmq():
    """compress.py and decompress.py agree on the magic bytes."""
    assert MAGIC_COMPRESS == b'LMQ1'
    assert MAGIC_DECOMPRESS == b'LMQ1'


@pytest.mark.parametrize('seed', [0, 1, 7, 42, 99])
@pytest.mark.parametrize('L', [8, 16, 32])
def test_roundtrip_symbol_perfect(seed, L):
    """Every symbol must come back exactly. This is the test the LMQ v2
    bug would have failed (~12% recovery vs the required 100%).
    """
    np.random.seed(seed)
    shape = (1, 32, 79)
    latent = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
    expected = _expected_symbols(latent, L)

    b = _compress_bytes(latent, quality_mode=0, fsq_levels=L)
    assert b[:4] == b'LMQ1', f'unexpected magic: {b[:4]!r}'

    lat_dec, qm, Ld, _, _, _ = _decompress_bytes(b)
    assert qm == 0
    assert Ld == L

    v = latent.flatten()
    vmin, vmax = float(v.min()), float(v.max())
    recovered = _recovered_symbols(lat_dec, vmin, vmax, L)

    n_match = int(np.sum(expected == recovered))
    assert n_match == expected.size, (
        f'Only {n_match}/{expected.size} symbols recovered '
        f'(seed={seed}, L={L}) — rANS roundtrip is broken')


@pytest.mark.parametrize('L', [8, 16, 32])
def test_dequantization_error_within_half_bin(L):
    """The dequantized latent must land within half a bin of the original."""
    np.random.seed(0)
    latent = np.random.uniform(-1.0, 1.0, (1, 32, 79)).astype(np.float32)
    span = float(latent.max() - latent.min()) + 1e-8
    bin_width = span / L

    b = _compress_bytes(latent, quality_mode=0, fsq_levels=L)
    lat_dec, *_ = _decompress_bytes(b)

    err = np.abs(lat_dec - latent).max()
    assert err <= 0.51 * bin_width, (
        f'Max dequant error {err:.4f} exceeds half-bin {0.5*bin_width:.4f}')


def test_pathological_inputs():
    """Adversarial cases that broke the old encoder: all-zero, dirac, alternating."""
    cases = {
        'all_zero': np.zeros((1, 32, 79), dtype=np.float32),
        'dirac':    np.zeros((1, 32, 79), dtype=np.float32),
        'altern':   np.tile([-1.0, 1.0], (1, 32, 40))[:, :, :79].astype(np.float32),
    }
    cases['dirac'][0, 0, 0] = 1.0

    for name, latent in cases.items():
        b = _compress_bytes(latent, quality_mode=0, fsq_levels=8)
        lat_dec, *_ = _decompress_bytes(b)
        assert lat_dec.shape == latent.shape, f'{name}: shape mismatch'
        # Within half-bin
        span = float(latent.max() - latent.min()) + 1e-8
        bin_w = span / 8
        if span > 1e-6:  # for non-degenerate cases
            assert np.abs(lat_dec - latent).max() <= 0.51 * bin_w, \
                f'{name}: dequant exceeds half-bin'


def test_with_lpc_and_details_roundtrip():
    """Quality-mode 2 includes LPC + detail subbands. Verify they survive."""
    np.random.seed(42)
    latent = np.random.uniform(-1, 1, (1, 32, 79)).astype(np.float32)
    lpc = np.random.uniform(-0.5, 0.5, (21, 8)).astype(np.float32)
    details = [
        {'l1_detail': np.random.randint(-100, 100, 1250).astype(np.int32),
         'l2_detail': np.random.randint(-50, 50, 625).astype(np.int32),
         'l3_detail': np.random.randint(-20, 20, 312).astype(np.int32)}
        for _ in range(21)
    ]

    b = _compress_bytes(latent, lpc_coeffs=lpc, subbands_per_ch=details,
                        quality_mode=2, fsq_levels=32)
    lat_dec, qm, L, _, lpc_b, det_b = _decompress_bytes(b)
    assert qm == 2
    assert L == 32
    assert len(lpc_b) > 0
    assert len(det_b) > 0
    # Latent symbols must roundtrip
    expected = _expected_symbols(latent, L=32)
    v = latent.flatten()
    recovered = _recovered_symbols(lat_dec, float(v.min()), float(v.max()), 32)
    assert np.array_equal(expected, recovered)
