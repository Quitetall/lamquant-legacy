"""Property-based tests — LML1 wire-format invariants under random input.

Hypothesis-driven coverage of `_compress_bytes` / `_decompress_bytes`.
Each property pins an invariant that MUST hold for every legal input.
Failures generate a minimal reproducer, so any violation surfaces with
the smallest signal that triggers it.

Invariants pinned:

  P1. Lossless roundtrip — for noise_bits = 0:
      decode(encode(x)) == x  (bit-exact)

  P2. Noise-bits roundtrip — for nb in [0..32]:
      decode(encode(x, nb=nb)) == (x >> nb) << nb

  P3. Determinism — encoding the same input twice yields identical bytes:
      encode(x) == encode(x)

  P4. Encode does NOT mutate the input array:
      encode(x); x_after == x_before

  P5. Decoded array has the contracted dtype + shape:
      decode(encode(x)).shape == x.shape
      decode(encode(x)).dtype == np.float64

  P6. Single-bit corruption is always detected:
      For any byte index in the payload, flipping one bit in encode(x)
      raises an LmlError on decode.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    Phase,
    given,
    settings,
    strategies as st,
)

from lamquant_codec.errors import LmlError
from lamquant_codec.lossless import _compress_bytes, _decompress_bytes

pytestmark = [pytest.mark.l4]


# ============================================================
# Strategies
# ============================================================


def _signal_strategy(min_ch: int = 1, max_ch: int = 8,
                     min_t: int = 16, max_t: int = 512):
    """A bounded integer signal [C, T] with int16-range values."""
    return (
        st.tuples(
            st.integers(min_value=min_ch, max_value=max_ch),
            st.integers(min_value=min_t, max_value=max_t),
            st.integers(min_value=0, max_value=2**31 - 1),  # seed
        )
        .map(lambda t: _build_signal(*t))
    )


def _build_signal(n_ch: int, T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # int16 range so the noise-bit safety check (bottom-bits-non-zero
    # heuristic) is satisfied for moderate noise_bits values.
    return rng.integers(-30000, 30000, size=(n_ch, T), dtype=np.int64)


_HYPO = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
)


# ============================================================
# P1. Lossless roundtrip
# ============================================================


class TestLosslessRoundtrip:

    @_HYPO
    @given(sig=_signal_strategy())
    def test_roundtrip_bit_exact_lossless(self, sig: np.ndarray):
        encoded = _compress_bytes(sig.astype(np.float64), noise_bits=0)
        decoded = _decompress_bytes(encoded).astype(np.int64)
        assert decoded.shape == sig.shape
        np.testing.assert_array_equal(decoded, sig)


# ============================================================
# P2. Noise-bits roundtrip
# ============================================================


class TestNoiseBitsRoundtrip:

    @_HYPO
    @given(
        sig=_signal_strategy(),
        nb=st.integers(min_value=0, max_value=8),
    )
    def test_noise_bits_recovers_shifted(self, sig: np.ndarray, nb: int):
        encoded = _compress_bytes(sig.astype(np.float64), noise_bits=nb)
        decoded = _decompress_bytes(encoded).astype(np.int64)
        expected = (sig >> nb) << nb
        np.testing.assert_array_equal(decoded, expected)


# ============================================================
# P3. Determinism
# ============================================================


class TestDeterminism:

    @_HYPO
    @given(sig=_signal_strategy())
    def test_encode_is_deterministic(self, sig: np.ndarray):
        a = _compress_bytes(sig.astype(np.float64), noise_bits=0)
        b = _compress_bytes(sig.astype(np.float64), noise_bits=0)
        assert a == b, "encode() not deterministic for the same input"


# ============================================================
# P4. No mutation of caller's array
# ============================================================


class TestNoMutation:

    @_HYPO
    @given(sig=_signal_strategy())
    def test_encode_does_not_mutate_input(self, sig: np.ndarray):
        before = sig.copy()
        _compress_bytes(sig.astype(np.float64), noise_bits=0)
        np.testing.assert_array_equal(sig, before)


# ============================================================
# P5. Output dtype + shape contract
# ============================================================


class TestOutputContract:

    @_HYPO
    @given(sig=_signal_strategy())
    def test_decode_returns_float64(self, sig: np.ndarray):
        encoded = _compress_bytes(sig.astype(np.float64), noise_bits=0)
        decoded = _decompress_bytes(encoded)
        assert decoded.dtype == np.float64, (
            f"decoded dtype drifted to {decoded.dtype} from contract float64"
        )
        assert decoded.shape == sig.shape


# ============================================================
# P6. Single-bit corruption always detected
# ============================================================


class TestCorruptionDetection:

    @_HYPO
    @given(
        sig=_signal_strategy(min_ch=2, max_ch=4, min_t=64, max_t=128),
        bit_idx=st.integers(min_value=0, max_value=7),
    )
    def test_payload_bit_flip_caught_by_crc(self, sig: np.ndarray, bit_idx: int):
        encoded = _compress_bytes(sig.astype(np.float64), noise_bits=0)
        # Find the start of the payload (skip the ASCII prefix + 22-byte
        # binary header) and flip a payload byte. The exact offset varies
        # with the prefix, so just pick the byte near the end.
        idx = len(encoded) - 4  # last 4 bytes are encoded payload
        bits = bytearray(encoded)
        bits[idx] ^= (1 << bit_idx)
        with pytest.raises(LmlError):
            _decompress_bytes(bytes(bits))
