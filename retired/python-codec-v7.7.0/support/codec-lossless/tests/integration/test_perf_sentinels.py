"""Performance regression sentinels — clinical-grade non-functional contract.

These four sentinels run nightly only (marked @slow @perf). They use
generous bands (≥3× headroom over current benchmarks) so they're alarms
for catastrophic regressions, not micro-benchmarks.

Pinned invariants on a 21-channel × 2500-sample int16 pink-noise buffer:

  * Compression ratio  > 1.8     (current ~2.3 on clinical EEG)
  * Encode latency     < 80 ms   (median of 5 cold-cache runs)
  * Decode latency     < 40 ms
  * Peak RSS delta     < 64 MB

Failures here surface as a regression flag in nightly CI. The plan is
NOT to chase the wall clock — it's to catch a 10× slowdown that would
make the codec unusable in clinical workflows.
"""
from __future__ import annotations

import resource
import time

import numpy as np
import pytest

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
from tests.helpers.signals import make_synthetic_eeg

pytestmark = [pytest.mark.slow, pytest.mark.perf, pytest.mark.l3]


# ============================================================
# Synthetic pink-noise EEG
# ============================================================


@pytest.fixture(scope="module")
def pink_noise_21ch_2500() -> np.ndarray:
    """Band-limited synthetic EEG — 21 channels × 2500 samples (10 s @ 250 Hz).

    `make_synthetic_eeg` generates a multi-tone + noise signal whose
    spectrum approximates real clinical EEG (delta, theta, alpha bands
    plus a small white-noise floor). It compresses to a CR similar to
    real recordings (~2.0-2.3 with the lossless codec) — uniform random
    noise would only give CR ~1.1 and isn't a meaningful regression
    sentinel.
    """
    sig = make_synthetic_eeg(n_channels=21, n_samples=2500, seed=2026)
    # Convert to int16-range integers for the codec (it operates on
    # integer samples). Multiplying by ~50 keeps the dynamic range
    # below int16's max (32k) for typical microvolt-scale signals.
    return np.round(sig).astype(np.int64)


# ============================================================
# 1. Compression-ratio floor
# ============================================================


class TestCompressionRatioFloor:

    def test_cr_above_1_8(self, pink_noise_21ch_2500):
        """CR floor: 21×2500 int16 (105 KB raw) must compress to < 58 KB.

        Current measured: ~2.3 on clinical EEG. The 1.8 floor flags a
        ~25% regression that would invalidate clinical storage budgets.
        """
        sig = pink_noise_21ch_2500
        raw_bytes = sig.size * 2  # 16-bit per sample
        encoded = _compress_bytes(sig.astype(np.float64), noise_bits=0)
        cr = raw_bytes / len(encoded)
        assert cr > 1.8, (
            f"CR regression: got {cr:.2f}, floor is 1.8 "
            f"(raw={raw_bytes} bytes, encoded={len(encoded)} bytes)"
        )


# ============================================================
# 2. Encode latency ceiling
# ============================================================


class TestEncodeLatency:

    def test_encode_below_80ms(self, pink_noise_21ch_2500):
        """Encode latency: median of 5 runs must finish under 80 ms.

        Current measured: ~3-15 ms with numba JIT warm. The 80 ms ceiling
        flags a 5-25× regression that would block real-time clinical
        capture (10 s window arrives every 10 s — encode must keep up).
        """
        sig = pink_noise_21ch_2500.astype(np.float64)
        # Warm the JIT + numba caches first (not measured).
        _compress_bytes(sig, noise_bits=0)

        latencies_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            _compress_bytes(sig, noise_bits=0)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        median_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
        assert median_ms < 80.0, (
            f"Encode regression: median {median_ms:.1f} ms, ceiling 80 ms. "
            f"Per-run: {[f'{ms:.1f}' for ms in latencies_ms]}"
        )


# ============================================================
# 3. Decode latency ceiling
# ============================================================


class TestDecodeLatency:

    def test_decode_below_40ms(self, pink_noise_21ch_2500):
        """Decode latency: median of 5 must finish under 40 ms.

        Decode is intrinsically faster than encode (no LPC analysis). The
        40 ms ceiling flags a 4-10× regression.
        """
        sig = pink_noise_21ch_2500.astype(np.float64)
        encoded = _compress_bytes(sig, noise_bits=0)
        # Warm.
        _decompress_bytes(encoded)

        latencies_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            _decompress_bytes(encoded)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        median_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
        assert median_ms < 40.0, (
            f"Decode regression: median {median_ms:.1f} ms, ceiling 40 ms. "
            f"Per-run: {[f'{ms:.1f}' for ms in latencies_ms]}"
        )


# ============================================================
# 4. Peak RSS delta (memory ceiling)
# ============================================================


class TestPeakRssDelta:

    def test_encode_decode_under_64mb_rss_delta(self, pink_noise_21ch_2500):
        """Peak RSS delta around encode+decode must be under 64 MB.

        Measures `ru_maxrss` before vs after a single roundtrip. Linux
        reports ru_maxrss in KB; bound is 64 MB to allow numba JIT cache
        and numpy temporaries while flagging a real leak.
        """
        sig = pink_noise_21ch_2500.astype(np.float64)
        # Warm caches first so JIT cost isn't billed.
        _decompress_bytes(_compress_bytes(sig, noise_bits=0))

        rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for _ in range(20):
            _decompress_bytes(_compress_bytes(sig, noise_bits=0))
        rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_mb = (rss_after_kb - rss_before_kb) / 1024.0
        assert delta_mb < 64.0, (
            f"Peak RSS regression: delta {delta_mb:.1f} MB, ceiling 64 MB. "
            f"Before {rss_before_kb} KB, after {rss_after_kb} KB."
        )
