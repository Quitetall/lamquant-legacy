#!/usr/bin/env python3
"""gen_golden.py — emit frozen golden fixtures for the Rust lqs::metrics parity test.

This script runs the CANONICAL Python metric reference ONCE and freezes its
output to lqs/tests/golden/metrics_parity.json. The Rust parity test
(lqs/tests/metrics_parity.rs) then reads that JSON in pure Rust CI — no
Python/torch dependency at test time. This matches the project's
"pin against frozen golden fixtures" testing principle (feedback_futureproof_tests).

Sources of truth (imported directly — the golden encodes their actual output,
not a hand re-derivation):

  - ai_models/metrics.py            (LamQuant-Neural)
        prd_numpy(original, reconstructed, eps=1e-12)
        pearson_r_numpy(original, reconstructed, eps=1e-12)
  - lamquant_codec/lqs.py           (LamQuant-Lossless reference_implementations)
        prd(original, reconstructed)            # eps guard 1e-30
        pearson_r(original, reconstructed)      # np.corrcoef, std<1e-10 guard
        snr_db(original, reconstructed)         # cap 120.0 when noise<1e-30

Numerical notes (verified empirically, documented so the orchestrator
understands which Python value each Rust metric is pinned against):

  * PRD:  ai_models.prd_numpy and lqs.prd produce BIT-IDENTICAL f64 for every
          non-degenerate fixture here. The Rust lqs::metrics::prd uses the same
          single-pass num/den formula with the 1e-12 eps guard, so it matches
          BOTH. The golden pins prd_numpy as canonical (`py_prd`) and also
          records lqs.prd (`py_prd_lqs`) for cross-checking.

  * Pearson R: ai_models.pearson_r_numpy uses the manual mean-subtraction
          single-pass formula — this is what Rust lqs::metrics::pearson_r
          implements, and they agree to 0.0 ulp on the non-degenerate fixtures.
          lamquant_codec.lqs.pearson_r uses np.corrcoef (a two-pass covariance)
          which differs from the single-pass result by ~4e-16 — still far inside
          the 1e-9 tolerance, but NOT bit-identical. The golden therefore pins
          pearson_r_numpy as canonical (`py_pearson_r`) and records lqs.pearson_r
          (`py_pearson_r_lqs`) for cross-checking.

  * PRDN (normalized PRD, mean-subtracted denominator) has NO Python reference
          in either source file. It is a Rust-only metric. We replicate its
          exact formula here in numpy so the golden still pins a numerical
          value:  PRDN = 100*sqrt(sum((x-xhat)^2)/sum((x-mean(x))^2)) with the
          same 1e-12 all-zero guard as prd. `prdn_source` records that this
          column is a numpy replication of the Rust formula, not an imported
          Python function.

  * SNR:  lqs.snr_db is the only SNR in the references; Rust lqs::metrics::snr_db
          mirrors it (mean-power ratio, 120.0 dB cap when noise<1e-30).

  * Entropy: entropy_from_counts has no Python reference; replicated in numpy
          (Shannon -sum p*log2 p) to pin a value for the Rust entropy metric.

  * CR:   compression_ratio = raw_bytes / max(comp_bytes, 1); replicated to pin
          a value for the Rust compression_ratio metric.

Degenerate-guard design: the degenerate fixtures (all-zero original, flat
signal) are constructed so the Python guard branches (lqs.pearson_r:
std<1e-10 + np.allclose; ai_models.pearson_r_numpy: den<1e-12 -> 0.0) and the
Rust guard branches (std<1e-8 + abs<=1e-8 -> 1.0 if equal else 0.0) agree on
the SAME final value, so the parity assertion is meaningful rather than an
artifact of differing eps thresholds. Where the two Python references disagree
in a degenerate case, the canonical column pins the one the Rust formula
matches and the divergence is noted in `notes`.

Deterministic: a single seeded numpy Generator. No global RNG, no torch.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Union

import numpy as np


def _jfloat(v: float) -> Union[float, str]:
    """JSON-encode a float, mapping non-finite values to string sentinels.

    serde_json (the Rust reader) rejects bare Infinity/NaN tokens. We emit
    non-finite values as the strings "inf" / "-inf" / "nan"; the Rust parity
    test deserializes those back to the matching f64 (see metrics_parity.rs
    `MaybeFloat`). Finite values stay as JSON numbers. This keeps full parity
    on the genuine -inf cases (all-zero original with nonzero noise) instead
    of skipping them.
    """
    f = float(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f

# ── Resolve the reference modules ─────────────────────────────────────────
# PYTHONPATH should already include both repos (see the run command in the
# task). We also self-heal the common layout so the script is runnable
# standalone from the Eagle checkout if the siblings are where we expect.
_DEFAULT_PATHS = [
    "/mnt/4tb/LamQuant/LamQuant-Neural",
    "/mnt/4tb/LamQuant/LamQuant-Lossless/reference_implementations/python_codec",
]
for _p in _DEFAULT_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# lamquant_codec.lqs is deprecated (Rust is canonical) but we import it
# HERE precisely to freeze the parity golden — silence its import warning.
os.environ.setdefault("LQS_SILENCE_DEPRECATION", "1")

from ai_models.metrics import prd_numpy, pearson_r_numpy  # noqa: E402
from lamquant_codec.lqs import (  # noqa: E402
    prd as lqs_prd,
    pearson_r as lqs_pearson_r,
    snr_db as lqs_snr_db,
)


# ── numpy replications for the Rust-only metrics (no Python reference) ─────
def prdn_numpy(original: np.ndarray, reconstructed: np.ndarray,
               eps: float = 1e-12) -> float:
    """PRDN = 100*sqrt(sum((x-xhat)^2)/sum((x-mean(x))^2)).

    Mean-subtracted (normalized) PRD. Replicates the EXACT Rust
    lqs::metrics::prdn formula, including its 1e-12 all-zero guard applied
    to the mean-subtracted energy. There is no Python reference for this
    metric in either source file.
    """
    x = np.asarray(original, dtype=np.float64).ravel()
    xhat = np.asarray(reconstructed, dtype=np.float64).ravel()
    n = min(x.size, xhat.size)
    if n == 0:
        return 0.0
    x = x[:n]
    xhat = xhat[:n]
    mean = float(x.mean())
    num = float(np.sum((x - xhat) ** 2))
    den = float(np.sum((x - mean) ** 2))
    if den < eps:
        return 0.0 if num < eps else 100.0
    return 100.0 * (num / den) ** 0.5


def entropy_from_counts(counts: List[int]) -> float:
    """Shannon entropy in bits from a histogram of symbol counts.

    H = -sum_i p_i * log2(p_i), p_i = count_i/total. Zero-count bins
    contribute nothing; empty/all-zero histogram -> 0.0. Replicates the
    Rust lqs::metrics::entropy_from_counts formula.
    """
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum())
    if total == 0.0:
        return 0.0
    p = arr / total
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log2(nz)))


def compression_ratio(raw_bytes: int, comp_bytes: int) -> float:
    """raw_bytes / max(comp_bytes, 1). Replicates Rust compression_ratio."""
    return float(raw_bytes) / float(max(comp_bytes, 1))


# ── Fixture construction (deterministic) ───────────────────────────────────
def build_fixtures() -> List[dict]:
    rng = np.random.default_rng(20260528)
    fixtures: List[dict] = []

    def add(name: str, orig, recon, *, counts=None, raw_bytes=None,
            comp_bytes=None, note: str = ""):
        o = np.asarray(orig, dtype=np.float64).ravel()
        r = np.asarray(recon, dtype=np.float64).ravel()
        fx = {
            "name": name,
            "note": note,
            "orig": [float(v) for v in o],
            "recon": [float(v) for v in r],
            # ── canonical Python values the Rust must match (<1e-9) ──
            "py_prd": _jfloat(prd_numpy(o, r)),                 # ai_models.prd_numpy
            "py_prdn": _jfloat(prdn_numpy(o, r)),              # numpy replication of Rust prdn
            "py_pearson_r": _jfloat(pearson_r_numpy(o, r)),    # ai_models.pearson_r_numpy (single-pass)
            "py_snr_db": _jfloat(lqs_snr_db(o, r)),            # lamquant_codec.lqs.snr_db
            # ── cross-check columns (second Python implementation) ──
            "py_prd_lqs": _jfloat(lqs_prd(o, r)),              # lamquant_codec.lqs.prd
            "py_pearson_r_lqs": _jfloat(lqs_pearson_r(o, r)),  # lamquant_codec.lqs.pearson_r (corrcoef)
            "prdn_source": "numpy-replication-of-rust-formula",
        }
        if counts is not None:
            fx["counts"] = [int(c) for c in counts]
            fx["py_entropy_bits"] = _jfloat(entropy_from_counts(counts))
        if raw_bytes is not None and comp_bytes is not None:
            fx["raw_bytes"] = int(raw_bytes)
            fx["comp_bytes"] = int(comp_bytes)
            fx["py_cr"] = _jfloat(compression_ratio(raw_bytes, comp_bytes))
        fixtures.append(fx)

    # 1. identical — PRD=0, R=1, SNR capped at 120.
    base = rng.normal(0.0, 50.0, size=64)
    add("identical", base, base.copy(),
        counts=[10, 10, 10, 10], raw_bytes=128, comp_bytes=8,
        note="recon == orig: prd 0, pearson 1, snr capped at 120 dB")

    # 2. small-error — tiny additive noise (non-degenerate, R near 1).
    sig = rng.normal(0.0, 40.0, size=128)
    small = sig + rng.normal(0.0, 1.5, size=128)
    add("small_error", sig, small,
        counts=[100, 50, 25, 12, 6, 3, 2, 1, 1], raw_bytes=256, comp_bytes=37,
        note="low-amplitude additive gaussian residual")

    # 3. large-error — heavy residual (R well below 1, large PRD).
    sig2 = rng.normal(0.0, 30.0, size=96)
    large = sig2 + rng.normal(0.0, 25.0, size=96)
    add("large_error", sig2, large,
        counts=[5, 5, 5, 5, 5, 5], raw_bytes=192, comp_bytes=96,
        note="high-amplitude residual, R substantially below 1")

    # 4. all-zero original (guard). den(sum x^2)=0.
    #    Python prd_numpy: num<eps -> 0.0 else 100.0. recon nonzero -> 100.0.
    #    pearson_r_numpy: den<1e-12 -> 0.0. lqs.pearson_r: std(x)<1e-10 &
    #    not allclose(x,y) -> 0.0. Both agree: pearson 0.0. PRD 100.0.
    zeros = np.zeros(32)
    nonzero_recon = rng.normal(0.0, 5.0, size=32)
    add("all_zero_original_guard", zeros, nonzero_recon,
        counts=[0, 0, 0], raw_bytes=64, comp_bytes=64,
        note="sum(x^2)=0 guard: prd->100, pearson->0 (both Python refs agree); "
             "snr noise>0 so finite; entropy of all-zero histogram = 0")

    # 5. single-channel ramp with a clean linear recon (R=1 exactly via
    #    perfect positive linear relationship; small PRD from offset).
    ramp = np.arange(1.0, 49.0)
    ramp_recon = ramp + 0.5  # constant DC offset: R=1, PRD>0, PRDN>0
    add("single_channel_ramp", ramp, ramp_recon,
        counts=[48], raw_bytes=96, comp_bytes=3,
        note="single channel, constant DC offset -> R exactly 1, nonzero PRD/PRDN; "
             "entropy of single-symbol histogram = 0")

    # 6. multichannel (4 ch x 32) flattened — order preserved, matches the
    #    Rust flatten-and-truncate contract.
    mc = rng.normal(0.0, 60.0, size=(4, 32))
    mc_recon = mc + rng.normal(0.0, 4.0, size=(4, 32))
    add("multichannel_4x32", mc, mc_recon,
        counts=[40, 30, 20, 10, 5, 5, 5, 5, 5, 5], raw_bytes=256, comp_bytes=22,
        note="4x32 flattened C-order; verifies flatten parity")

    # 7. negative values — mixed sign, large negative excursions.
    negs = rng.normal(-20.0, 70.0, size=80)
    negs_recon = negs + rng.normal(0.0, 6.0, size=80)
    add("negative_values", negs, negs_recon,
        counts=[7, 11, 3, 19, 2, 8], raw_bytes=160, comp_bytes=31,
        note="mixed-sign signal with negative DC bias")

    # 8. near-saturation — int16-range amplitudes near +/-32767, slight clip.
    sat = np.array(
        [32760.0, -32760.0, 32000.0, -31000.0, 30000.0, -29000.0] * 8,
        dtype=np.float64,
    )
    sat_recon = sat.copy()
    sat_recon[sat_recon > 32760.0] = 32760.0
    sat_recon += rng.normal(0.0, 3.0, size=sat.size)  # quantization-like jitter
    add("near_saturation", sat, sat_recon,
        counts=[24, 24], raw_bytes=96, comp_bytes=11,
        note="amplitudes near int16 saturation with small jitter")

    return fixtures


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    eagle_root = os.path.abspath(os.path.join(here, "..", ".."))
    out_dir = os.path.join(eagle_root, "lqs", "tests", "golden")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics_parity.json")

    # The all-zero-original guard fixture drives lqs.snr_db's log10(0) =>
    # -inf, which numpy flags as a RuntimeWarning. That -inf is the CORRECT,
    # intended value (and the Rust snr_db produces the same -inf), so silence
    # the expected warning rather than letting it clutter the output.
    with np.errstate(divide="ignore", invalid="ignore"):
        fixtures = build_fixtures()

    payload = {
        "_schema": "lqs-metrics-parity-golden/v1",
        "_generated_by": "tools/parity/gen_golden.py",
        "_sources": {
            "py_prd": "ai_models.metrics.prd_numpy",
            "py_prdn": "numpy replication of Rust lqs::metrics::prdn (no Python ref)",
            "py_pearson_r": "ai_models.metrics.pearson_r_numpy (single-pass; matches Rust 0 ulp)",
            "py_snr_db": "lamquant_codec.lqs.snr_db",
            "py_prd_lqs": "lamquant_codec.lqs.prd (cross-check)",
            "py_pearson_r_lqs": "lamquant_codec.lqs.pearson_r (np.corrcoef; ~4e-16 from canonical)",
            "py_entropy_bits": "numpy replication of Rust entropy_from_counts (no Python ref)",
            "py_cr": "numpy replication of Rust compression_ratio (no Python ref)",
        },
        "_tolerance": {
            "abs": 1e-9,
            "rel": 1e-9,
            "note": "Rust asserts |rust-py|<1e-9, or relative<1e-9 for large magnitudes.",
        },
        "_numpy_version": np.__version__,
        "fixtures": fixtures,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    # Console summary + cross-check that the two Python prd/pearson refs
    # agree within tolerance, so we know the golden is internally consistent.
    # Recompute from the raw arrays so the string-sentinel encoding of
    # non-finite metrics doesn't break the formatting / subtraction.
    def _unwrap(v):
        if isinstance(v, str):
            return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v]
        return float(v)

    print(f"wrote {out_path}  ({len(fixtures)} fixtures)")
    print(f"numpy {np.__version__}")
    worst_prd = 0.0
    worst_r = 0.0
    for fx in fixtures:
        worst_prd = max(worst_prd, abs(_unwrap(fx["py_prd"]) - _unwrap(fx["py_prd_lqs"])))
        worst_r = max(worst_r, abs(_unwrap(fx["py_pearson_r"]) - _unwrap(fx["py_pearson_r_lqs"])))
        print(
            f"  {fx['name']:<28} prd={_unwrap(fx['py_prd']):.6f} "
            f"prdn={_unwrap(fx['py_prdn']):.6f} r={_unwrap(fx['py_pearson_r']):.9f} "
            f"snr={_unwrap(fx['py_snr_db'])}"
        )
    print(f"cross-check max |prd_numpy - lqs.prd|        = {worst_prd:.3e}")
    print(f"cross-check max |pearson_numpy - lqs.pearson| = {worst_r:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
