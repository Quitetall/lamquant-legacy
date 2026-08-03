"""Decomposition: RawEEG → SubbandDecomposition.

LPC (per-channel, order 8 by default) + 3-level Le Gall 5/3 lifting DWT.
Produces the L3 approximation that feeds the neural encoder, plus the
three detail subbands used by the lossless path (Mode 2).

This is the second pipeline stage after preprocess.
"""
import numpy as np
from lamquant_codec.codec_types import RawEEG, SubbandDecomposition

# LPC + lifting primitives live in lamquant_codec.ops/.
from lamquant_codec.ops.pipeline import preprocess_subband_single


def decompose(raw: RawEEG, lpc_order: int = 8, autocorr_len: int = 256) -> SubbandDecomposition:
    """Run LPC + 3-level lifting DWT on a RawEEG window.

    Args:
        raw: Preprocessed RawEEG [C, T=2500].
        lpc_order: LPC analysis order (default 8).
        autocorr_len: Autocorrelation window length (default 256).

    Returns:
        SubbandDecomposition containing l3_approx [C, 313] + detail subbands
        + per-channel LPC coefficients.
    """
    sig = raw.signal
    if sig.ndim != 2:
        raise ValueError(f"Expected [C, T] RawEEG.signal, got shape {sig.shape}")

    l3, coeffs, subs = preprocess_subband_single(
        sig.astype(np.float32), order=lpc_order, autocorr_len=autocorr_len)

    # subs is a list[dict] keyed by 'l1_detail', 'l2_detail', 'l3_detail',
    # 'l3_approx' — one entry per channel. Stack detail bands across channels.
    C = sig.shape[0]
    if subs:
        l1_detail = np.stack([subs[c]['l1_detail'] for c in range(C)], axis=0)
        l2_detail = np.stack([subs[c]['l2_detail'] for c in range(C)], axis=0)
        l3_detail = np.stack([subs[c]['l3_detail'] for c in range(C)], axis=0)
    else:
        l1_detail = np.zeros((C, 1250))
        l2_detail = np.zeros((C, 625))
        l3_detail = np.zeros((C, 312))
    if coeffs is not None and len(coeffs) > 0:
        lpc_coeffs = np.asarray(coeffs)
    else:
        lpc_coeffs = np.zeros((C, lpc_order))

    return SubbandDecomposition(
        l3_approx=np.asarray(l3),
        l3_detail=l3_detail,
        l2_detail=l2_detail,
        l1_detail=l1_detail,
        lpc_coeffs=lpc_coeffs,
        lpc_order=lpc_order,
        source_signal=sig,
    )


__all__ = ['decompose']
