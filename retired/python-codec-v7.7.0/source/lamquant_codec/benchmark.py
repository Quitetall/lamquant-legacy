"""Architecture-agnostic EEG quality benchmarks.

All methods take (original_signal, packet) where packet is an EEGPacket.
This module knows nothing about codec internals — it only measures
reconstruction quality from the standard packet interface.

Usage:
    from lamquant_codec.codec_types import EEGPacket
    from lamquant_codec.benchmark import Benchmark

    report = Benchmark.full_report(original, packet)
    # {'prd': 12.3, 'r': 0.89, 'cr': 274.0, 'snr_db': 18.2, ...}

    assert Benchmark.prd(original, packet) < 15.0
    assert Benchmark.compression_ratio(original, packet) > 250
"""

import numpy as np
from lamquant_codec.codec_types import EEGPacket

# Standard 10-20 EEG bands (Hz)
_BANDS = {
    'delta':  (0.5,  4.0),
    'theta':  (4.0,  8.0),
    'alpha':  (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}


class Benchmark:
    """Static methods for EEG reconstruction quality evaluation.

    Every method accepts:
        original: np.ndarray [C, T] — ground truth signal
        packet:   EEGPacket          — codec output

    Methods never access codec internals. If the codec changes but
    still produces valid packets, all benchmarks keep working.
    """

    @staticmethod
    def prd(original: np.ndarray, packet: EEGPacket) -> float:
        """Percentage Root-mean-square Difference.

        PRD = 100 * ||x - x_hat|| / ||x||

        PRD=0% is lossless. Clinical EEG typically requires PRD < 10%.
        """
        orig = original.flatten().astype(np.float64)
        recon = packet.signal.flatten().astype(np.float64)
        norm = np.sqrt(np.sum(orig ** 2))
        if norm < 1e-12:
            return 0.0 if np.allclose(orig, recon) else float('inf')
        return 100.0 * np.sqrt(np.sum((orig - recon) ** 2)) / norm

    @staticmethod
    def pearson_r(original: np.ndarray, packet: EEGPacket) -> float:
        """Pearson correlation coefficient (global, all channels flattened)."""
        orig = original.flatten().astype(np.float64)
        recon = packet.signal.flatten().astype(np.float64)
        if orig.std() < 1e-12 or recon.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(orig, recon)[0, 1])

    @staticmethod
    def per_channel_r(original: np.ndarray, packet: EEGPacket) -> np.ndarray:
        """Pearson R per channel. Returns [C] array."""
        C = original.shape[0]
        rs = np.zeros(C)
        for c in range(C):
            o = original[c].astype(np.float64)
            r = packet.signal[c].astype(np.float64)
            if o.std() < 1e-12 or r.std() < 1e-12:
                rs[c] = 0.0
            else:
                rs[c] = float(np.corrcoef(o, r)[0, 1])
        return rs

    @staticmethod
    def compression_ratio(original: np.ndarray, packet: EEGPacket) -> float:
        """Compression ratio = raw_bytes / compressed_bytes."""
        if packet.compressed_bytes <= 0:
            return float('inf')
        return packet.raw_bytes / packet.compressed_bytes

    @staticmethod
    def snr_db(original: np.ndarray, packet: EEGPacket) -> float:
        """Signal-to-noise ratio in dB.

        SNR = 10 * log10(||x||^2 / ||x - x_hat||^2)
        """
        orig = original.flatten().astype(np.float64)
        recon = packet.signal.flatten().astype(np.float64)
        signal_power = np.sum(orig ** 2)
        noise_power = np.sum((orig - recon) ** 2)
        if noise_power < 1e-20:
            return float('inf')  # lossless
        if signal_power < 1e-20:
            return 0.0
        return 10.0 * np.log10(signal_power / noise_power)

    @staticmethod
    def rmse(original: np.ndarray, packet: EEGPacket) -> float:
        """Root mean squared error (in signal units, typically uV)."""
        orig = original.flatten().astype(np.float64)
        recon = packet.signal.flatten().astype(np.float64)
        return float(np.sqrt(np.mean((orig - recon) ** 2)))

    @staticmethod
    def max_error(original: np.ndarray, packet: EEGPacket) -> float:
        """Maximum absolute sample error."""
        return float(np.max(np.abs(
            original.astype(np.float64) - packet.signal.astype(np.float64)
        )))

    @staticmethod
    def is_lossless(original: np.ndarray, packet: EEGPacket) -> bool:
        """Check bit-exact lossless reconstruction (integer domain)."""
        orig_int = np.round(original).astype(np.int64)
        recon_int = np.round(packet.signal).astype(np.int64)
        return bool(np.array_equal(orig_int, recon_int))

    @staticmethod
    def per_band_prd(original: np.ndarray, packet: EEGPacket,
                     sample_rate: int = 250) -> dict:
        """PRD per frequency band (delta/theta/alpha/beta/gamma).

        Uses FFT bandpass to isolate each band, then computes PRD
        on the band-limited signals. Useful for understanding which
        frequency content is best/worst preserved.
        """
        C, T = original.shape
        freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate)
        results = {}

        # Forward FFT is band-independent — compute each channel's rFFT
        # once and reuse the cached spectra for every band (the band only
        # selects a frequency `mask`, not the transform).
        orig_ffts = [np.fft.rfft(original[c].astype(np.float64)) for c in range(C)]
        recon_ffts = [np.fft.rfft(packet.signal[c].astype(np.float64)) for c in range(C)]

        for band_name, (lo, hi) in _BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_prd_sum = 0.0
            band_norm_sum = 0.0

            for c in range(C):
                orig_fft = orig_ffts[c]
                recon_fft = recon_ffts[c]

                orig_band = np.zeros_like(orig_fft)
                recon_band = np.zeros_like(recon_fft)
                orig_band[mask] = orig_fft[mask]
                recon_band[mask] = recon_fft[mask]

                orig_t = np.fft.irfft(orig_band, n=T)
                recon_t = np.fft.irfft(recon_band, n=T)

                band_prd_sum += np.sum((orig_t - recon_t) ** 2)
                band_norm_sum += np.sum(orig_t ** 2)

            if band_norm_sum < 1e-20:
                results[band_name] = 0.0
            else:
                results[band_name] = 100.0 * np.sqrt(band_prd_sum / band_norm_sum)

        return results

    @staticmethod
    def full_report(original: np.ndarray, packet: EEGPacket) -> dict:
        """Complete quality report — all metrics in one call.

        Returns dict with keys:
            prd, r, cr, snr_db, rmse, max_error, lossless,
            per_channel_r (array), per_band_prd (dict),
            mode, compressed_bytes, n_samples
        """
        return {
            'prd': Benchmark.prd(original, packet),
            'r': Benchmark.pearson_r(original, packet),
            'cr': Benchmark.compression_ratio(original, packet),
            'snr_db': Benchmark.snr_db(original, packet),
            'rmse': Benchmark.rmse(original, packet),
            'max_error': Benchmark.max_error(original, packet),
            'lossless': Benchmark.is_lossless(original, packet),
            'per_channel_r': Benchmark.per_channel_r(original, packet),
            'per_band_prd': Benchmark.per_band_prd(original, packet,
                                                    packet.sample_rate),
            'mode': packet.mode,
            'compressed_bytes': packet.compressed_bytes,
            'n_samples': packet.n_samples,
        }
