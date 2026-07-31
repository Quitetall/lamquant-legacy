"""Quality contracts: what the codec promises, independent of how it's built.

A QualityContract defines the bounds a codec output must satisfy. Tests
assert against contracts, never against internal configuration. This
decouples testing from implementation permanently.

Two contracts match two modes:
    'lossless' — PRD=0, R=1.0, bit-exact, self-decodable (.lml)
    'neural'   — PRD/R/CR bounds with per-band clinical hierarchy (.lmq)

Usage:
    from lamquant_codec.contract import CONTRACTS, check_contract

    violations = check_contract(original, packet, CONTRACTS['neural'])
    assert not violations, "\\n".join(violations)

The per-band PRD bounds encode clinical importance directly:
    delta (0.5-4 Hz):  tight — sleep staging, encephalopathy
    theta (4-8 Hz):    tight — drowsiness, temporal lobe pathology
    alpha (8-13 Hz):   moderate — posterior dominant rhythm
    beta (13-30 Hz):   relaxed — frontal activity
    gamma (30+ Hz):    very relaxed — mostly EMG noise at 250 Hz

A decoder that nails delta at PRD<10% but synthesizes noisy gamma at
PRD=45% passes. That's clinically correct behavior.
"""

from dataclasses import dataclass
import numpy as np

# Single source of truth for dataclass definitions — types.py.
from lamquant_codec.codec_types import QualityContract, EEGPacket, TestVector
from lamquant_codec.benchmark import Benchmark


# ============================================================
# Production contracts
# ============================================================

CONTRACTS = {
    'lossless': QualityContract(
        mode='lossless',
        max_prd=0.0,
        min_r=1.0,
        min_cr=3.0,
        max_cr=10.0,
        lossless=True,
        bands=None,
        downstream_tasks=None,
    ),
    'neural': QualityContract(
        mode='neural',
        max_prd=20.0,       # conservative, decoder-dependent
        min_r=0.80,
        min_cr=200,
        max_cr=600,
        lossless=False,
        bands={
            'delta':  {'max_prd': 10.0},   # 0.5-4 Hz — clinically critical
            'theta':  {'max_prd': 12.0},   # 4-8 Hz — drowsiness, temporal
            'alpha':  {'max_prd': 15.0},   # 8-13 Hz — posterior dominant rhythm
            'beta':   {'max_prd': 25.0},   # 13-30 Hz — frontal, lower priority
            'gamma':  {'max_prd': 50.0},   # 30+ Hz — mostly EMG noise, relaxed
        },
        downstream_tasks={
            'seizure_detection': {
                'min_sensitivity': 0.98,
                'max_fpr_delta': 0.02,     # FPR increase vs uncompressed
            },
            'sleep_staging': {
                'min_kappa': 0.75,
            },
        },
    ),
}


# ============================================================
# Contract checking
# ============================================================

@dataclass
class Violation:
    """One contract violation."""
    field: str
    expected: str
    actual: str

    def __str__(self):
        return f"{self.field}: expected {self.expected}, got {self.actual}"


def check_contract(original: np.ndarray, packet: EEGPacket,
                   contract: QualityContract) -> list:
    """Check whether a codec output satisfies its quality contract.

    Args:
        original: [C, T] ground truth signal
        packet: EEGPacket from any codec path
        contract: QualityContract defining the bounds

    Returns:
        List of Violation objects. Empty list = contract satisfied.
    """
    violations = []

    # Lossless guarantee
    if contract.lossless:
        if not Benchmark.is_lossless(original, packet):
            max_err = Benchmark.max_error(original, packet)
            violations.append(Violation(
                'lossless', 'bit-exact (max_error=0)', f'max_error={max_err}'))

    # PRD (epsilon tolerance for float precision on lossless)
    prd = Benchmark.prd(original, packet)
    if prd > contract.max_prd + 1e-9:
        violations.append(Violation(
            'prd', f'<= {contract.max_prd}%', f'{prd:.2f}%'))

    # Pearson R (epsilon tolerance for float precision on lossless)
    r = Benchmark.pearson_r(original, packet)
    if r < contract.min_r - 1e-9:
        violations.append(Violation(
            'r', f'>= {contract.min_r}', f'{r:.4f}'))

    # Compression ratio
    cr = Benchmark.compression_ratio(original, packet)
    if cr < contract.min_cr:
        violations.append(Violation(
            'cr', f'>= {contract.min_cr}:1', f'{cr:.1f}:1'))
    if cr > contract.max_cr:
        violations.append(Violation(
            'cr', f'<= {contract.max_cr}:1', f'{cr:.1f}:1'))

    # Per-band PRD
    if contract.bands and original.shape[-1] >= 64:
        sample_rate = getattr(packet, 'sample_rate', 250)
        band_prds = Benchmark.per_band_prd(original, packet, sample_rate)
        for band_name, bounds in contract.bands.items():
            if band_name in band_prds:
                band_prd = band_prds[band_name]
                max_band_prd = bounds.get('max_prd', float('inf'))
                if band_prd > max_band_prd:
                    violations.append(Violation(
                        f'band_prd.{band_name}',
                        f'<= {max_band_prd}%',
                        f'{band_prd:.1f}%'))

    return violations


def check_contract_strict(original: np.ndarray, packet: EEGPacket,
                          contract: QualityContract) -> None:
    """Like check_contract but raises AssertionError on any violation.

    Use in tests:
        check_contract_strict(original, packet, CONTRACTS['lossless'])
    """
    violations = check_contract(original, packet, contract)
    if violations:
        msg = f"Contract '{contract.mode}' violated:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        raise AssertionError(msg)


__all__ = [
    'QualityContract', 'TestVector',  # re-exported from types.py for back-compat
    'CONTRACTS', 'Violation',
    'check_contract', 'check_contract_strict',
]
