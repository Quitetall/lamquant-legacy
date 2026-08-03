"""Boundary datatypes for the LamQuant codec pipeline.

TYPES_VERSION = "1.0"  # Versioned independently from software version.
                        # Bump only when dataclass fields change.
                        # .lmq/.lml files record this version in the header.
                        # Old files remain readable via default field values.

Every module in the pipeline communicates through these typed dataclasses.
No module imports another module — only types.py. This is the single
source of truth for every boundary in the system.

Pipeline:
    RawEEG → preprocess → decompose → encode → compress → CompressedPacket
    CompressedPacket → decompress → decode → EEGPacket
    SubbandDecomposition → lossless → CompressedPacket (Mode 2 bypass)
    (RawEEG, EEGPacket) → benchmark → BenchmarkReport
    (BenchmarkReport, QualityContract) → check_contract → violations

Architecture diagram:
    ┌────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌────────────────┐
    │ RawEEG │───▶│preprocess │───▶│decompose │───▶│ encode   │───▶│  compress       │
    └────────┘    └───────────┘    └──────────┘    └──────────┘    └───────┬────────┘
                                        │                                  │
                                        │ (Mode 2)                         ▼
                                        └──────▶ lossless ──▶  CompressedPacket
                                                                           │
    ┌──────────┐    ┌──────────┐    ┌──────────────┐                       │
    │EEGPacket │◀───│ decode   │◀───│ decompress   │◀──────────────────────┘
    └────┬─────┘    └──────────┘    └──────────────┘
         │
         ▼
    ┌───────────┐    ┌────────────────┐    ┌───────────────┐
    │ benchmark │───▶│BenchmarkReport │───▶│check_contract │──▶ violations
    └───────────┘    └────────────────┘    └───────────────┘
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

TYPES_VERSION = "1.0"


@dataclass
class RawEEG:
    """ADC output. The universal input to everything.

    This is what the ADS1299 produces and what the preprocess step expects.
    All signals in the system start as RawEEG.
    """
    signal: np.ndarray              # [C, T] int16/int32/float32
    sample_rate: int = 250          # Hz
    channels: int = 21              # 10-20 montage
    timestamp_us: int = 0           # unix microseconds
    channel_labels: Optional[list] = None  # ['Fp1', 'Fp2', ...]

    def __post_init__(self):
        if self.signal.ndim == 1:
            self.signal = self.signal.reshape(1, -1)
        self.channels = self.signal.shape[0]
        if self.channel_labels is None:
            self.channel_labels = [f'ch{i}' for i in range(self.channels)]

    @property
    def n_samples(self) -> int:
        return self.signal.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass
class SubbandDecomposition:
    """Output of lifting DWT. Input to encoder OR lossless codec.

    Contains the 3-level Le Gall 5/3 decomposition of the LPC residual.
    The L3 approximation [C, 313] is what the neural encoder processes.
    The detail subbands are used by Mode 2 lossless or discarded in Mode 1.
    """
    l3_approx: np.ndarray           # [C, 313] — TNN input
    l3_detail: np.ndarray           # [C, 312]
    l2_detail: np.ndarray           # [C, 625]
    l1_detail: np.ndarray           # [C, 1250]
    lpc_coeffs: np.ndarray          # [C, order] Q31 or float64
    lpc_order: int = 8
    source_signal: Optional[np.ndarray] = None  # original [C, T] for reference


@dataclass
class LatentTokens:
    """Output of encoder. Input to entropy coder AND decoder.

    Quantized token representation of the L3 approximation.
    Shape is typically [32, 79] — 32 latent dims × 79 timesteps.
    """
    tokens: np.ndarray              # [D, T] quantized integers (FSQ indices)
    latent: Optional[np.ndarray] = None  # [D, T] continuous latent (pre-quant)
    fsq_levels: Optional[list] = None    # per-block FSQ level map
    snac_preset: str = 'compact'
    shape: tuple = (32, 79)
    vmin: float = -1.0              # quantization range
    vmax: float = 1.0
    side_info: Optional[dict] = None    # decoder side data (quality_mode, lpc_bytes, etc.)

    def __post_init__(self):
        if self.tokens is not None:
            self.shape = self.tokens.shape


@dataclass
class CompressedPacket:
    """Output of entropy coder. What goes over BLE / into .lmq/.lml files.

    This is the wire format — bytes that are transmitted or stored.
    The `data` field contains the complete packet including headers.
    """
    data: bytes                     # complete packet (header + payload)
    mode: str = 'neural'            # 'neural' or 'lossless'
    compressed_bytes: int = 0
    raw_bytes: int = 21 * 2500 * 2  # original signal size (int16)
    quality_mode: int = 0           # 0=alerting, 1=monitoring, 2=clinical
    metadata: dict = field(default_factory=dict)  # debugging, NOT for logic

    def __post_init__(self):
        self.compressed_bytes = len(self.data)

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / max(self.compressed_bytes, 1)


@dataclass
class EEGPacket:
    """Output of decoder. Universal interchange for benchmarks.

    This is what every decoder produces and every benchmark consumes.
    The signal is the reconstructed EEG in the original domain.
    """
    signal: np.ndarray              # [C, T] reconstructed EEG
    sample_rate: int = 250
    n_channels: int = 21
    mode: str = 'neural'            # 'neural', 'lossless', 'decoder'
    compressed_bytes: int = 0
    raw_bytes: int = 21 * 2500 * 2
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.signal.ndim == 1:
            raise ValueError("signal must be at least 2D [channels, samples]")
        if self.signal.ndim == 3 and self.signal.shape[0] == 1:
            self.signal = self.signal[0]
        self.n_channels = self.signal.shape[0]

    @property
    def n_samples(self) -> int:
        return self.signal.shape[1]

    @classmethod
    def from_reconstruction(cls, signal, compressed_bytes, mode='neural',
                            sample_rate=250, raw_bytes=None, metadata=None):
        if raw_bytes is None:
            raw_bytes = signal.shape[-2] * signal.shape[-1] * 2
        return cls(signal=np.asarray(signal, dtype=np.float64),
                   sample_rate=sample_rate, mode=mode,
                   compressed_bytes=compressed_bytes, raw_bytes=raw_bytes,
                   metadata=metadata or {})

    @classmethod
    def from_lossless(cls, signal, compressed_bytes, sample_rate=250, metadata=None):
        return cls.from_reconstruction(signal, compressed_bytes, mode='lossless',
                                       sample_rate=sample_rate, metadata=metadata)

    @classmethod
    def from_decoder(cls, signal, compressed_bytes, sample_rate=250, metadata=None):
        return cls.from_reconstruction(signal, compressed_bytes, mode='decoder',
                                       sample_rate=sample_rate, metadata=metadata)


@dataclass
class QualityContract:
    """What the codec promises for a given operating mode.

    Tests assert against this, never against internal configuration.
    The contract survives every architectural change.
    """
    mode: str
    max_prd: float
    min_r: float
    min_cr: float
    max_cr: float = float('inf')
    lossless: bool = False
    bands: Optional[dict] = None
    downstream_tasks: Optional[dict] = None

    def __post_init__(self):
        if self.lossless:
            if self.max_prd != 0.0:
                raise ValueError(
                    f"Lossless contract requires max_prd=0.0, got {self.max_prd}")
            if self.min_r != 1.0:
                raise ValueError(
                    f"Lossless contract requires min_r=1.0, got {self.min_r}")


@dataclass
class BenchmarkReport:
    """Output of any quality measurement.

    Contains all standard metrics for comparing codec output to original.
    The violations field is populated by check_contract().
    """
    prd: float = 0.0
    r: float = 0.0
    cr: float = 0.0
    snr_db: float = 0.0
    rmse: float = 0.0
    max_error: float = 0.0
    per_band_prd: dict = field(default_factory=dict)
    per_channel_r: Optional[np.ndarray] = None
    is_lossless: bool = False
    violations: list = field(default_factory=list)  # empty = pass
    n_samples: int = 0
    mode: str = ''
    compressed_bytes: int = 0


@dataclass
class TestVector:
    """Complete test case: input signal + codec pipeline + contract.

    The produce_packet callable is a black box — tests don't know or care
    what's inside. When a test fails, the developer inspects
    packet.metadata for debugging — assertions only touch the contract.
    """
    name: str
    signal: np.ndarray
    produce_packet: object   # Callable[[np.ndarray], EEGPacket]
    contract: QualityContract
