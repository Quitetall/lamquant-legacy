"""
LamQuant file info — typed dataclass for .lml and .lmq file metadata.

The single source of truth for file inspection. Every tool that reads
an LML/LMQ file produces an LQFileInfo. Every tool that displays file
metadata consumes an LQFileInfo. No ad-hoc dict parsing.

Usage:
    from lamquant_codec.file_info import LQFileInfo
    info = LQFileInfo.from_lml("/path/to/file.lml")
    print(info.cr)           # 2.26
    print(info.lossless)     # True
    print(info.channels)     # ['EEG FP1-LE', 'EEG FP2-LE', ...]
"""
import json
import os
import struct
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class LQFileInfo:
    """Typed metadata for any LamQuant compressed file (.lml or .lmq).

    Frozen dataclass — immutable after creation. Serializable to dict/JSON.
    All future format versions must populate these fields.
    """

    # ── Identity ──
    path: str = ""
    filename: str = ""
    size_bytes: int = 0
    format: str = ""                  # "LML" or "LMQ"
    format_version: str = ""          # "LML1", "LMQ5", etc.

    # ── Signal ──
    n_channels: int = 0
    n_samples: int = 0
    n_windows: int = 0
    window_size: int = 0              # samples per window
    sample_rate: float = 0.0
    duration_s: float = 0.0
    channel_names: List[str] = field(default_factory=list)
    phys_unit: str = ""               # "uV", "mV", "V"

    # ── Compression ──
    raw_size_bytes: int = 0           # uncompressed int16 size
    cr: float = 0.0                   # compression ratio
    noise_bits: int = 0               # 0 = lossless
    lossless: bool = True
    n_levels: int = 0                 # DWT lifting depth
    klt: bool = False

    # ── Integrity ──
    signal_sha256: str = ""           # SHA-256 of the original signal
    crc32: str = ""                   # per-window CRC (format-level)

    # ── Provenance ──
    source_file: str = ""             # original EDF filename
    source_path: str = ""
    conversion_date: str = ""
    patient_code: str = ""
    gender: str = ""
    start_date: str = ""
    n_annotations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_lml(cls, path: str) -> "LQFileInfo":
        """Parse an LML file and return typed metadata."""
        path = str(path)
        fsize = os.path.getsize(path)

        with open(path, "rb") as f:
            data = f.read(min(fsize, 65536))

        magic = data[:4]
        if magic != b"LML1":
            raise ValueError(f"Not an LML file: magic={magic!r}")

        # Auto-detect: 20-byte v1 header (version=1 at bytes 4-5) vs
        # legacy 18-byte header (n_ch at bytes 4-5).
        probe = struct.unpack("<H", data[4:6])[0]
        if probe == 1:
            # 20-byte v1: magic(4) + version(2) + n_ch(2) + n_win(2) +
            #             total(4) + ws(2) + meta_len(4)
            if len(data) < 20:
                raise ValueError(f"Truncated LML v1 header: {len(data)} bytes")
            _, _ver, n_ch, n_windows, total_samples, window_size, meta_len = \
                struct.unpack("<4sHHHIHI", data[:20])
            hdr_size = 20
        else:
            # 18-byte legacy: magic(4) + n_ch(2) + n_win(2) + total(4) +
            #                 ws(2) + meta_len(4)
            if len(data) < 18:
                raise ValueError(f"Truncated LML header: {len(data)} bytes")
            _, n_ch, n_windows, total_samples, window_size, meta_len = \
                struct.unpack("<4sHHIHI", data[:18])
            hdr_size = 18

        meta = {}
        if meta_len > 0 and hdr_size + meta_len <= len(data):
            meta = json.loads(data[hdr_size:hdr_size + meta_len].decode("utf-8"))

        sr = float(meta.get("sample_rate", 250))
        raw_size = n_ch * total_samples * 2
        cr = raw_size / fsize if fsize > 0 else 0
        duration = total_samples / sr if sr > 0 else 0
        nb = meta.get("noise_bits", 0) if isinstance(meta, dict) else 0

        return cls(
            path=path,
            filename=os.path.basename(path),
            size_bytes=fsize,
            format="LML",
            format_version="LML1",
            n_channels=n_ch,
            n_samples=total_samples,
            n_windows=n_windows,
            window_size=window_size,
            sample_rate=sr,
            duration_s=round(duration, 1),
            channel_names=meta.get("channels", []),
            phys_unit=meta.get("phys_dim", ""),
            raw_size_bytes=raw_size,
            cr=round(cr, 3),
            noise_bits=nb,
            lossless=(nb == 0),
            n_levels=3,  # default for LML v5
            klt=False,
            signal_sha256=meta.get("signal_sha256", ""),
            source_file=meta.get("source_file", ""),
            source_path=meta.get("source_path", ""),
            conversion_date=meta.get("conversion_date", ""),
            patient_code=meta.get("patient_code", ""),
            gender=meta.get("gender", ""),
            start_date=meta.get("startdate", ""),
            n_annotations=len(meta.get("annotations", [])),
        )

    @classmethod
    def from_path(cls, path: str) -> "LQFileInfo":
        """Auto-detect format and parse."""
        path = str(path)
        if path.endswith(".lml"):
            return cls.from_lml(path)
        # Future: .lmq support
        raise ValueError(f"Unknown format: {path}")


# ────────────────────────────────────────────────────────────────────
# LQPacketHeader — typed replacement for peek_header() dict
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LQPacketHeader:
    """Header of a single LMQ4/LMQ5 compressed window packet."""
    version: str = ""          # "LMQ4" or "LMQ5"
    n_channels: int = 0
    n_samples: int = 0
    n_levels: int = 0
    klt: bool = False
    noise_bits: int = 0
    lossless: bool = True
    lpc_meta_bytes: int = 0
    payload_bytes: int = 0
    crc32: int = 0
    total_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────
# ConversionResult — typed replacement for convert_edf_to_lml() dict
# ────────────────────────────────────────────────────────────────────

@dataclass
class ConversionResult:
    """Result of an EDF → LML conversion. Replaces ad-hoc dict."""
    ok: bool = False
    error: Optional[str] = None

    # File stats (populated on success)
    source: str = ""
    n_windows: int = 0
    n_channels: int = 0
    total_samples: int = 0
    sample_rate: float = 0.0
    duration_s: float = 0.0
    compressed_size: int = 0
    raw_size: int = 0
    cr: float = 0.0
    signal_sha256: str = ""
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def success(cls, **kwargs) -> "ConversionResult":
        return cls(ok=True, **kwargs)

    @classmethod
    def failure(cls, error: str) -> "ConversionResult":
        return cls(ok=False, error=error[:200])


# ────────────────────────────────────────────────────────────────────
# SystemProfile — typed replacement for syscheck recommend() dict
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemProfile:
    """System benchmark and recommended configuration."""

    # Hardware
    platform: str = ""
    cpu_cores: int = 0
    ram_total_gib: float = 0.0
    ram_available_gib: float = 0.0
    disk_free_gib: float = 0.0

    # Benchmark
    compress_ms_per_window: float = 0.0
    decompress_ms_per_window: float = 0.0
    sha256_mibs: float = 0.0
    est_ms_per_file: float = 0.0

    # Recommendations
    recommended_workers: int = 0
    numba_cache_dir: str = ""  # set at runtime via tempfile.gettempdir()
    refresh_hz: float = 10.0

    # Estimates (optional, set when corpus is provided)
    corpus_files: int = 0
    single_thread_hours: float = 0.0
    parallel_hours: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
