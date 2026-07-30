"""
LamQuant Codec — EEG neural codec package.

Two file formats:
    .lmq — Neural compressed (ternary encoder + FSQ + rANS)
    .lml — Lossless (integer lifting + LPC + Golomb-Rice)

Core API:
    # Read either format (READ-ONLY reference reader for the legacy
    # divergent Python layout — see lamquant_codec.fileformat).
    with lq.open("session.lmq") as r:
        for window in r:
            print(window.timestamp, window.payload_size)

    # Writing the divergent Python container is no longer available. The
    # Rust PyO3 codec (lamquant_core, magic LML1) is the SOLE emitter of
    # canonical .lml:
    #     lamquant_core.container_write(...)

    # Benchmark any codec output
    report = lq.Benchmark.full_report(original, packet)

Fast import path
----------------
Symbols are grouped by how much they cost to load:

    CHEAP  (eager)  — types, registry, contract, benchmark, lqs, pipeline
                      primitives (compress/decompress). No torch, no model
                      weights. Import cost: ~50 ms.

    HEAVY  (lazy)   — TernaryCodec, SubbandCodec, LosslessCodec, Decoder,
                      Encoder. These pull torch and the student-model
                      module. Import cost on first touch: ~1400 ms.

Callers that only need types/contracts/benchmarks never pay the torch tax.
The heavy symbols are resolved via `module.__getattr__` (PEP 562) on first
attribute access.
"""

# Version: single source of truth is pyproject.toml.
# This reads it at import time so there's never a mismatch.
def _read_version():
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("lamquant")
    except Exception:
        pass
    # Fallback: read pyproject.toml directly via the relocation-
    # resilient repo-root resolver (lamquant_codec._paths).
    try:
        from lamquant_codec._paths import REPO_ROOT
        toml = REPO_ROOT / "pyproject.toml"
        for line in toml.read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split('"')[1]
    except Exception:
        pass
    return "0.0.0"

__version__ = _read_version()

# OpenHuman LamQuant — the product. OH! is the shorthand.
__product_name__ = "OpenHuman LamQuant"
__cli_name__ = "oh"
__cli_version__ = "1.0.0"

# ============================================================
# Eager (cheap) surface — no torch, no checkpoints.
# ============================================================

# Types — single source of truth for all boundary datatypes.
from lamquant_codec.codec_types import (
    TYPES_VERSION,
    RawEEG, SubbandDecomposition, LatentTokens,
    CompressedPacket, EEGPacket, QualityContract,
    BenchmarkReport, TestVector,
)

# Registry — plugin system for encoders, decoders, metrics.
from lamquant_codec.registry import (
    register_encoder, register_decoder, register_entropy_coder, register_metric,
    get_encoder, get_decoder, get_entropy_coder,
    list_encoders, list_decoders, list_entropy_coders, list_metrics,
)

# Pipeline functions that don't need torch at import time.
# (encode.py / decode.py import torch lazily inside the function body.)
from lamquant_codec.compress import compress
from lamquant_codec.decompress import decompress
from lamquant_codec.preprocess import preprocess

# Benchmark + contract + LQS — pure numpy / dataclasses, no torch.
from lamquant_codec.benchmark import Benchmark
from lamquant_codec.contract import (
    CONTRACTS, check_contract, check_contract_strict,
)
from lamquant_codec.lqs import (
    LQSLevel, LQS_LEVELS, BandRequirement, TaskRequirement,
    run_compliance, ComplianceResult,
)


# ============================================================
# Lazy (heavy) surface — torch + student model on first access.
# ============================================================

# Names that require torch / checkpoints. Resolved on first use.
_LAZY_MAP = {
    # Lossless codec class — stays in lamquant_codec.codec (PUBLIC).
    'LosslessCodec':  ('lamquant_codec.codec', 'LosslessCodec'),
    # Neural codec wrappers — MOVED to PRIVATE LamQuant-Neural on
    # 2026-05-29. Accessing these as `lamquant_codec.SubbandCodec` resolves
    # to `lamquant_neural.codec` IF that wheel is installed; otherwise
    # __getattr__ raises a clear "install lamquant-neural" error. The
    # public Lossless package no longer hard-depends on the neural defs.
    'TernaryCodec':   ('lamquant_neural.codec', 'TernaryCodec'),
    'SubbandCodec':   ('lamquant_neural.codec', 'SubbandCodec'),
    # Pipeline encode/decode (torch inside function bodies but eager module
    # import is harmless; we still lazy-load for symmetry with codecs).
    'encode':         ('lamquant_codec.encode',   'encode'),
    'decode':         ('lamquant_codec.decode',   'decode'),
    'decompose':      ('lamquant_codec.decompose','decompose'),
    # File I/O — READ-ONLY reference reader for the legacy divergent
    # Python layout. The writer path (NeuralWriter / LosslessWriter) and
    # the convert() transcoder were REMOVED (2026-05-28); the Rust PyO3
    # codec (lamquant_core, LML1) is the sole canonical emitter.
    'LMQReader':      ('lamquant_codec.fileformat', 'LMQReader'),
    'open':           ('lamquant_codec.fileformat', 'open_file'),
    'info':           ('lamquant_codec.fileformat', 'info'),
    'Window':         ('lamquant_codec.fileformat', 'Window'),
    'FileHeader':     ('lamquant_codec.fileformat', 'FileHeader'),
    'Decoder':        ('lamquant_codec.fileformat', 'Decoder'),
    'Encoder':        ('lamquant_codec.fileformat', 'Encoder'),
    'MAGIC_NEURAL':   ('lamquant_codec.fileformat', 'MAGIC_NEURAL'),
    'MAGIC_LOSSLESS': ('lamquant_codec.fileformat', 'MAGIC_LOSSLESS'),
    # Lossless pipeline compress (imports torch-free, but keeps symmetry).
    'lossless':       ('lamquant_codec.lossless',  'compress'),
    # MNE-Python integration — lazy because mne is heavy (~300 ms import).
    'read_raw':       ('lamquant_codec.mne_io', 'read_raw'),
    'read_raw_lml':   ('lamquant_codec.mne_io', 'read_raw_lml'),
    'read_raw_lmq':   ('lamquant_codec.mne_io', 'read_raw_lmq'),
    'write_raw':      ('lamquant_codec.mne_io', 'write_raw'),
    'write_lml':      ('lamquant_codec.mne_io', 'write_lml'),
    'write_lmq':      ('lamquant_codec.mne_io', 'write_lmq'),
}


def __getattr__(name):
    """PEP 562 lazy attribute loading.

    First access to a heavy symbol imports its module; subsequent accesses
    get the cached value from globals() (no overhead).
    """
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        import importlib
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            # The neural wrappers live in the PRIVATE LamQuant-Neural wheel.
            # The public Lossless package degrades with a clear error rather
            # than hard-failing at import time.
            if module_path.startswith("lamquant_neural"):
                raise ImportError(
                    f"lamquant_codec.{name} now lives in LamQuant-Neural "
                    f"(module {module_path!r}); install it with "
                    f"`pip install lamquant-neural`. The lossless LML/LMA "
                    f"path does not require it."
                ) from exc
            raise
        value = getattr(mod, attr)
        globals()[name] = value   # cache on the package — zero cost thereafter
        return value
    raise AttributeError(f"module 'lamquant_codec' has no attribute {name!r}")


def __dir__():
    """Expose both eager and lazy names to dir() and autocomplete."""
    return sorted(set(globals()) | set(_LAZY_MAP))


def warm_jit() -> None:
    """Pre-compile numba JIT'd lossless DSP functions.

    Call this once at app/GUI startup so the first real `.lml` decode
    doesn't pay the ~100 ms one-time numba compilation cost. After the
    first run on any given machine, the compiled binary is cached in
    `__pycache__` and subsequent process launches are near-instant.

    Safe to call multiple times — numba's compile cache makes repeats
    free. Returns nothing; raises if numba/scipy/ops modules fail
    to import (which would mean a broken install).
    """
    import numpy as np
    from lamquant_codec.ops.lpc import (
        synthesize_int as lpc_synthesize_int,
        analyze_int as lpc_analyze_int,
        synthesize_channel as lpc_synthesize_channel,
        analyze_channel as lpc_analyze_channel,
    )
    from lamquant_codec.ops.lifting import (
        forward_1d_int as lifting_1d_forward_int,
        inverse_1d_int as lifting_1d_inverse_int,
        forward_1d as lifting_1d_forward,
        inverse_1d as lifting_1d_inverse,
    )

    # Tiny throwaway tensors of the exact dtypes the JIT signature is pinned to.
    sig_int = np.zeros(64, dtype=np.int64)
    sig_f = np.zeros(64, dtype=np.float64)
    coeffs_q27 = np.zeros(8, dtype=np.int32)
    coeffs_f = np.zeros(8, dtype=np.float64)

    # Trigger compilation of the two JIT primitives (one int64, one float64).
    _ = lpc_synthesize_int(sig_int, coeffs_q27, 8)
    _ = lpc_synthesize_channel(sig_f, coeffs_f)

    # Warm vectorised numpy paths (no JIT but exercises caches / ufuncs).
    _ = lpc_analyze_int(sig_int, coeffs_f, 8)
    _ = lpc_analyze_channel(sig_f, order=8, autocorr_len=32)
    _ = lifting_1d_forward_int(sig_int)
    _ = lifting_1d_inverse_int(np.zeros(32, dtype=np.int64),
                                np.zeros(32, dtype=np.int64))
    _ = lifting_1d_forward(sig_f)
    _ = lifting_1d_inverse(np.zeros(32, dtype=np.float64),
                            np.zeros(32, dtype=np.float64))

    # JIT'd Golomb-Rice + rANS — these compile on first call to encode/decode.
    from lamquant_codec.ops.golomb import encode_dense, decode_dense
    from lamquant_codec.ops.rans import (
        compute_freq, encode_with_freq, decode as rans_decode,
    )
    coeffs_int = np.zeros(8, dtype=np.int64)
    gr_bytes = encode_dense(coeffs_int)
    _ = decode_dense(gr_bytes)
    syms = np.zeros(8, dtype=np.int64)
    freq = compute_freq(syms, n_sym=4, total_freq=4096)
    rans_bytes = encode_with_freq(syms, freq, total_freq=4096)
    _ = rans_decode(rans_bytes, freq, 8, total_freq=4096)


__all__ = [
    # Eager pipeline
    "preprocess", "compress", "decompress",
    # JIT warmup helper for GUIs / long-lived apps
    "warm_jit",
    # MNE-Python integration (lazy — mne is heavy)
    "read_raw", "read_raw_lml", "read_raw_lmq",
    "write_raw", "write_lml", "write_lmq",
    # Lazy pipeline
    "decompose", "encode", "decode",
    # Legacy codec classes (lazy)
    "TernaryCodec", "SubbandCodec", "LosslessCodec",
    # Packet + benchmarks
    "EEGPacket", "Benchmark",
    # Boundary datatypes
    "RawEEG", "SubbandDecomposition", "LatentTokens", "CompressedPacket",
    "BenchmarkReport", "TYPES_VERSION",
    # File I/O — READ-ONLY reference reader (lazy). Writer + convert removed.
    "LMQReader",
    "open", "info", "Window", "FileHeader",
    "Decoder", "Encoder", "MAGIC_NEURAL", "MAGIC_LOSSLESS",
    # Contracts
    "QualityContract", "CONTRACTS", "TestVector",
    "check_contract", "check_contract_strict",
    # LQS open standard
    "LQSLevel", "LQS_LEVELS", "BandRequirement", "TaskRequirement",
    "run_compliance", "ComplianceResult",
]
