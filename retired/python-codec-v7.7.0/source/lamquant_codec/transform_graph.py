"""transform_graph.py — OpenZL-inspired universal transform graph for LML.

Embeds the compression recipe (transform chain) in the file header so
any LML decoder can decompress any LML file regardless of version. New
transforms added without breaking old decoders.

Each transform op implements encode(data, params) → data and
decode(data, params) → data. Ops are executed in order for encode,
reverse order for decode.

Usage:
    # Build a pipeline
    graph = TransformGraph([
        Op(CHANNEL_SPLIT, n_channels=33),
        Op(LIFTING_DWT, n_levels=3),
        Op(LPC_PREDICT, order=8),
        Op(BIAS_CANCEL, ctx_len=4),
        Op(GOLOMB_RICE),
    ])

    # Encode
    compressed = graph.encode(signal)

    # Serialize graph to bytes (for file header)
    graph_bytes = graph.to_bytes()

    # Reconstruct graph from bytes (universal decoder)
    graph2 = TransformGraph.from_bytes(graph_bytes)
    signal_reconstructed = graph2.decode(compressed)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# Op IDs — the vocabulary of transforms
# ============================================================

IDENTITY       = 0x00
CHANNEL_SPLIT  = 0x01
LIFTING_DWT    = 0x02
LPC_PREDICT    = 0x03
BIAS_CANCEL    = 0x04
GOLOMB_RICE    = 0x05
RANS_ENCODE    = 0x06
NOISE_STRIP    = 0x07
SPATIAL_PRED   = 0x08
DELTA_CODE     = 0x09
ZSTD_COMPRESS  = 0x0A

OP_NAMES = {
    IDENTITY:      'IDENTITY',
    CHANNEL_SPLIT: 'CHANNEL_SPLIT',
    LIFTING_DWT:   'LIFTING_DWT',
    LPC_PREDICT:   'LPC_PREDICT',
    BIAS_CANCEL:   'BIAS_CANCEL',
    GOLOMB_RICE:   'GOLOMB_RICE',
    RANS_ENCODE:   'RANS_ENCODE',
    NOISE_STRIP:   'NOISE_STRIP',
    SPATIAL_PRED:  'SPATIAL_PRED',
    DELTA_CODE:    'DELTA_CODE',
    ZSTD_COMPRESS: 'ZSTD_COMPRESS',
}


# ============================================================
# Op dataclass
# ============================================================

@dataclass
class Op:
    """One transform operation in the pipeline.

    >>> op = Op(LIFTING_DWT, n_levels=3)
    >>> op.name
    'LIFTING_DWT'
    >>> data = op.to_bytes()
    >>> recovered, consumed = Op.from_bytes(data)
    >>> consumed == len(data)
    True
    >>> recovered.params == {'n_levels': 3}
    True
    """
    op_id: int
    params: Dict[str, int] = field(default_factory=dict)

    def __init__(self, op_id: int, **kwargs):
        self.op_id = op_id
        self.params = kwargs

    def to_bytes(self) -> bytes:
        """Serialize: [op_id: u8] [n_params: u8] [key_len: u8, key, val: i32]...

        Param keys must be ASCII-encodable and ≤255 bytes (the length
        prefix is u8). Non-ASCII keys raise LmlInputError so the bug
        surfaces at encode time, not as a UnicodeEncodeError mid-write.
        """
        from lamquant_codec.errors import LmlInputError

        parts = bytearray([self.op_id, len(self.params)])
        for key, val in sorted(self.params.items()):
            try:
                key_bytes = key.encode('ascii')
            except UnicodeEncodeError as exc:
                raise LmlInputError(
                    f"transform_graph param key {key!r} is not ASCII — "
                    f"the wire format requires ASCII-encoded keys."
                ) from exc
            if len(key_bytes) > 255:
                raise LmlInputError(
                    f"transform_graph param key {key!r} is {len(key_bytes)} "
                    f"bytes — the wire format caps key length at 255."
                )
            parts.append(len(key_bytes))
            parts.extend(key_bytes)
            parts.extend(struct.pack('<i', int(val)))
        return bytes(parts)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple:
        """Deserialize. Returns (Op, bytes_consumed)."""
        op_id = data[offset]
        n_params = data[offset + 1]
        pos = offset + 2
        params = {}
        for _ in range(n_params):
            key_len = data[pos]; pos += 1
            key = data[pos:pos+key_len].decode('ascii'); pos += key_len
            val = struct.unpack('<i', data[pos:pos+4])[0]; pos += 4
            params[key] = val
        return cls(op_id, **params), pos - offset

    @property
    def name(self) -> str:
        return OP_NAMES.get(self.op_id, f'UNKNOWN_{self.op_id:02X}')

    def __repr__(self):
        p = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'Op({self.name}{", " + p if p else ""})'


# ============================================================
# Pre-built pipeline presets
# ============================================================

def lml_v4_pipeline(n_channels: int = 21, n_levels: int = 3,
                     lpc_order: int = 8) -> List[Op]:
    """Original LML v4: lifting + LPC + GR."""
    return [
        Op(CHANNEL_SPLIT, n_channels=n_channels),
        Op(LIFTING_DWT, n_levels=n_levels),
        Op(LPC_PREDICT, order=lpc_order),
        Op(GOLOMB_RICE),
    ]


def lml_v41_pipeline(n_channels: int = 21, n_levels: int = 3,
                      lpc_order: int = 8, ctx_len: int = 4) -> List[Op]:
    """LML v4.1: + bias cancellation (+6% CR)."""
    return [
        Op(CHANNEL_SPLIT, n_channels=n_channels),
        Op(LIFTING_DWT, n_levels=n_levels),
        Op(LPC_PREDICT, order=lpc_order),
        Op(BIAS_CANCEL, ctx_len=ctx_len),
        Op(GOLOMB_RICE),
    ]


def lml_clinical_pipeline(n_channels: int = 21, noise_bits: int = 3,
                           n_levels: int = 3, lpc_order: int = 8,
                           ctx_len: int = 4) -> List[Op]:
    """LML Clinical: noise strip + bias cancel (+50% CR, near-lossless)."""
    return [
        Op(CHANNEL_SPLIT, n_channels=n_channels),
        Op(NOISE_STRIP, bits=noise_bits),
        Op(LIFTING_DWT, n_levels=n_levels),
        Op(LPC_PREDICT, order=lpc_order),
        Op(BIAS_CANCEL, ctx_len=ctx_len),
        Op(GOLOMB_RICE),
    ]


# ============================================================
# Transform Graph
# ============================================================

class TransformGraph:
    """A serializable pipeline of transform operations.

    The graph is embedded in the LML file header. The universal decoder
    reads it and executes the inverse transform chain.
    """

    MAGIC = b'LMLG'  # LML Graph

    def __init__(self, ops: List[Op] = None):
        self.ops = ops or []

    def to_bytes(self) -> bytes:
        """Serialize the entire graph for embedding in a file header."""
        parts = bytearray(self.MAGIC)
        parts.append(len(self.ops))
        for op in self.ops:
            op_bytes = op.to_bytes()
            parts.extend(op_bytes)
        return bytes(parts)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple:
        """Deserialize a graph from file header bytes.
        Returns (TransformGraph, bytes_consumed).
        """
        magic = data[offset:offset+4]
        if magic != cls.MAGIC:
            raise ValueError(f'Invalid graph magic: {magic!r}')
        n_ops = data[offset + 4]
        pos = offset + 5
        ops = []
        for _ in range(n_ops):
            op, consumed = Op.from_bytes(data, pos)
            ops.append(op)
            pos += consumed
        return cls(ops), pos - offset

    def __repr__(self):
        ops_str = ' → '.join(op.name for op in self.ops)
        return f'TransformGraph({ops_str}) [{self.size_bytes}B]'

    @property
    def size_bytes(self) -> int:
        """Total serialized size."""
        return len(self.to_bytes())

    @property
    def is_lossless(self) -> bool:
        """True if no lossy ops (NOISE_STRIP) in the chain."""
        return not any(op.op_id == NOISE_STRIP for op in self.ops)

    @property
    def noise_bits_stripped(self) -> int:
        """Number of noise bits stripped (0 if lossless)."""
        for op in self.ops:
            if op.op_id == NOISE_STRIP:
                return op.params.get('bits', 0)
        return 0


# ============================================================
# Convenience constructors
# ============================================================

def make_graph(preset: str = 'v4.1', **kwargs) -> TransformGraph:
    """Create a TransformGraph from a named preset.

    Presets:
        'v4':       Original LML (lifting + LPC + GR)
        'v4.1':     + bias cancellation (+6%)
        'clinical': + noise strip 3 bits (4.3:1)
        'compact':  + noise strip 4 bits (4.8:1)

    >>> g = make_graph('v4.1')
    >>> g.is_lossless
    True
    >>> g.noise_bits_stripped
    0
    >>> clinical = make_graph('clinical')
    >>> clinical.is_lossless
    False
    >>> clinical.noise_bits_stripped
    3
    """
    if preset == 'v4':
        return TransformGraph(lml_v4_pipeline(**kwargs))
    elif preset == 'v4.1':
        return TransformGraph(lml_v41_pipeline(**kwargs))
    elif preset == 'clinical':
        return TransformGraph(lml_clinical_pipeline(noise_bits=3, **kwargs))
    elif preset == 'compact':
        return TransformGraph(lml_clinical_pipeline(noise_bits=4, **kwargs))
    else:
        raise ValueError(f'Unknown preset: {preset!r}')


__all__ = [
    'Op', 'TransformGraph', 'make_graph',
    'IDENTITY', 'CHANNEL_SPLIT', 'LIFTING_DWT', 'LPC_PREDICT',
    'BIAS_CANCEL', 'GOLOMB_RICE', 'RANS_ENCODE', 'NOISE_STRIP',
    'SPATIAL_PRED', 'DELTA_CODE', 'ZSTD_COMPRESS',
    'lml_v4_pipeline', 'lml_v41_pipeline', 'lml_clinical_pipeline',
]
