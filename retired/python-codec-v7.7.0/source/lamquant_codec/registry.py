"""Plugin registry for LamQuant codec components.

Every encoder, decoder, entropy coder, and metric is a registered plugin.
Adding a new component is one file with a decorator — no other file changes.

Usage:
    # Register a new encoder
    @register_encoder("tnn_v1")
    class TNNEncoder:
        def encode(self, decomposition: SubbandDecomposition) -> LatentTokens:
            ...

    # Look up and use
    encoder = get_encoder("tnn_v1")
    tokens = encoder.encode(decomposition)

    # List available plugins
    list_encoders()  # ['tnn_v1', 'tnn_v2_dwsep']
"""

from typing import Callable, Dict, Any

# Global registries
ENCODERS: Dict[str, Any] = {}
DECODERS: Dict[str, Any] = {}
ENTROPY_CODERS: Dict[str, Any] = {}
METRICS: Dict[str, Callable] = {}
PREPROCESSORS: Dict[str, Any] = {}


def register_encoder(name: str):
    """Register an encoder plugin."""
    def decorator(cls):
        ENCODERS[name] = cls
        return cls
    return decorator


def register_decoder(name: str):
    """Register a decoder plugin."""
    def decorator(cls):
        DECODERS[name] = cls
        return cls
    return decorator


def register_entropy_coder(name: str):
    """Register an entropy coder plugin."""
    def decorator(cls):
        ENTROPY_CODERS[name] = cls
        return cls
    return decorator


def register_metric(name: str):
    """Register a quality metric function."""
    def decorator(fn):
        METRICS[name] = fn
        return fn
    return decorator


def register_preprocessor(name: str):
    """Register a preprocessor plugin."""
    def decorator(cls):
        PREPROCESSORS[name] = cls
        return cls
    return decorator


# Lookup helpers
def get_encoder(name: str):
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {list(ENCODERS)}")
    return ENCODERS[name]()


def get_decoder(name: str):
    if name not in DECODERS:
        raise KeyError(f"Unknown decoder '{name}'. Available: {list(DECODERS)}")
    return DECODERS[name]()


def get_entropy_coder(name: str):
    if name not in ENTROPY_CODERS:
        raise KeyError(f"Unknown entropy coder '{name}'. Available: {list(ENTROPY_CODERS)}")
    return ENTROPY_CODERS[name]()


def list_encoders() -> list:
    return list(ENCODERS.keys())

def list_decoders() -> list:
    return list(DECODERS.keys())

def list_entropy_coders() -> list:
    return list(ENTROPY_CODERS.keys())

def list_metrics() -> list:
    return list(METRICS.keys())
