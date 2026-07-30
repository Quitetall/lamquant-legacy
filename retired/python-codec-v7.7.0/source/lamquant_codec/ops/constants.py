"""Canonical wire-format and codec constants.

Single source of truth. Every module that needs these values imports
from here -- no local redefinitions.
"""

# -- Neural per-window packet (LMQ) --
MAGIC_LMQ = b'LMQ1'
DEFAULT_RANS_TOTAL = 4096

# -- Quality modes --
QUALITY_ALERTING = 0     # L3 only
QUALITY_MONITORING = 1   # L3 + L2
QUALITY_CLINICAL = 2     # L3 + L2 + L1

FSQ_LEVELS_BY_MODE = {
    QUALITY_ALERTING: 8,
    QUALITY_MONITORING: 16,
    QUALITY_CLINICAL: 32,
}

# -- Lossless per-window packet (LML) --
MAGIC_LML = b'LML1'

# -- DSP parameters --
BIAS_CTX_LEN = 32        # bias cancellation context window (samples)
Q_LPC = 27               # LPC coefficient fixed-point precision (bits)
DEFAULT_LPC_ORDER = 8
DEFAULT_AUTOCORR_LEN = 256

__all__ = [
    'MAGIC_LMQ', 'MAGIC_LML', 'DEFAULT_RANS_TOTAL',
    'QUALITY_ALERTING', 'QUALITY_MONITORING', 'QUALITY_CLINICAL',
    'FSQ_LEVELS_BY_MODE', 'BIAS_CTX_LEN', 'Q_LPC',
    'DEFAULT_LPC_ORDER', 'DEFAULT_AUTOCORR_LEN',
]
