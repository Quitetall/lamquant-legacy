"""L1 — Wire format constant pinning tests.

Every format constant in the codec gets an explicit assertion here. If any
constant changes, this file fails BEFORE the change can silently break
cross-language decoders, archived files, or deployed firmware.

AUDIT (2026-04-28): Created to close gap found by wire-format-auditor and
format-consistency-checker agents. Previously, format constants were only
exercised implicitly through roundtrip tests. A wrong constant could pass
all roundtrips (within a single Python process) but break cross-language
or cross-version decode.

Constants pinned:
  - Magic bytes: LML1, LMQ1, LQN1, LQL1, LQFT, LMA1
  - Header sizes: struct.calcsize validated against SIZE attributes
  - Codec parameters: DEFAULT_RANS_TOTAL, BIAS_CTX_LEN, Q_LPC
  - Quality mode mapping: FSQ_LEVELS_BY_MODE
  - Footer size: FOOTER_SIZE (was wrong — fixed 2026-04-28)
"""
import struct
import sys
import os

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))


# ============================================================
# Magic bytes — the identity of each format
# ============================================================

@pytest.mark.l1
class TestMagicBytes:
    """Every magic byte sequence is pinned to its exact value.

    If you need to change a magic byte, you are making a breaking format
    change. Update the spec, bump the version, and update this test.
    """

    def test_lml_per_window_magic(self):
        from lamquant_codec.ops.constants import MAGIC_LML
        assert MAGIC_LML == b'LML1', f"LML magic changed: {MAGIC_LML!r}"

    def test_lmq_per_window_magic(self):
        from lamquant_codec.ops.constants import MAGIC_LMQ
        assert MAGIC_LMQ == b'LMQ1', f"LMQ magic changed: {MAGIC_LMQ!r}"

    def test_lql_container_magic(self):
        from lamquant_codec.fileformat import MAGIC_LOSSLESS
        assert MAGIC_LOSSLESS == b'LQL1', f"Lossless container magic changed: {MAGIC_LOSSLESS!r}"

    def test_lqn_container_magic(self):
        from lamquant_codec.fileformat import MAGIC_NEURAL
        assert MAGIC_NEURAL == b'LQN1', f"Neural container magic changed: {MAGIC_NEURAL!r}"

    def test_lqft_footer_magic(self):
        from lamquant_codec.fileformat import MAGIC_FOOTER
        assert MAGIC_FOOTER == b'LQFT', f"Footer magic changed: {MAGIC_FOOTER!r}"

    def test_lma_archive_magic(self):
        from lamquant_codec.lma import LMA_MAGIC
        assert LMA_MAGIC == b'LMA1', f"LMA magic changed: {LMA_MAGIC!r}"

    def test_all_magics_are_4_bytes(self):
        """Every magic must be exactly 4 bytes (protocol assumption)."""
        from lamquant_codec.ops.constants import MAGIC_LML, MAGIC_LMQ
        from lamquant_codec.fileformat import MAGIC_LOSSLESS, MAGIC_NEURAL, MAGIC_FOOTER
        from lamquant_codec.lma import LMA_MAGIC

        for name, magic in [
            ('MAGIC_LML', MAGIC_LML), ('MAGIC_LMQ', MAGIC_LMQ),
            ('MAGIC_LOSSLESS', MAGIC_LOSSLESS), ('MAGIC_NEURAL', MAGIC_NEURAL),
            ('MAGIC_FOOTER', MAGIC_FOOTER), ('LMA_MAGIC', LMA_MAGIC),
        ]:
            assert len(magic) == 4, f"{name} is {len(magic)} bytes, expected 4"

    def test_all_magics_are_distinct(self):
        """No two format magics should collide."""
        from lamquant_codec.ops.constants import MAGIC_LML, MAGIC_LMQ
        from lamquant_codec.fileformat import MAGIC_LOSSLESS, MAGIC_NEURAL, MAGIC_FOOTER
        from lamquant_codec.lma import LMA_MAGIC

        all_magics = [MAGIC_LML, MAGIC_LMQ, MAGIC_LOSSLESS, MAGIC_NEURAL,
                      MAGIC_FOOTER, LMA_MAGIC]
        assert len(set(all_magics)) == len(all_magics), \
            f"Duplicate magic bytes: {[m for m in all_magics if all_magics.count(m) > 1]}"


# ============================================================
# Header/footer sizes — struct.calcsize must match SIZE attribute
# ============================================================

@pytest.mark.l1
class TestHeaderSizes:
    """struct.calcsize(STRUCT) must equal SIZE for every header type.

    A mismatch means padding has been introduced by the struct module,
    which would silently break all wire format readers.
    """

    def test_file_header_size(self):
        from lamquant_codec.fileformat import FILE_HEADER_SIZE
        assert FILE_HEADER_SIZE == 64, \
            f"FILE_HEADER_SIZE changed: {FILE_HEADER_SIZE}"

    def test_neural_window_header_struct_matches_size(self):
        from lamquant_codec.fileformat import NeuralWindowHeader
        calc = struct.calcsize(NeuralWindowHeader.STRUCT)
        assert calc == NeuralWindowHeader.SIZE == 26, \
            f"NeuralWindowHeader: calcsize={calc}, SIZE={NeuralWindowHeader.SIZE}, expected 26"

    def test_lossless_window_header_struct_matches_size(self):
        from lamquant_codec.fileformat import LosslessWindowHeader
        calc = struct.calcsize(LosslessWindowHeader.STRUCT)
        assert calc == LosslessWindowHeader.SIZE == 22, \
            f"LosslessWindowHeader: calcsize={calc}, SIZE={LosslessWindowHeader.SIZE}, expected 22"

    def test_index_entry_struct_matches_size(self):
        from lamquant_codec.fileformat import IndexEntry
        calc = struct.calcsize(IndexEntry.STRUCT)
        assert calc == IndexEntry.SIZE == 16, \
            f"IndexEntry: calcsize={calc}, SIZE={IndexEntry.SIZE}, expected 16"

    def test_footer_size(self):
        """Footer = 8B uint64 index_offset + 4B LQFT magic = 12 bytes.

        AUDIT (2026-04-28): This test was created specifically because
        FOOTER_SIZE was wrong (was 8, now 12). The reader had a two-pass
        workaround that masked the bug. This test prevents regression.
        """
        from lamquant_codec.fileformat import FOOTER_SIZE, MAGIC_FOOTER
        expected = struct.calcsize('<Q') + len(MAGIC_FOOTER)  # 8 + 4 = 12
        assert FOOTER_SIZE == expected == 12, \
            f"FOOTER_SIZE={FOOTER_SIZE}, expected {expected} (8B offset + 4B magic)"

    def test_lml1_per_window_header_size(self):
        """LML1 per-window packet header = 22 bytes (separate from container header)."""
        # The LML1 wire format header: magic(4) + n_ch(2) + T(2) + n_levels(1)
        # + klt_flag(1) + lpc_len(4) + sub_len(4) + crc(4) = 22 bytes
        lml1_struct = '<4sHHBBIII'
        assert struct.calcsize(lml1_struct) == 22, \
            f"LML1 header struct calcsize: {struct.calcsize(lml1_struct)}"


# ============================================================
# Codec parameters — values that affect wire output
# ============================================================

@pytest.mark.l1
class TestCodecParameters:
    """Codec parameters that are baked into the wire format.

    Changing any of these changes the compressed output. Every existing
    .lml/.lmq file becomes unreadable. These are format constants, not
    tuning knobs.
    """

    def test_default_rans_total(self):
        from lamquant_codec.ops.constants import DEFAULT_RANS_TOTAL
        assert DEFAULT_RANS_TOTAL == 4096, \
            f"DEFAULT_RANS_TOTAL changed: {DEFAULT_RANS_TOTAL} (breaks all .lmq files)"

    def test_bias_ctx_len(self):
        from lamquant_codec.ops.constants import BIAS_CTX_LEN
        assert BIAS_CTX_LEN == 32, \
            f"BIAS_CTX_LEN changed: {BIAS_CTX_LEN} (breaks lossless codec output)"

    def test_q_lpc_precision(self):
        from lamquant_codec.ops.constants import Q_LPC
        assert Q_LPC == 27, \
            f"Q_LPC changed: {Q_LPC} (breaks fixed-point LPC in lossless codec)"

    def test_default_lpc_order(self):
        from lamquant_codec.ops.constants import DEFAULT_LPC_ORDER
        assert DEFAULT_LPC_ORDER == 8, \
            f"DEFAULT_LPC_ORDER changed: {DEFAULT_LPC_ORDER}"

    def test_fsq_levels_by_mode_mapping(self):
        """Quality mode → FSQ levels mapping must be exact.

        This mapping is baked into the neural packet format. A wrong
        level means the decoder reconstructs with wrong quantization.
        """
        from lamquant_codec.ops.constants import (
            FSQ_LEVELS_BY_MODE,
            QUALITY_ALERTING, QUALITY_MONITORING, QUALITY_CLINICAL,
        )
        assert FSQ_LEVELS_BY_MODE[QUALITY_ALERTING] == 8, \
            f"Alerting FSQ level: {FSQ_LEVELS_BY_MODE[QUALITY_ALERTING]}"
        assert FSQ_LEVELS_BY_MODE[QUALITY_MONITORING] == 16, \
            f"Monitoring FSQ level: {FSQ_LEVELS_BY_MODE[QUALITY_MONITORING]}"
        assert FSQ_LEVELS_BY_MODE[QUALITY_CLINICAL] == 32, \
            f"Clinical FSQ level: {FSQ_LEVELS_BY_MODE[QUALITY_CLINICAL]}"

    def test_quality_mode_values(self):
        """Quality mode enum values are part of the wire format."""
        from lamquant_codec.ops.constants import (
            QUALITY_ALERTING, QUALITY_MONITORING, QUALITY_CLINICAL,
        )
        assert QUALITY_ALERTING == 0
        assert QUALITY_MONITORING == 1
        assert QUALITY_CLINICAL == 2

    def test_format_version(self):
        from lamquant_codec.fileformat import FORMAT_VERSION
        assert FORMAT_VERSION == 1, \
            f"FORMAT_VERSION changed: {FORMAT_VERSION}"


# ============================================================
# Cross-layer consistency — container vs per-window formats
# ============================================================

@pytest.mark.l1
class TestCrossLayerConsistency:
    """The container format (LQL1/LQN1) wraps the per-window format (LML1/LMQ1).
    Their header sizes must not overlap or conflict."""

    def test_container_header_larger_than_window_headers(self):
        """File header (64B) must be larger than any window header."""
        from lamquant_codec.fileformat import (
            FILE_HEADER_SIZE, NeuralWindowHeader, LosslessWindowHeader,
        )
        assert FILE_HEADER_SIZE > NeuralWindowHeader.SIZE, \
            "File header smaller than neural window header"
        assert FILE_HEADER_SIZE > LosslessWindowHeader.SIZE, \
            "File header smaller than lossless window header"

    def test_lma_version(self):
        from lamquant_codec.lma import LMA_VERSION
        assert LMA_VERSION == 1, f"LMA_VERSION changed: {LMA_VERSION}"
