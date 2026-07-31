"""Transform-graph DAG executor — invariants and serialization round-trip.

Pins lamquant_codec.transform_graph public API:
  - Op IDs (IDENTITY, CHANNEL_SPLIT, LIFTING_DWT, LPC_PREDICT, BIAS_CANCEL,
    GOLOMB_RICE, RANS_ENCODE, NOISE_STRIP, SPATIAL_PRED, DELTA_CODE,
    ZSTD_COMPRESS)
  - Op dataclass: to_bytes / from_bytes round-trip
  - TransformGraph: MAGIC, to_bytes / from_bytes round-trip, is_lossless,
    noise_bits_stripped
  - lml_v4_pipeline / lml_v41_pipeline / lml_clinical_pipeline
  - make_graph preset dispatch
"""
from __future__ import annotations

import pytest

from lamquant_codec import transform_graph as tg

pytestmark = pytest.mark.l3


# ============================================================
# 1. Op IDs are stable
# ============================================================


class TestOpIDs:

    def test_op_ids_pinned(self):
        assert tg.IDENTITY == 0x00
        assert tg.CHANNEL_SPLIT == 0x01
        assert tg.LIFTING_DWT == 0x02
        assert tg.LPC_PREDICT == 0x03
        assert tg.BIAS_CANCEL == 0x04
        assert tg.GOLOMB_RICE == 0x05
        assert tg.RANS_ENCODE == 0x06
        assert tg.NOISE_STRIP == 0x07
        assert tg.SPATIAL_PRED == 0x08
        assert tg.DELTA_CODE == 0x09
        assert tg.ZSTD_COMPRESS == 0x0A

    def test_op_names_table_complete(self):
        for op_id in (tg.IDENTITY, tg.CHANNEL_SPLIT, tg.LIFTING_DWT,
                      tg.LPC_PREDICT, tg.BIAS_CANCEL, tg.GOLOMB_RICE,
                      tg.RANS_ENCODE, tg.NOISE_STRIP, tg.SPATIAL_PRED,
                      tg.DELTA_CODE, tg.ZSTD_COMPRESS):
            assert op_id in tg.OP_NAMES
            assert isinstance(tg.OP_NAMES[op_id], str)


# ============================================================
# 2. Op serialization round-trip
# ============================================================


class TestOpSerialization:

    def test_op_no_params_roundtrip(self):
        op = tg.Op(tg.GOLOMB_RICE)
        data = op.to_bytes()
        recovered, consumed = tg.Op.from_bytes(data)
        assert consumed == len(data)
        assert recovered.op_id == tg.GOLOMB_RICE
        assert recovered.params == {}

    def test_op_with_params_roundtrip(self):
        op = tg.Op(tg.LIFTING_DWT, n_levels=3)
        data = op.to_bytes()
        recovered, consumed = tg.Op.from_bytes(data)
        assert consumed == len(data)
        assert recovered.op_id == tg.LIFTING_DWT
        assert recovered.params == {'n_levels': 3}

    def test_op_with_multiple_params_sorted(self):
        op = tg.Op(tg.LPC_PREDICT, order=8, autocorr_len=256)
        data = op.to_bytes()
        recovered, _ = tg.Op.from_bytes(data)
        # Order in dict round-trips deterministically (sorted on encode).
        assert recovered.params == {'order': 8, 'autocorr_len': 256}

    def test_op_name_property(self):
        assert tg.Op(tg.LIFTING_DWT).name == 'LIFTING_DWT'
        assert tg.Op(tg.GOLOMB_RICE).name == 'GOLOMB_RICE'

    def test_op_unknown_id_name(self):
        op = tg.Op(0xFE)  # not in OP_NAMES
        assert op.name.startswith('UNKNOWN_')

    def test_op_repr_includes_name_and_params(self):
        s = repr(tg.Op(tg.LPC_PREDICT, order=8))
        assert 'LPC_PREDICT' in s
        assert 'order=8' in s

    def test_op_to_bytes_rejects_non_ascii_key(self):
        """Found by hypothesis on 2026-05-05: previously raised
        UnicodeEncodeError mid-write. Now raises LmlInputError at the
        boundary."""
        from lamquant_codec.errors import LmlInputError
        op = tg.Op(tg.LPC_PREDICT, **{"orderª": 8})
        with pytest.raises(LmlInputError):
            op.to_bytes()

    def test_op_to_bytes_rejects_oversize_key(self):
        from lamquant_codec.errors import LmlInputError
        op = tg.Op(tg.LPC_PREDICT, **{"a" * 300: 8})
        with pytest.raises(LmlInputError):
            op.to_bytes()


# ============================================================
# 3. TransformGraph serialization + magic byte
# ============================================================


class TestTransformGraph:

    def test_magic_pinned(self):
        assert tg.TransformGraph.MAGIC == b'LMLG'
        assert len(tg.TransformGraph.MAGIC) == 4

    def test_empty_graph_roundtrip(self):
        g = tg.TransformGraph([])
        data = g.to_bytes()
        assert data.startswith(b'LMLG')
        assert data[4] == 0  # n_ops
        recovered, consumed = tg.TransformGraph.from_bytes(data)
        assert consumed == len(data)
        assert recovered.ops == []

    def test_graph_roundtrip_full_pipeline(self):
        ops = tg.lml_v41_pipeline(n_channels=21, n_levels=3, lpc_order=8)
        g = tg.TransformGraph(ops)
        data = g.to_bytes()
        assert data.startswith(b'LMLG')

        recovered, consumed = tg.TransformGraph.from_bytes(data)
        assert consumed == len(data)
        assert len(recovered.ops) == len(ops)
        for orig, rec in zip(ops, recovered.ops):
            assert orig.op_id == rec.op_id
            assert orig.params == rec.params

    def test_from_bytes_rejects_invalid_magic(self):
        bogus = b'XXXX\x00'
        with pytest.raises(ValueError, match=r'magic'):
            tg.TransformGraph.from_bytes(bogus)

    def test_size_bytes_matches_to_bytes_length(self):
        g = tg.TransformGraph(tg.lml_v4_pipeline())
        assert g.size_bytes == len(g.to_bytes())


# ============================================================
# 4. is_lossless / noise_bits_stripped semantics
# ============================================================


class TestLossyDetection:

    def test_v4_pipeline_is_lossless(self):
        g = tg.TransformGraph(tg.lml_v4_pipeline())
        assert g.is_lossless is True
        assert g.noise_bits_stripped == 0

    def test_v41_pipeline_is_lossless(self):
        g = tg.TransformGraph(tg.lml_v41_pipeline())
        assert g.is_lossless is True
        assert g.noise_bits_stripped == 0

    def test_clinical_pipeline_is_lossy(self):
        g = tg.TransformGraph(tg.lml_clinical_pipeline(noise_bits=3))
        assert g.is_lossless is False
        assert g.noise_bits_stripped == 3

    def test_compact_preset_strips_4_bits(self):
        g = tg.make_graph('compact')
        assert g.is_lossless is False
        assert g.noise_bits_stripped == 4


# ============================================================
# 5. make_graph preset dispatch
# ============================================================


class TestMakeGraphPresets:

    @pytest.mark.parametrize("preset", ['v4', 'v4.1', 'clinical', 'compact'])
    def test_known_presets_build(self, preset):
        g = tg.make_graph(preset)
        assert isinstance(g, tg.TransformGraph)
        assert len(g.ops) >= 4  # At minimum: split + lift + lpc + golomb

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match=r'Unknown preset'):
            tg.make_graph('made_up_recipe')

    def test_v4_pipeline_terminates_with_golomb(self):
        ops = tg.lml_v4_pipeline()
        assert ops[-1].op_id == tg.GOLOMB_RICE

    def test_v41_pipeline_inserts_bias_cancel(self):
        ops = tg.lml_v41_pipeline()
        op_ids = [op.op_id for op in ops]
        assert tg.BIAS_CANCEL in op_ids
        assert tg.GOLOMB_RICE == ops[-1].op_id
