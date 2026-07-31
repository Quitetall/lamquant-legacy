"""Property-based tests — TransformGraph serialization invariants.

Generates random graphs (1-12 ops, with random params) and pins:

  P1. Op.to_bytes ↔ Op.from_bytes is an identity.
  P2. TransformGraph.to_bytes ↔ from_bytes is an identity.
  P3. TransformGraph.size_bytes always equals len(to_bytes()).
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from lamquant_codec import transform_graph as tg

pytestmark = [pytest.mark.l4]


_OP_IDS = [
    tg.IDENTITY, tg.CHANNEL_SPLIT, tg.LIFTING_DWT, tg.LPC_PREDICT,
    tg.BIAS_CANCEL, tg.GOLOMB_RICE, tg.RANS_ENCODE, tg.NOISE_STRIP,
    tg.SPATIAL_PRED, tg.DELTA_CODE, tg.ZSTD_COMPRESS,
]

# Param keys are part of the wire format (prefixed by u8 length, encoded
# as ASCII). The transform_graph spec restricts keys to printable ASCII
# letters + digits + underscore.
_PARAM_KEY = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_",
        max_codepoint=127,
    ),
    min_size=1, max_size=8,
)
_PARAM_VAL = st.integers(min_value=-(2**30), max_value=2**30 - 1)
_PARAMS = st.dictionaries(_PARAM_KEY, _PARAM_VAL, min_size=0, max_size=4)
# Filter out a generated `op_id` key in `params` — passing it via
# **params while also providing the positional arg raises TypeError
# (Python keyword/positional collision). Hypothesis WILL produce
# `op_id` as a dict key with the current 1-8 char ASCII alphabet.
_OP = st.builds(
    lambda op_id, params: tg.Op(
        op_id, **{k: v for k, v in params.items() if k != "op_id"}
    ),
    st.sampled_from(_OP_IDS), _PARAMS,
)
_GRAPH = st.lists(_OP, min_size=0, max_size=12).map(tg.TransformGraph)

_HYPO = settings(
    max_examples=80, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ============================================================
# P1. Op roundtrip
# ============================================================


class TestOpRoundtrip:

    @_HYPO
    @given(op=_OP)
    def test_op_to_bytes_from_bytes_identity(self, op):
        data = op.to_bytes()
        recovered, consumed = tg.Op.from_bytes(data)
        assert consumed == len(data)
        assert recovered.op_id == op.op_id
        assert recovered.params == op.params


# ============================================================
# P2. TransformGraph roundtrip
# ============================================================


class TestGraphRoundtrip:

    @_HYPO
    @given(graph=_GRAPH)
    def test_graph_to_bytes_from_bytes_identity(self, graph):
        data = graph.to_bytes()
        recovered, consumed = tg.TransformGraph.from_bytes(data)
        assert consumed == len(data)
        assert len(recovered.ops) == len(graph.ops)
        for orig, rec in zip(graph.ops, recovered.ops):
            assert orig.op_id == rec.op_id
            assert orig.params == rec.params


# ============================================================
# P3. size_bytes is len(to_bytes)
# ============================================================


class TestSizeBytesContract:

    @_HYPO
    @given(graph=_GRAPH)
    def test_size_bytes_matches_len(self, graph):
        assert graph.size_bytes == len(graph.to_bytes())
