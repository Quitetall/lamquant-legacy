"""ADR 0114 N2 — the Neural Evidence Graph type boundary across PyO3.

The Rust side makes "a Generated node can never reach a Measured consumer" a
compile error (see lamquant-neg's trybuild UI test). Python has no static types,
so the boundary is enforced at the `PyNeg` typed accessors: `.measured(id)` on a
non-measured node raises `ValueError`. These tests pin that runtime twin, plus
the fail-closed verify + content-address round-trip.
"""

import pytest

pytestmark = pytest.mark.rust


def test_typed_accessor_raises_on_wrong_class(rust_wheel):
    g = rust_wheel.PyNeg()
    m = g.add("measured", "abir-source", content_ref="sha:abc", summary="21x2500")
    gen = g.add("generated", "gan", parents=[m], summary="synthetic")

    # measured() on the measured node works and reports evidence.
    assert g.measured(m)["class"] == "measured"
    assert g.is_evidence(m) is True

    # generated() works; the node is NOT evidence.
    assert g.generated(gen)["class"] == "generated"
    assert g.is_evidence(gen) is False

    # The boundary: measured() on a Generated node must raise (invariant #2).
    with pytest.raises(ValueError):
        g.measured(gen)
    # ...and estimated() on it too — only the true class views.
    with pytest.raises(ValueError):
        g.estimated(gen)


def test_estimated_carries_uncertainty_and_is_not_evidence(rust_wheel):
    g = rust_wheel.PyNeg()
    m = g.add("measured", "abir-source", content_ref="sha:w0")
    e = g.add(
        "estimated",
        "lmq@e1",
        parents=[m],
        summary="R=0.63",
        uncertainty_metric="prd",
        uncertainty_value=18.4,
    )
    node = g.estimated(e)
    assert node["uncertainty_value"] == 18.4
    assert g.is_evidence(e) is False


def test_verify_and_content_address_round_trip(rust_wheel):
    g = rust_wheel.PyNeg()
    m = g.add("measured", "abir-source", content_ref="sha:w0")
    g.add("estimated", "lmq@e1", parents=[m], uncertainty_metric="prd", uncertainty_value=1.0)
    g.materialize_provenance_edges()
    g.verify()  # must not raise

    j = g.to_json()
    g2 = rust_wheel.PyNeg.from_json(j)
    g2.verify()
    # Content address is insertion-order-independent + stable across a round-trip.
    assert g.content_address() == g2.content_address()


def test_fractional_uncertainty_round_trips_and_verifies(rust_wheel):
    # Regression: a reloaded graph must VERIFY (not just have an equal content
    # address) even when an uncertainty value is an f64 whose JSON number
    # serialize/parse isn't bit-exact (e.g. 100.54785919189453). The content
    # address must survive to_json -> from_json -> verify.
    for v in (100.54785919189453, 18.4, 1.0 / 3.0):
        g = rust_wheel.PyNeg()
        m = g.add("measured", "abir-source", content_ref="sha256:aa")
        g.add(
            "estimated", "lqs@openecs", content_ref="sha256:bb", parents=[m],
            uncertainty_metric="prd_pct", uncertainty_value=v,
        )
        g.materialize_provenance_edges()
        back = rust_wheel.PyNeg.from_json(g.to_json())
        back.verify()  # must not raise
        assert g.content_address() == back.content_address()


def test_verify_rejects_non_finite_uncertainty(rust_wheel):
    g = rust_wheel.PyNeg()
    g.add("estimated", "lmq", uncertainty_metric="prd", uncertainty_value=float("nan"))
    with pytest.raises(ValueError):
        g.verify()


def test_unknown_class_and_missing_node_raise(rust_wheel):
    g = rust_wheel.PyNeg()
    with pytest.raises(ValueError):
        g.add("not-a-class", "x")
    with pytest.raises(KeyError):
        g.measured("no-such-id")
