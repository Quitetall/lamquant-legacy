"""Coverage tests for ``lamquant_codec.registry``.

Pins the public registry contract:
  - register_<kind> + get_<kind> round-trip a sentinel class
  - list_<kind> includes registered names
  - get_<kind> on an unknown name raises KeyError with a helpful message
  - get_<kind> instantiates (calls) the registered class
  - register_metric stores a callable (functions, not classes)
"""
from __future__ import annotations

import pytest

from lamquant_codec import registry
from lamquant_codec.registry import (
    DECODERS,
    ENCODERS,
    ENTROPY_CODERS,
    METRICS,
    PREPROCESSORS,
    get_decoder,
    get_encoder,
    get_entropy_coder,
    list_decoders,
    list_encoders,
    list_entropy_coders,
    list_metrics,
    register_decoder,
    register_encoder,
    register_entropy_coder,
    register_metric,
    register_preprocessor,
)


def _save_registries():
    return {
        'encoders': dict(ENCODERS),
        'decoders': dict(DECODERS),
        'entropy_coders': dict(ENTROPY_CODERS),
        'metrics': dict(METRICS),
        'preprocessors': dict(PREPROCESSORS),
    }


def _restore_registries(saved):
    ENCODERS.clear(); ENCODERS.update(saved['encoders'])
    DECODERS.clear(); DECODERS.update(saved['decoders'])
    ENTROPY_CODERS.clear(); ENTROPY_CODERS.update(saved['entropy_coders'])
    METRICS.clear(); METRICS.update(saved['metrics'])
    PREPROCESSORS.clear(); PREPROCESSORS.update(saved['preprocessors'])


@pytest.fixture(autouse=True)
def isolate_registries():
    """Snapshot + restore so tests don't pollute the global plugin maps."""
    saved = _save_registries()
    try:
        yield
    finally:
        _restore_registries(saved)


# ============================================================
# Encoder
# ============================================================


class TestRegisterEncoder:
    def test_decorator_returns_class(self) -> None:
        @register_encoder('_test_enc')
        class Dummy:
            pass
        assert Dummy is not None
        assert '_test_enc' in ENCODERS

    def test_get_returns_instance(self) -> None:
        @register_encoder('_test_enc_inst')
        class Dummy:
            kind = 'enc'
        obj = get_encoder('_test_enc_inst')
        assert isinstance(obj, Dummy)
        assert obj.kind == 'enc'

    def test_get_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match='Unknown encoder'):
            get_encoder('does_not_exist_xyz')

    def test_list_includes_registered(self) -> None:
        @register_encoder('_test_enc_listed')
        class Dummy: pass
        assert '_test_enc_listed' in list_encoders()


# ============================================================
# Decoder
# ============================================================


class TestRegisterDecoder:
    def test_register_and_lookup(self) -> None:
        @register_decoder('_test_dec')
        class Dummy: pass
        obj = get_decoder('_test_dec')
        assert isinstance(obj, Dummy)

    def test_get_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match='Unknown decoder'):
            get_decoder('does_not_exist_xyz')

    def test_list_decoders_returns_list(self) -> None:
        @register_decoder('_test_dec_listed')
        class Dummy: pass
        result = list_decoders()
        assert isinstance(result, list)
        assert '_test_dec_listed' in result


# ============================================================
# Entropy coder
# ============================================================


class TestRegisterEntropyCoder:
    def test_register_and_lookup(self) -> None:
        @register_entropy_coder('_test_ent')
        class Dummy: pass
        obj = get_entropy_coder('_test_ent')
        assert isinstance(obj, Dummy)

    def test_get_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match='Unknown entropy coder'):
            get_entropy_coder('does_not_exist_xyz')

    def test_list_entropy_coders_returns_list(self) -> None:
        @register_entropy_coder('_test_ent_listed')
        class Dummy: pass
        assert '_test_ent_listed' in list_entropy_coders()


# ============================================================
# Metric (function, not class)
# ============================================================


class TestRegisterMetric:
    def test_register_function_callable(self) -> None:
        @register_metric('_test_metric')
        def my_metric(x, y):
            return 0.0
        assert '_test_metric' in METRICS
        # Stored object is the function itself.
        assert METRICS['_test_metric'] is my_metric
        # And it's still callable directly.
        assert callable(METRICS['_test_metric'])

    def test_list_metrics(self) -> None:
        @register_metric('_test_metric_listed')
        def fn(x, y): return 0.0
        assert '_test_metric_listed' in list_metrics()


# ============================================================
# Preprocessor
# ============================================================


class TestRegisterPreprocessor:
    def test_register_stores_class(self) -> None:
        @register_preprocessor('_test_prep')
        class Dummy: pass
        assert '_test_prep' in PREPROCESSORS
        assert PREPROCESSORS['_test_prep'] is Dummy
