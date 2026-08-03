"""Experimental / unwired codec components.

These modules contain research-stage code that is NOT part of the production
encode/decode pipeline. They live here (rather than at the top level) to
keep the main `lamquant_codec` surface unambiguous about what ships.

Each module documents its own status. Wire one up by:
  1. Confirm the design via `python -m lamquant_codec.experimental.<name>`
  2. Add a public entry point in `lamquant_codec/<name>.py`
  3. Register via the codec plugin registry (see lamquant_codec.registry)
  4. Add tests under tests/test_<name>.py

Current contents:

- learned_entropy.py — SNNEntropyHead + ConditionalEntropyModel
  Drop-in replacements for the static rANS frequency table. ~10-40%
  bitrate reduction projected. Not yet wired into compress.py.

- token_compression.py — token RLE + hexagonal Q2D2 + savings estimators
  Pre-rANS optimizations for quiescent EEG segments. Not yet on the
  hot path.
"""
__all__ = []
