# Retired Python codec — test and tool harness

What the main tree still held of the suite that exercised the duplicate
pure-Python codec, after subtracting everything `support/` already preserved.

**Read `support/` first.** Package 32 sequestered most of this suite there
already. This directory holds only the 58 files that `support/` did NOT
already have byte-identically:

  * **43** were never sequestered at all.
  * **15** existed in `support/` but had DIVERGED -- the main tree's copy had
    moved on. Those are kept here under their original paths; `support/` keeps
    the older copy. Neither is authoritative over the other, and a diff between
    them is the honest record of what changed after the first sequester.

The other 57 were byte-identical to their `support/` copies and are not
duplicated here.

**These were already dead when they moved.** Package 32 retired
`lamquant_codec` from the main tree, and every file here imports it, so the
whole set had been failing at collection (59 collection errors) — they were
protecting nothing where they sat. Moving them changes no behaviour; it
records the fact.

Paths under this directory mirror their original locations exactly
(`tests/...`, `tools/scripts/...`), so the move is reversible and any file's
provenance is readable from its path.

## Not all of this is equally dead

Two groups, and the difference matters if anyone revives this:

- **Has a shipping counterpart.** `lossless`, `lma`, `ops.golomb`, `ops.lpc`,
  `ops.rans`, `ops.lifting`, `edf_to_lml` — all of these exist in the Rust
  binding today (`lml_compress`, `container_read`, `golomb_encode_dense`,
  `rans_encode`, `edf_read_digital`). These tests cover DSP and wire behaviour
  that is still shipped, so porting them protects something real.
  `tools/scripts/_rust_codec.py` in the main tree shows the pattern.
- **No counterpart.** `models.encoder`, `models.blocks`, `cli`, `training`,
  the bare package — Python-only internals with nothing equivalent shipping.
  Rewriting these would be inventing a new suite, not porting one.

Nothing is deleted. Retire by sequester.
