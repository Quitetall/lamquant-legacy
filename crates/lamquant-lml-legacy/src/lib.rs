//! LamQuant LML — the **LEGACY v1 wire** (ADR 0069 L3/L4).
//!
//! Owns the structurally-suboptimal old container/wire scaffolding outside the
//! current LamQuant product. Two features:
//!   * `legacy-decode` — the **FROZEN v1 reader**; **DEFAULT ON forever** (old
//!     `.lml`/`.lma` must decode). Host/std. On-MCU v1 back-compat is the separate
//!     kernel packet decoder + `SAW_LEGACY_CRC`, which stays in `lamquant-lml-mcu`.
//!   * `legacy-encode` — the retiring v1 writer; the differential **ORACLE**,
//!     demoted out of the default profile at cutover.
//!
//! The kernels (lifting / lpc / golomb / rans / crc) and their hardening are NOT
//! legacy — they stay in `lamquant-lml-mcu` and are *called* from here. Only the
//! structurally-suboptimal scaffolding (the 32-byte-header + JSON-metadata + window
//! -index wire, the `Signal=Vec<Vec<i64>>` currency, the metadata builders) lives
//! in this crate.
//!
//! **L3: empty scaffold** — proves the crate + workspace + feature graph compile.
//! **L4** moves `container.rs` / `offset_table.rs` / `stream.rs` in and splits the
//! read half (`legacy-decode`, frozen) from the write half (`legacy-encode`, oracle),
//! re-exported at the stable `lamquant_core::container` path so every call site + the
//! S1 golden + `legacy_crc_decode` + the L1 oracle stay byte-identical by construction.

// The v1 wire modules (ADR 0069 L4), relocated verbatim from `lamquant-lossless`.
// For this first cut all three ride under `legacy-decode` (default ON), so both the
// frozen reader and the oracle writer are available; the finer read/`legacy-encode`
// split is a tracked follow-up. Their `crate::{error,lml,lpc,deployment,crc32}` refs
// now resolve against `lamquant-lml-mcu`, and `crate::{backend,compress_with_mode_parallel}`
// against `lamquant-lml-desktop`.
#[cfg(feature = "legacy-decode")]
pub mod container;
#[cfg(feature = "legacy-decode")]
pub mod offset_table;
#[cfg(feature = "legacy-decode")]
pub mod stream;

// The RETIRED training tensor packs (ADR 0143 / 0144). LQTP1 stays in the codec
// as the live pack; LQTP2 (`LQT2`) and LQTP3 (`LQT3`) never reached main and are
// frozen here so old snapshots decode forever and the legacy adapter can open
// them. `bcs.training.lqtp.v1+` supersedes both; there is no standalone LQTP4.
#[cfg(feature = "legacy-tensor-pack")]
pub mod tensor_pack_v2;
#[cfg(feature = "legacy-tensor-pack")]
pub mod tensor_pack_v3;
