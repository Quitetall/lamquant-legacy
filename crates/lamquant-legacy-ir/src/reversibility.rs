//! Reversibility markers — the type-level tag distinguishing a transform that can
//! be inverted bit-exactly from one that may discard information (ADR 0069 Pillar
//! 3, relocated here by ADR 0074).
//!
//! These are pure zero-sized types + a const-bool trait, `no_std`-clean, so they
//! remain in this retired compatibility crate alongside the modality markers — the two
//! typestate families that make LamQuant's codecs safe by construction. The
//! `Pass`/`LmlPipeline` machinery that *gates* on them stays host-side in
//! `lamquant-lossless` (it composes the host `Stage` trait); this crate owns only
//! the vocabulary, so any codec tier (including a future no_std pass layer) can
//! name `Reversible`/`Lossy` without pulling in `std`.

/// Marker trait distinguishing whether a pass tagged with it may discard
/// information. Implemented only by [`Reversible`] and [`Lossy`] — those two are
/// the entire universe of tags; there is no third option and no way to fake
/// reversibility (the marker is a distinct zero-sized type, not a runtime bool a
/// pass impl could get wrong).
pub trait Reversibility {
    /// `true` if a pass tagged with this marker may throw information away.
    /// `Reversible::LOSSY == false`, `Lossy::LOSSY == true`. This is the single
    /// boolean a dyn-erased pass surfaces to runtime callers.
    const LOSSY: bool;
}

/// Tag: a transform whose `process` is invertible bit-exactly via its
/// `unprocess`. The only marker the LML/lossless pipeline builder accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reversible;

impl Reversibility for Reversible {
    const LOSSY: bool = false;
}

/// Tag: a transform that may discard information. LMQ (neural/lossy) may run
/// these; LML refuses them both statically (the pipeline builder's trait bound)
/// and dynamically (the runtime `bool` gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lossy;

impl Reversibility for Lossy {
    const LOSSY: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_zero_sized_and_carry_the_right_const() {
        assert_eq!(core::mem::size_of::<Reversible>(), 0);
        assert_eq!(core::mem::size_of::<Lossy>(), 0);
        const {
            assert!(!<Reversible as Reversibility>::LOSSY);
            assert!(<Lossy as Reversibility>::LOSSY);
        }
    }
}
