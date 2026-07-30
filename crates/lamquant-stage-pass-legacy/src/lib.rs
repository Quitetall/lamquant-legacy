//! Retired LamQuant codec composition API.
//!
//! Main LamQuant execution uses ABIR Nodes, kernels, compiled plans, and
//! receipts. This crate preserves the final `Stage`/`Pass` API from the exact
//! source revision named by [`SOURCE_REVISION`] for isolated compatibility
//! tooling only.

/// Immutable LamQuant Lossless revision that owns these retired definitions.
pub const SOURCE_REVISION: &str = "db7ff36aff529886195e067ea9628d3e7a08cd84";

pub use lamquant_lml_archive::{codec_stages, pass, pipeline, pipeline_dsl};

#[cfg(test)]
mod tests {
    use super::pipeline::{Stage, StageExt};
    use super::{codec_stages, pass, pipeline_dsl, SOURCE_REVISION};

    struct AddOne;

    impl Stage for AddOne {
        type Input = u32;
        type Output = u32;

        fn process(
            &mut self,
            input: Self::Input,
        ) -> lamquant_lml_archive::error::LmlResult<Self::Output> {
            Ok(input + 1)
        }
    }

    #[test]
    fn exact_source_revision_and_composition_surface_remain_available() {
        assert_eq!(SOURCE_REVISION, "db7ff36aff529886195e067ea9628d3e7a08cd84");
        let mut chain = AddOne.then(AddOne);
        assert_eq!(chain.process(40).unwrap(), 42);
    }

    #[test]
    fn every_retired_module_is_reexported() {
        let _ = core::mem::size_of::<pass::PassRegistry>();
        let _ = core::mem::size_of::<pipeline_dsl::PipelineSpec>();
        let _ = core::mem::size_of::<codec_stages::DecompressStage>();
    }
}
