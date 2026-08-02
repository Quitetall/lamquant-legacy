//! Reversible/Lossy `Pass` framework over [`Stage`] (ADR 0069 Pillar 3).
//!
//! A [`Pass`] is a [`Stage`] additionally tagged, at the type level, with
//! whether running it can discard information:
//!
//!   - [`Reversible`] — `process` has an inverse ([`ReversiblePass::unprocess`])
//!     such that `unprocess(process(x)) == x`. Integer transforms (delta
//!     coding, the reversible cross-channel KLT, ...) live here.
//!   - [`Lossy`] — `process` may throw information away; there is no
//!     `unprocess`. Neural/LMQ-side transforms live here.
//!
//! **LML statically refuses Lossy.** The lossless codec never runs a
//! `Lossy` pass — not "we promise not to call it", but "the type does not
//! fit through the door": [`LmlPipeline`] only accepts `S: Pass<Rev =
//! Reversible>`, so wiring a `Lossy` pass into it is rejected by the
//! compiler before the program exists. See the `compile_fail` doctest on
//! [`LmlPipeline`] for the actual rejection.
//!
//! [`DynPass`] + [`PassRegistry`] add a *dyn*-erased layer for the S5 DSL
//! (pass-pipeline text naming a pass at runtime, where the static
//! `Pass::Rev` type has already been erased). There, the same refusal is
//! re-checked dynamically via [`run_in_lml`] — a `bool` gate instead of a
//! trait bound, because that's all a runtime name resolution has left to
//! check.
//!
//! Bible R6/R8 (strict types, compile-time invariants), R26 (compile-error
//! over runtime-error): the compile-time refusal is the primary
//! mechanism; the runtime refusal is the necessary fallback for the one
//! path (dynamic pass selection) where compile-time enforcement cannot
//! reach.

use crate::error::{LmlError, LmlResult};
use crate::pipeline::{Chain, Stage, StageExt};
use std::collections::HashMap;
// `Box`/`String`/`Vec` come from the std prelude (this module is gated
// behind `archive`, which implies `std`) — no explicit import needed.

// ─── Reversibility markers ───

/// Sealed type-level classification for processing passes.
pub trait Reversibility: private::Sealed {
    /// Whether a pass is contractually reversible.
    const REVERSIBLE: bool;
}

/// Marker for passes with a tested inverse.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reversible;

/// Marker for passes which may discard information.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Lossy;

impl Reversibility for Reversible {
    const REVERSIBLE: bool = true;
}

impl Reversibility for Lossy {
    const REVERSIBLE: bool = false;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::Reversible {}
    impl Sealed for super::Lossy {}
}

// ─── Pass ───────────────────────────────────────────────────────────────

/// A [`Stage`] additionally tagged with its reversibility and a stable
/// name. The name is what a dyn-erased [`PassRegistry`] keys on and what
/// the S5 pass-pipeline DSL will reference textually; the `Rev` tag is
/// what [`LmlPipeline`]/[`run_in_lml`] gate on.
pub trait Pass: Stage {
    /// [`Reversible`] or [`Lossy`] — see the module docs.
    type Rev: Reversibility;
    /// Stable, DSL-facing name (registered in [`PassRegistry`] under this
    /// same string by convention, though the registry does not enforce
    /// the match — see [`DynPass::name`]).
    const NAME: &'static str;
}

/// Composition preserves reversibility: a [`Chain`] of two `Reversible`
/// passes is itself a `Reversible` pass (composing two invertible maps
/// gives an invertible map). This is what lets [`LmlPipeline::then`] be
/// called more than once — `Chain<S, B>` needs to satisfy `Pass<Rev =
/// Reversible>` again to become the seed of the next `.then()` (or the
/// target of `.process()`).
///
/// There is deliberately no matching `impl Pass for Chain<A, B>` when
/// either side is `Lossy` — that absence, not an explicit rejection, is
/// what makes chaining a `Lossy` pass into an LML pipeline fail to
/// compile: the trait bound `Chain<A, B>: Pass<Rev = Reversible>` simply
/// has no impl to resolve to.
impl<A, B> Pass for Chain<A, B>
where
    A: Pass<Rev = Reversible>,
    B: Pass<Rev = Reversible, Input = A::Output>,
{
    type Rev = Reversible;
    const NAME: &'static str = "chain";
}

/// A [`Pass`] tagged [`Reversible`] additionally provides the inverse of
/// `process`. The contract: for any `x: Self::Input` accepted by
/// `process`, `unprocess(process(x)?)? == x` (process then unprocess is
/// the identity) — proven per concrete pass by a round-trip test (see
/// `reversible_pass_round_trips` below).
pub trait ReversiblePass: Pass<Rev = Reversible> {
    /// Invert one `process` call. Takes `&mut self` (not `&self`) for the
    /// same reason `Stage::process` does — a pass may carry adaptive
    /// state (e.g. an RLS predictor) that both directions need to keep in
    /// sync.
    fn unprocess(&mut self, out: Self::Output) -> LmlResult<Self::Input>;
}

// ─── LML: the reversible-only builder (compile-time refusal) ───────────

/// The LML (lossless) pipeline builder. This is the entire mechanism
/// behind "LML statically refuses Lossy": every constructor/combinator
/// here is bounded `where P: Pass<Rev = Reversible>`, so a `Lossy` pass
/// simply does not implement the trait the builder requires — the
/// program that tries to wire one in does not compile.
///
/// ```
/// use lamquant_stage_pass_legacy::pass::{LmlPipeline, Pass, Reversible, ReversiblePass};
/// use lamquant_stage_pass_legacy::pipeline::Stage;
/// use lamquant_stage_pass_legacy::error::LmlResult;
///
/// struct Increment(u8);
/// impl Stage for Increment {
///     type Input = u8;
///     type Output = u8;
///     fn process(&mut self, input: u8) -> LmlResult<u8> {
///         Ok(input.wrapping_add(self.0))
///     }
/// }
/// impl Pass for Increment {
///     type Rev = Reversible;
///     const NAME: &'static str = "increment";
/// }
/// impl ReversiblePass for Increment {
///     fn unprocess(&mut self, out: u8) -> LmlResult<u8> {
///         Ok(out.wrapping_sub(self.0))
///     }
/// }
///
/// let mut pipeline = LmlPipeline::start(Increment(5)).then(Increment(2));
/// assert_eq!(pipeline.process(10).unwrap(), 17);
/// ```
///
/// Wiring a [`Lossy`] pass into the same builder is a **compile** error,
/// not a documentation promise:
///
/// ```compile_fail
/// use lamquant_stage_pass_legacy::pass::{LmlPipeline, Lossy, Pass};
/// use lamquant_stage_pass_legacy::pipeline::Stage;
/// use lamquant_stage_pass_legacy::error::LmlResult;
///
/// // Drops the low 4 bits of every byte — genuinely lossy, no inverse.
/// struct BitCrusher;
/// impl Stage for BitCrusher {
///     type Input = Vec<u8>;
///     type Output = Vec<u8>;
///     fn process(&mut self, input: Vec<u8>) -> LmlResult<Vec<u8>> {
///         Ok(input.into_iter().map(|b| b & 0xF0).collect())
///     }
/// }
/// impl Pass for BitCrusher {
///     type Rev = Lossy; // <- not Reversible
///     const NAME: &'static str = "bit_crusher";
/// }
///
/// // error[E0271]: type mismatch resolving `<BitCrusher as Pass>::Rev == Reversible`
/// // `LmlPipeline::start` requires `S: Pass<Rev = Reversible>`; BitCrusher's
/// // `Rev` is `Lossy`, so this refuses to compile — LML cannot even name a
/// // pipeline that starts with a Lossy pass, let alone run one.
/// let _pipeline = LmlPipeline::start(BitCrusher);
/// ```
#[derive(Debug, Clone)]
pub struct LmlPipeline<S> {
    stage: S,
}

impl<S> LmlPipeline<S>
where
    S: Pass<Rev = Reversible>,
{
    /// Seed a reversible-only pipeline. `S: Pass<Rev = Reversible>` is
    /// checked right here, at the call site — a `Lossy` pass cannot be
    /// the first stage of an LML pipeline.
    pub fn start(stage: S) -> Self {
        Self { stage }
    }

    /// Extend the pipeline with another reversible pass. The bound on
    /// `next` is the same as on `start`: `Lossy` is rejected at every
    /// link, not just the first.
    pub fn then<B>(self, next: B) -> LmlPipeline<Chain<S, B>>
    where
        B: Pass<Rev = Reversible, Input = S::Output>,
    {
        LmlPipeline {
            stage: StageExt::then(self.stage, next),
        }
    }

    /// Run the composed pipeline. Composition was already proven
    /// well-typed (and all-reversible) at build time; nothing left to
    /// check here.
    pub fn process(&mut self, input: S::Input) -> LmlResult<S::Output> {
        self.stage.process(input)
    }
}

// ─── Dyn-erased runtime layer (S5 DSL prep) ─────────────────────────────

/// Erased payload for the dyn-erased pass boundary. The DSL selects a
/// pass by *name* at runtime, at which point the static `Input`/`Output`
/// associated types can no longer drive dispatch — [`DynPass`] trades
/// compile-time pipe validation for one concrete runtime type (a byte
/// buffer, matching the codec's own container/entropy-coder boundaries,
/// which are already byte-shaped — see `codec_stages::EncodedContainer`)
/// plus the [`DynPass::lossy`] runtime gate.
pub type ErasedPayload = Vec<u8>;

/// Dyn-erased view of a [`Pass`], for runtime (DSL-driven) pass selection.
/// Prefer the statically-typed [`LmlPipeline`]/[`StageExt::then`] path
/// whenever the pipeline shape is known at compile time; `DynPass` exists
/// for the minority of call sites that pick a pass by name out of a
/// registry or a pass-pipeline config string.
pub trait DynPass {
    /// The pass's name (by convention, [`Pass::NAME`] of the wrapped
    /// type).
    fn name(&self) -> &str;
    /// `true` if this pass may discard information ([`Lossy`]). The LML
    /// runtime path ([`run_in_lml`]) checks this and refuses to run
    /// *before* `run` is ever called.
    fn lossy(&self) -> bool;
    /// Execute the pass on an erased byte payload.
    fn run(&mut self, input: ErasedPayload) -> LmlResult<ErasedPayload>;
}

/// Adapter lifting a concrete byte-shaped [`Pass`] into the dyn-erased
/// [`DynPass`] surface. `P` must already speak [`ErasedPayload`] in and
/// out — erasure here forgets the *type*, never the *shape*; a `Pass`
/// with a richer `Input`/`Output` needs its own byte (de)serialization
/// before it can be registered dynamically.
pub struct DynPassAdapter<P>(pub P);

impl<P> DynPass for DynPassAdapter<P>
where
    P: Pass<Input = ErasedPayload, Output = ErasedPayload>,
{
    fn name(&self) -> &str {
        P::NAME
    }

    fn lossy(&self) -> bool {
        !<P::Rev as Reversibility>::REVERSIBLE
    }

    fn run(&mut self, input: ErasedPayload) -> LmlResult<ErasedPayload> {
        self.0.process(input)
    }
}

/// Name → boxed-constructor registry for dyn-erased passes (S5 DSL prep):
/// the pass-pipeline DSL resolves a pass by name at parse time, then asks
/// the registry to build a fresh [`DynPass`] instance to actually run.
///
/// Constructors and the passes they build are bounded `Send + Sync` even
/// though nothing here requires it *yet* — a registry is exactly the kind
/// of shared, long-lived object that ends up behind an `Arc`/`Mutex` (a
/// TUI panel, a DSL config service) the moment it has more than one
/// caller, and retrofitting the bound later is a breaking API change for
/// every registered constructor. Cheaper to require it from the start.
#[allow(clippy::type_complexity)]
#[derive(Default)]
pub struct PassRegistry {
    ctors: HashMap<String, Box<dyn Fn() -> Box<dyn DynPass + Send + Sync> + Send + Sync>>,
    /// Param-aware constructors (ADR 0074 M6): receive the DSL's parsed
    /// `key=value` params and may fail on a bad value.
    param_ctors: HashMap<
        String,
        Box<dyn Fn(&[(String, String)]) -> LmlResult<Box<dyn DynPass + Send + Sync>> + Send + Sync>,
    >,
}

impl PassRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a named constructor. Re-registering an existing name
    /// overwrites the previous constructor — plain `HashMap::insert`
    /// semantics, no silent no-op.
    pub fn register<F>(&mut self, name: impl Into<String>, ctor: F)
    where
        F: Fn() -> Box<dyn DynPass + Send + Sync> + Send + Sync + 'static,
    {
        self.ctors.insert(name.into(), Box::new(ctor));
    }

    /// Register a named constructor that BINDS the DSL's parsed `key=value`
    /// params (ADR 0074 M6 — closes the G3 gap, where params were validated
    /// syntactically but then discarded). The constructor is fallible: a bad or
    /// out-of-range param value is a build error, never a silent default.
    pub fn register_with<F>(&mut self, name: impl Into<String>, ctor: F)
    where
        F: Fn(&[(String, String)]) -> LmlResult<Box<dyn DynPass + Send + Sync>>
            + Send
            + Sync
            + 'static,
    {
        self.param_ctors.insert(name.into(), Box::new(ctor));
    }

    /// `true` if a (param-less or param-aware) constructor is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.ctors.contains_key(name) || self.param_ctors.contains_key(name)
    }

    /// Build a pass by name, BINDING the parsed params: a param-aware constructor
    /// ([`register_with`](Self::register_with)) receives them; a param-less one
    /// ([`register`](Self::register)) ignores them (back-compat). Unknown → error.
    pub fn build_with(
        &self,
        name: &str,
        params: &[(String, String)],
    ) -> LmlResult<Box<dyn DynPass + Send + Sync>> {
        if let Some(ctor) = self.param_ctors.get(name) {
            ctor(params)
        } else if let Some(ctor) = self.ctors.get(name) {
            Ok(ctor())
        } else {
            Err(LmlError::InvalidHeader(std::format!(
                "PassRegistry: no pass registered under name {name:?}"
            )))
        }
    }

    /// Build a fresh dyn-erased pass instance by name.
    ///
    /// Unknown names use `LmlError::InvalidHeader` because this isolated
    /// legacy crate retains the frozen error algebra owned by
    /// `lamquant-lml-mcu`; adding a new variant would change that contract.
    pub fn build(&self, name: &str) -> LmlResult<Box<dyn DynPass + Send + Sync>> {
        match self.ctors.get(name) {
            Some(ctor) => Ok(ctor()),
            None => Err(LmlError::InvalidHeader(std::format!(
                "PassRegistry: no pass registered under name {name:?}"
            ))),
        }
    }
}

/// Runtime refusal gate for the LML (lossless) runtime path: run a
/// dyn-erased pass only if it reports `lossy() == false`. This is the
/// dynamic-dispatch counterpart to [`LmlPipeline`]'s compile-time
/// refusal — necessary because a pass name resolved from the DSL/config
/// at runtime has already erased the static `Pass::Rev` type, so the
/// check has to happen again, on the boolean, before `run` executes.
///
/// Refusal reuses `LmlError::InvalidHeader` for the same frozen-error-contract
/// reason as [`PassRegistry::build`] above.
pub fn run_in_lml(pass: &mut dyn DynPass, input: ErasedPayload) -> LmlResult<ErasedPayload> {
    if pass.lossy() {
        return Err(LmlError::InvalidHeader(std::format!(
            "LML (lossless) runtime refuses to run Lossy pass {:?} \
             (ADR 0069 Pillar 3: LML statically + dynamically refuses Lossy passes)",
            pass.name()
        )));
    }
    pass.run(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XOR-with-key byte cipher: its own inverse, byte-exact. Stands in
    /// for a real reversible transform (delta coding, the integer KLT,
    /// ...) for the purposes of this test.
    struct XorCipher {
        key: u8,
    }

    impl Stage for XorCipher {
        type Input = Vec<u8>;
        type Output = Vec<u8>;
        fn process(&mut self, input: Vec<u8>) -> LmlResult<Vec<u8>> {
            Ok(input.into_iter().map(|b| b ^ self.key).collect())
        }
    }

    impl Pass for XorCipher {
        type Rev = Reversible;
        const NAME: &'static str = "xor_cipher";
    }

    impl ReversiblePass for XorCipher {
        fn unprocess(&mut self, out: Vec<u8>) -> LmlResult<Vec<u8>> {
            // XOR is its own inverse: unprocess == process with the same key.
            self.process(out)
        }
    }

    /// Truncate-to-`keep`-bytes: genuinely lossy — the dropped tail is
    /// gone, there is no `unprocess`.
    struct TruncateLossy {
        keep: usize,
    }

    impl Stage for TruncateLossy {
        type Input = Vec<u8>;
        type Output = Vec<u8>;
        fn process(&mut self, input: Vec<u8>) -> LmlResult<Vec<u8>> {
            let mut v = input;
            v.truncate(self.keep);
            Ok(v)
        }
    }

    impl Pass for TruncateLossy {
        type Rev = Lossy;
        const NAME: &'static str = "truncate_lossy";
    }

    #[test]
    fn reversibility_lossy_const_matches_tag() {
        const {
            assert!(<Reversible as Reversibility>::REVERSIBLE);
            assert!(!<Lossy as Reversibility>::REVERSIBLE);
        }
    }

    #[test]
    fn reversible_pass_round_trips() {
        // process then unprocess == identity (the ReversiblePass contract).
        let mut p = XorCipher { key: 0x5A };
        let original = std::vec![1u8, 2, 3, 250, 0, 255];
        let encoded = p.process(original.clone()).unwrap();
        assert_ne!(
            encoded, original,
            "sanity: the transform actually changed the bytes"
        );
        let decoded = p.unprocess(encoded).unwrap();
        assert_eq!(decoded, original, "process then unprocess must be identity");
    }

    #[test]
    fn lml_pipeline_composes_reversible_passes() {
        let mut pipeline =
            LmlPipeline::start(XorCipher { key: 0x11 }).then(XorCipher { key: 0x22 });
        let out = pipeline.process(std::vec![9, 8, 7]).unwrap();
        assert_eq!(
            out,
            std::vec![9 ^ 0x11 ^ 0x22, 8 ^ 0x11 ^ 0x22, 7 ^ 0x11 ^ 0x22]
        );
    }

    #[test]
    fn dyn_pass_adapter_reports_name_and_lossiness() {
        let xor = DynPassAdapter(XorCipher { key: 3 });
        assert_eq!(xor.name(), "xor_cipher");
        assert!(!xor.lossy());

        let trunc = DynPassAdapter(TruncateLossy { keep: 2 });
        assert_eq!(trunc.name(), "truncate_lossy");
        assert!(trunc.lossy());
    }

    #[test]
    fn lml_runtime_refuses_lossy_dyn_pass() {
        let mut trunc = DynPassAdapter(TruncateLossy { keep: 2 });
        let err = run_in_lml(&mut trunc, std::vec![1, 2, 3, 4]).unwrap_err();
        match err {
            LmlError::InvalidHeader(msg) => {
                assert!(msg.contains("truncate_lossy"), "message: {msg}");
                assert!(msg.contains("refuses"), "message: {msg}");
            }
            other => panic!("expected InvalidHeader refusal, got {other:?}"),
        }
    }

    #[test]
    fn lml_runtime_runs_reversible_dyn_pass() {
        let mut xor = DynPassAdapter(XorCipher { key: 7 });
        let out = run_in_lml(&mut xor, std::vec![1, 2, 3]).unwrap();
        assert_eq!(out, std::vec![1 ^ 7, 2 ^ 7, 3 ^ 7]);
    }

    #[test]
    fn pass_registry_builds_by_name() {
        let mut reg = PassRegistry::new();
        reg.register("xor_cipher", || {
            Box::new(DynPassAdapter(XorCipher { key: 0xFF })) as Box<dyn DynPass + Send + Sync>
        });
        assert!(reg.contains("xor_cipher"));
        assert!(!reg.contains("nonexistent"));

        let built = reg.build("xor_cipher").unwrap();
        assert_eq!(built.name(), "xor_cipher");
        assert!(!built.lossy());
    }

    #[test]
    fn pass_registry_unknown_name_errors() {
        let reg = PassRegistry::new();
        assert!(reg.build("nonexistent").is_err());
    }

    #[test]
    fn pass_registry_round_trips_through_run_in_lml() {
        let mut reg = PassRegistry::new();
        reg.register("xor_cipher", || {
            Box::new(DynPassAdapter(XorCipher { key: 0x42 })) as Box<dyn DynPass + Send + Sync>
        });
        reg.register("truncate_lossy", || {
            Box::new(DynPassAdapter(TruncateLossy { keep: 1 })) as Box<dyn DynPass + Send + Sync>
        });

        let mut reversible = reg.build("xor_cipher").unwrap();
        assert!(run_in_lml(reversible.as_mut(), std::vec![1, 2]).is_ok());

        let mut lossy = reg.build("truncate_lossy").unwrap();
        assert!(run_in_lml(lossy.as_mut(), std::vec![1, 2]).is_err());
    }
}
