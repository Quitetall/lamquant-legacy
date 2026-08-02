//! Strict-typed pipeline trait for codec / container / sink stages.
//!
//! Each [`Stage`] takes exactly one input type and produces exactly one
//! output type. Misuse (chaining incompatible stages, dropping a
//! required step) fails at compile time — Bible R6 (strict types at
//! boundaries), R8 (type invariants as passive harness), R26
//! (compile-error > runtime-error).
//!
//! Composition via [`StageExt::then`]: the output type of stage A must
//! exactly match the input type of stage B, and both must agree on a
//! single error type. The Rust trait system enforces this — there is
//! no runtime check, no string-typed schema, no graph validator. If
//! it builds, the pipe is well-typed.
//!
//! Stages are stateful so that config (compress mode, sample rate,
//! window size, …) lives with the transformer and the per-record
//! `process` call carries data only. This is the "one datatype in,
//! one datatype out" discipline the codec relies on for FDA-style
//! traceability — every transformation has a single, named, audited
//! input/output contract.
//!
//! Bible R1 (Unix philosophy): each stage does ONE thing. Composition
//! handles the rest.
//!
//! See [`crate::pass`] for the `Reversible`/`Lossy` tagging layer built on
//! top of `Stage` (ADR 0069 Pillar 3): a `Pass` is a `Stage` additionally
//! tagged with its reversibility, and LML's reversible-only builder
//! statically refuses to accept a `Lossy` one.

use crate::error::LmlResult;

/// One transformation step in a typed pipeline.
///
/// Implementations:
///   - own their configuration at construction time (no per-call
///     config args; the data type tells the whole story)
///   - implement exactly ONE (Input, Output, Error) triple — this is
///     enforced by associated types rather than generic params, so
///     you literally cannot write a Stage that takes two different
///     input types
///   - return `LmlResult<Output>` to keep error handling uniform
///     across stage boundaries (lets `then(...)` chain without type
///     gymnastics)
pub trait Stage {
    /// The single input type. Anything that the stage consumes
    /// (signal frames, container bytes, archive entries) belongs
    /// here. Per-call ergonomics: methods take `Self::Input` by
    /// value, so non-`Clone` types pass cleanly through chains.
    type Input;
    /// The single output type. The next stage's `Input` must match
    /// this exactly. Tying these via associated types is what gives
    /// us compile-time pipe validation.
    type Output;
    /// Transform one input into one output. Returning Err short-
    /// circuits the chain via the `?` operator inside `Chain::process`.
    fn process(&mut self, input: Self::Input) -> LmlResult<Self::Output>;
}

/// Composed pipeline of two stages. Built via [`StageExt::then`];
/// the public surface should never name `Chain` directly — let type
/// inference handle the wiring.
#[derive(Debug)]
pub struct Chain<A, B> {
    first: A,
    second: B,
}

impl<A, B> Stage for Chain<A, B>
where
    A: Stage,
    B: Stage<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    fn process(&mut self, input: A::Input) -> LmlResult<B::Output> {
        let mid = self.first.process(input)?;
        self.second.process(mid)
    }
}

/// Combinator extension trait — every `Stage` automatically gains
/// `.then(next)` so call sites read in source order:
///
/// ```ignore
/// reader.then(decompressor).then(exporter).process(())?;
/// ```
///
/// `next.Input == self.Output` is checked at compile time via the
/// where clause.
pub trait StageExt: Stage + Sized {
    fn then<B>(self, next: B) -> Chain<Self, B>
    where
        B: Stage<Input = Self::Output>,
    {
        Chain {
            first: self,
            second: next,
        }
    }
}

impl<T: Stage> StageExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity-by-cast stage used to demonstrate the trait shape.
    /// Real stages live in lml / container / source.
    struct StringifyU32;
    impl Stage for StringifyU32 {
        type Input = u32;
        type Output = String;
        fn process(&mut self, input: u32) -> LmlResult<String> {
            Ok(input.to_string())
        }
    }

    struct ParseU32;
    impl Stage for ParseU32 {
        type Input = String;
        type Output = u32;
        fn process(&mut self, input: String) -> LmlResult<u32> {
            input
                .parse()
                .map_err(|e| crate::error::LmlError::InvalidHeader(format!("parse: {e}")))
        }
    }

    /// Stage with config in fields; per-call argument is data only.
    struct AddOffset {
        offset: u32,
    }
    impl Stage for AddOffset {
        type Input = u32;
        type Output = u32;
        fn process(&mut self, input: u32) -> LmlResult<u32> {
            Ok(input + self.offset)
        }
    }

    #[test]
    fn single_stage_runs() {
        let mut s = StringifyU32;
        assert_eq!(s.process(42).unwrap(), "42");
    }

    #[test]
    fn config_in_stage_data_in_call() {
        // "One datatype in" → u32; "one datatype out" → u32. The
        // `offset: 10` config travels with the stage, not the data.
        let mut s = AddOffset { offset: 10 };
        assert_eq!(s.process(7).unwrap(), 17);
    }

    #[test]
    fn then_composes_round_trip() {
        let mut chain = StringifyU32.then(ParseU32);
        // u32 → String → u32
        assert_eq!(chain.process(123).unwrap(), 123);
    }

    #[test]
    fn three_stage_chain() {
        let mut chain = AddOffset { offset: 5 }
            .then(StringifyU32)
            .then(ParseU32)
            .then(AddOffset { offset: 1 });
        assert_eq!(chain.process(10).unwrap(), 16); // (10+5) → "15" → 15 → 16
    }

    #[test]
    fn err_short_circuits() {
        struct AlwaysErr;
        impl Stage for AlwaysErr {
            type Input = u32;
            type Output = u32;
            fn process(&mut self, _: u32) -> LmlResult<u32> {
                Err(crate::error::LmlError::InvalidHeader("nope".into()))
            }
        }
        let counter = std::rc::Rc::new(std::cell::Cell::new(0u32));
        struct Counted(std::rc::Rc<std::cell::Cell<u32>>);
        impl Stage for Counted {
            type Input = u32;
            type Output = u32;
            fn process(&mut self, input: u32) -> LmlResult<u32> {
                self.0.set(self.0.get() + 1);
                Ok(input)
            }
        }
        let mut chain = AlwaysErr.then(Counted(counter.clone()));
        assert!(chain.process(7).is_err());
        assert_eq!(
            counter.get(),
            0,
            "downstream stage must NOT run after upstream Err"
        );
    }

    // Compile-time check: incompatible types refuse to compose.
    // Uncommenting this fails the build, which is the point — the
    // type system rejects pipe misuse before runtime ever sees it.
    //
    //     fn _wont_compile() {
    //         let _ = StringifyU32.then(StringifyU32);
    //         // Error: Stage<Input=String> required but found Stage<Input=u32>
    //     }
}
