//! Textual pass-pipeline DSL over [`crate::pass`].
//!
//! A pipeline is an ordered list of named passes, one per line:
//!
//! ```text
//! ; reversible pipeline
//! delta_filter
//! xor_cipher key=66
//! ```
//!
//! Grammar (simple, panic-free on arbitrary/untrusted input):
//!   - a line whose trimmed content is empty, or starts with `;` or `#`,
//!     is a comment/blank line and is ignored;
//!   - every other line is `pass_name` followed by zero or more
//!     space-separated `key=value` params;
//!   - a param `value` is either a **bareword** (runs to the next
//!     whitespace, no escaping) or a Rust-`{:?}`-quoted string (`"..."`),
//!     unescaped with the exact same grammar as [`crate::ir::parse_quoted`]
//!     (`\" \\ \n \t \r \0 \u{HEX}`) — see [`parse_quoted_value`], which
//!     mirrors that function's algorithm so a value written the same way a
//!     `to_ir_text` string field is written parses identically.
//!
//! [`parse_pipeline`] turns DSL text into a [`PipelineSpec`] — pure syntax,
//! no registry lookup, so it never fails on an unknown pass name.
//! [`build_lml_pipeline`] resolves each step against a [`PassRegistry`],
//! reusing the *existing* [`crate::pass::run_in_lml`] / [`DynPass`]
//! machinery unchanged: the LML (lossless) path refuses any step whose
//! [`DynPass::lossy`] reports `true`, returning
//! [`PipelineDslError::LossyPassInLml`]. This is the **runtime**
//! counterpart to [`crate::pass::LmlPipeline`]'s compile-time refusal
//! (see that module's `compile_fail` doctest) — necessary here because the
//! DSL selects a pass by *name*, at runtime, out of a config string, so
//! the static `Pass::Rev` type has already been erased by the time a name
//! resolves to a [`DynPass`]. There is no `where P: Pass<Rev = Reversible>`
//! bound left to write; the `bool` gate is all a runtime name resolution
//! has to check against, and this module re-checks it before ever wiring
//! a step into the built pipeline.
//!
//! [`build_lml_pipeline`] resolves each step by name and passes its parsed
//! parameters to [`PassRegistry::build_with`]. Registrations created with
//! [`PassRegistry::register_with`] validate and bind those parameters;
//! backward-compatible registrations created with [`PassRegistry::register`]
//! are parameterless and ignore them. Syntax is validated in both cases.

use crate::pass::{run_in_lml, DynPass, ErasedPayload, PassRegistry};
use std::vec::Vec;

// ─── Pipeline spec (pure syntax) ────────────────────────────────────────

/// One line of a parsed pipeline: a pass name plus its `key=value` params,
/// in the order they appeared. Params are `(String, String)` pairs —
/// see the module docs for binding behavior of parameter-aware and
/// parameterless registrations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassStep {
    pub name: String,
    pub params: Vec<(String, String)>,
}

/// An ordered, parsed pipeline — the output of [`parse_pipeline`] and the
/// input to [`build_lml_pipeline`].
pub type PipelineSpec = Vec<PassStep>;

// ─── Errors ──────────────────────────────────────────────────────────────

/// Typed error for both DSL parsing ([`parse_pipeline`]) and pipeline
/// building ([`build_lml_pipeline`]). Panic-free on any input: every
/// grammar violation or resolution failure is one of these variants,
/// never an unwrap/index/slice panic (see `negative_battery` +
/// `truncation_sweep_never_panics` in the test module).
///
/// `line` fields are 0-based indices into the input text's lines
/// (`str::lines()` numbering), matching the convention already used by
/// `ir::IrParseError::{BadChannelLine,BadSidecarLine}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineDslError {
    /// A pass line's `key=value` token doesn't match that shape — no `=`
    /// before the next whitespace/end, or an empty key before the `=`.
    BadParam { line: usize },
    /// A quoted (`"..."`) param value's closing (unescaped) quote was
    /// never found before the line ended.
    UnterminatedString { line: usize },
    /// A `\` inside a quoted param value wasn't followed by one of
    /// `" \ n t r 0 u{HEX}` (or the `u{...}` body was empty, non-hex, or
    /// encoded a value outside the Unicode scalar range) — mirrors
    /// `ir::IrParseError::BadEscape`.
    BadEscape { line: usize },
    /// [`parse_pipeline`] succeeded, but [`build_lml_pipeline`] found no
    /// constructor registered under this step's name in the
    /// [`PassRegistry`] it was given.
    UnknownPass { name: String },
    /// [`build_lml_pipeline`] resolved a step whose [`DynPass::lossy`]
    /// reports `true` — the LML (lossless) build path refuses to wire a
    /// Lossy pass into the pipeline at all (ADR 0069 Pillar 3: refused
    /// both statically, via `LmlPipeline`'s trait bound, and here,
    /// dynamically, via this runtime check).
    LossyPassInLml { name: String },
    /// A param-aware constructor ([`PassRegistry::register_with`], ADR 0074 M6)
    /// rejected this step's bound params — e.g. an out-of-range or unparseable
    /// value. `detail` carries the constructor's message.
    BadParamValue { name: String, detail: String },
}

impl core::fmt::Display for PipelineDslError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadParam { line } => {
                write!(f, "malformed `key=value` param on pipeline line {line}")
            }
            Self::UnterminatedString { line } => {
                write!(f, "unterminated quoted param value on pipeline line {line}")
            }
            Self::BadEscape { line } => write!(
                f,
                "invalid escape sequence in quoted param value on pipeline line {line}"
            ),
            Self::UnknownPass { name } => {
                write!(f, "no pass registered under name {name:?}")
            }
            Self::LossyPassInLml { name } => write!(
                f,
                "LML (lossless) pipeline refuses Lossy pass {name:?} \
                 (ADR 0069 Pillar 3: LML statically + dynamically refuses Lossy passes)"
            ),
            Self::BadParamValue { name, detail } => {
                write!(f, "pass {name:?} rejected its params: {detail}")
            }
        }
    }
}

impl std::error::Error for PipelineDslError {}

// ─── Parsing (panic-free, no registry lookup) ───────────────────────────

/// Split `s` at its first whitespace char, returning `(before, from-the-
/// whitespace-on)`. `str::find` with a `char` pattern always returns a
/// byte offset on a char boundary, so this slice can never panic or split
/// a multi-byte UTF-8 sequence.
fn split_first_token(s: &str) -> (&str, &str) {
    match s.find(char::is_whitespace) {
        Some(idx) => (&s[..idx], &s[idx..]),
        None => (s, ""),
    }
}

/// Parse a Rust-`{:?}`-quoted param value. Precondition (checked by the
/// only caller, [`parse_param`]): `s` starts with `"`. Mirrors
/// `ir::parse_quoted`'s escape grammar (`" \ n t r 0 u{HEX}`) exactly, so
/// a value written the same way a `to_ir_text` string field is written
/// parses identically. Walks `s.chars()` (never raw byte indexing), so
/// multi-byte UTF-8 is never split; every malformed/truncated escape is a
/// typed [`PipelineDslError`], never a panic.
fn parse_quoted_value(s: &str, line: usize) -> Result<(String, &str), PipelineDslError> {
    let mut chars = s.chars();
    chars.next(); // the opening '"' — precondition guaranteed by the caller.
    let mut out = String::new();
    loop {
        match chars.next() {
            None => return Err(PipelineDslError::UnterminatedString { line }),
            Some('"') => return Ok((out, chars.as_str())),
            Some('\\') => match chars.next() {
                None => return Err(PipelineDslError::UnterminatedString { line }),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('0') => out.push('\0'),
                Some('u') => {
                    if chars.next() != Some('{') {
                        return Err(PipelineDslError::BadEscape { line });
                    }
                    let mut hex = String::new();
                    loop {
                        match chars.next() {
                            Some('}') => break,
                            Some(c) if c.is_ascii_hexdigit() && hex.len() < 6 => hex.push(c),
                            _ => return Err(PipelineDslError::BadEscape { line }),
                        }
                    }
                    if hex.is_empty() {
                        return Err(PipelineDslError::BadEscape { line });
                    }
                    let cp = u32::from_str_radix(&hex, 16)
                        .map_err(|_| PipelineDslError::BadEscape { line })?;
                    let ch = char::from_u32(cp).ok_or(PipelineDslError::BadEscape { line })?;
                    out.push(ch);
                }
                Some(_) => return Err(PipelineDslError::BadEscape { line }),
            },
            Some(c) => out.push(c),
        }
    }
}

/// Parse one `key=value` param off the front of `rest` (which the caller
/// guarantees is non-empty and has no leading whitespace), returning the
/// `(key, value)` pair and whatever text remains after it.
fn parse_param(rest: &str, line: usize) -> Result<((String, String), &str), PipelineDslError> {
    let bad = || PipelineDslError::BadParam { line };
    // The key runs up to the first '=' — but only if that '=' comes
    // before the next whitespace (a bare token with no '=' at all, or
    // whitespace before any '=', is malformed).
    let eq_idx = rest.find('=');
    let ws_idx = rest.find(char::is_whitespace);
    let key_end = match (eq_idx, ws_idx) {
        (Some(e), Some(w)) if e < w => e,
        (Some(e), None) => e,
        _ => return Err(bad()),
    };
    if key_end == 0 {
        return Err(bad()); // empty key, e.g. "=value"
    }
    let key = rest[..key_end].to_string();
    // '=' is a single ASCII byte, so key_end + 1 is always a char boundary.
    let after_eq = &rest[key_end + 1..];
    let (value, remainder) = if after_eq.starts_with('"') {
        parse_quoted_value(after_eq, line)?
    } else {
        match after_eq.find(char::is_whitespace) {
            Some(idx) => (after_eq[..idx].to_string(), &after_eq[idx..]),
            None => (after_eq.to_string(), ""),
        }
    };
    Ok(((key, value), remainder))
}

/// Parse one non-comment, non-blank, already-trimmed pipeline line into a
/// [`PassStep`].
fn parse_pass_line(trimmed: &str, line: usize) -> Result<PassStep, PipelineDslError> {
    let (name, mut rest) = split_first_token(trimmed);
    let mut params = Vec::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let (param, remainder) = parse_param(rest, line)?;
        params.push(param);
        rest = remainder;
    }
    Ok(PassStep {
        name: name.to_string(),
        params,
    })
}

/// Parse pass-pipeline DSL text into a [`PipelineSpec`]. Pure syntax —
/// does not consult a [`PassRegistry`], so an unknown pass name is not an
/// error here (that's [`build_lml_pipeline`]'s job, once a registry is in
/// hand). Panic-free on any input; see the module docs for the grammar
/// and the test module's `negative_battery` / `truncation_sweep_never_panics`
/// for the adversarial-input proof.
pub fn parse_pipeline(text: &str) -> Result<PipelineSpec, PipelineDslError> {
    let mut steps = Vec::new();
    for (line, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') || trimmed.starts_with('#') {
            continue;
        }
        steps.push(parse_pass_line(trimmed, line)?);
    }
    Ok(steps)
}

// ─── Building (registry resolution + the LML Lossy refusal) ────────────

/// A built, ready-to-run LML pipeline: an ordered chain of resolved
/// dyn-erased passes, each already proven `lossy() == false` by
/// [`build_lml_pipeline`]. [`LmlDslPipeline::run`] replays
/// [`crate::pass::run_in_lml`] across the chain (byte payload in, byte
/// payload out), re-checking the Lossy gate per step as
/// belt-and-suspenders with the build-time check.
pub struct LmlDslPipeline {
    steps: Vec<Box<dyn DynPass + Send + Sync>>,
}

impl core::fmt::Debug for LmlDslPipeline {
    /// `DynPass` is not `Debug` (it's a trait object over arbitrary pass
    /// state), so this prints the one thing that's always known: each
    /// step's name, in order — enough for `Result::unwrap`/`unwrap_err`
    /// panic messages and test failure output.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LmlDslPipeline")
            .field(
                "steps",
                &self.steps.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl LmlDslPipeline {
    /// Run every resolved pass in order.
    pub fn run(&mut self, input: ErasedPayload) -> crate::error::LmlResult<ErasedPayload> {
        let mut payload = input;
        for step in self.steps.iter_mut() {
            payload = run_in_lml(step.as_mut(), payload)?;
        }
        Ok(payload)
    }

    /// Number of resolved passes in the chain.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// `true` if the chain has no steps (an empty pipeline is a legal,
    /// if useless, identity chain).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Resolve a [`PipelineSpec`] against a [`PassRegistry`] and build a
/// runnable [`LmlDslPipeline`], refusing any [`Lossy`](crate::pass::Lossy)
/// step via the existing [`DynPass::lossy`] runtime check — the dynamic
/// counterpart to [`crate::pass::LmlPipeline`]'s compile-time refusal
/// (see the module docs). Parsed parameters are supplied to
/// [`PassRegistry::build_with`]; parameter-aware registrations bind them,
/// while parameterless registrations retain their fixed construction.
pub fn build_lml_pipeline(
    spec: &PipelineSpec,
    registry: &PassRegistry,
) -> Result<LmlDslPipeline, PipelineDslError> {
    let mut steps = Vec::with_capacity(spec.len());
    for step in spec {
        if !registry.contains(&step.name) {
            return Err(PipelineDslError::UnknownPass {
                name: step.name.clone(),
            });
        }
        // `register_with` constructors receive `step.params`; parameterless
        // `register` constructors intentionally ignore them. `contains`
        // guarantees name resolution, so failure here means parameter
        // rejection and is surfaced as `BadParamValue`.
        let built = registry.build_with(&step.name, &step.params).map_err(|e| {
            PipelineDslError::BadParamValue {
                name: step.name.clone(),
                detail: e.to_string(),
            }
        })?;
        if built.lossy() {
            return Err(PipelineDslError::LossyPassInLml {
                name: step.name.clone(),
            });
        }
        steps.push(built);
    }
    Ok(LmlDslPipeline { steps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LmlResult;
    use crate::pass::{DynPassAdapter, Lossy, Pass, Reversible, ReversiblePass};
    use crate::pipeline::Stage;

    /// XOR-with-key byte cipher: its own inverse, byte-exact. Mirrors
    /// `pass.rs`'s test-private `XorCipher` exactly (same shape,
    /// duplicated here rather than reused because that one is private to
    /// `pass::tests`).
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
            self.process(out)
        }
    }

    /// Truncate-to-`keep`-bytes: genuinely lossy — mirrors `pass.rs`'s
    /// test-private `TruncateLossy`.
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

    fn registry_with_both() -> PassRegistry {
        let mut reg = PassRegistry::new();
        reg.register("xor_cipher", || {
            Box::new(DynPassAdapter(XorCipher { key: 0x5A })) as Box<dyn DynPass + Send + Sync>
        });
        reg.register("truncate_lossy", || {
            Box::new(DynPassAdapter(TruncateLossy { keep: 2 })) as Box<dyn DynPass + Send + Sync>
        });
        reg
    }

    // ─── Parsing shape ───────────────────────────────────────────────

    #[test]
    fn parses_comments_blank_lines_and_bareword_params() {
        let text = "\
; reversible pipeline
# also a comment

delta_filter
xor_cipher key=66
";
        let spec = parse_pipeline(text).unwrap();
        assert_eq!(spec.len(), 2);
        assert_eq!(spec[0].name, "delta_filter");
        assert!(spec[0].params.is_empty());
        assert_eq!(spec[1].name, "xor_cipher");
        assert_eq!(spec[1].params, vec![("key".to_string(), "66".to_string())]);
    }

    #[test]
    fn parses_quoted_param_values_with_escapes() {
        let spec = parse_pipeline("relabel name=\"a\\tb\\nc\" mode=strict\n").unwrap();
        assert_eq!(spec.len(), 1);
        assert_eq!(
            spec[0].params,
            vec![
                ("name".to_string(), "a\tb\nc".to_string()),
                ("mode".to_string(), "strict".to_string()),
            ]
        );
    }

    // ─── The load-bearing gate: runtime Lossy refusal ─────────────────

    /// Runtime counterpart to `pass::LmlPipeline`'s `compile_fail`
    /// doctest: a DSL pipeline naming a registered Lossy pass must be
    /// REJECTED by `build_lml_pipeline`, not silently wired in.
    #[test]
    fn lml_pipeline_dsl_rejects_lossy_pass() {
        let reg = registry_with_both();
        let spec = parse_pipeline("xor_cipher key=90\ntruncate_lossy keep=2\n").unwrap();
        let err = build_lml_pipeline(&spec, &reg).unwrap_err();
        match err {
            PipelineDslError::LossyPassInLml { name } => assert_eq!(name, "truncate_lossy"),
            other => panic!("expected LossyPassInLml, got {other:?}"),
        }
    }

    // ─── Positive: parse + build + run, with roundtrip ────────────────

    #[test]
    fn reversible_pipeline_parses_builds_and_runs_roundtrip() {
        let reg = registry_with_both();
        let spec = parse_pipeline("; reversible-only\nxor_cipher key=90\n").unwrap();
        assert_eq!(spec.len(), 1);

        let mut pipeline = build_lml_pipeline(&spec, &reg).unwrap();
        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());

        let original = vec![1u8, 2, 3, 250, 0, 255];
        let encoded = pipeline.run(original.clone()).unwrap();
        assert_ne!(
            encoded, original,
            "sanity: the pass actually changed the bytes"
        );

        // XOR is its own inverse under `process`: running the SAME built
        // pipeline forward a second time undoes the first — proving the
        // pass's reversibility survived the DSL/registry/DynPass path
        // ("roundtrip if the passes invert").
        let decoded = pipeline.run(encoded).unwrap();
        assert_eq!(decoded, original);
    }

    // ─── Errors ────────────────────────────────────────────────────────

    #[test]
    fn unknown_pass_errors() {
        let reg = PassRegistry::new();
        let spec = parse_pipeline("nonexistent_pass\n").unwrap();
        let err = build_lml_pipeline(&spec, &reg).unwrap_err();
        match err {
            PipelineDslError::UnknownPass { name } => assert_eq!(name, "nonexistent_pass"),
            other => panic!("expected UnknownPass, got {other:?}"),
        }
    }

    /// One crafted malformed/refused input per `PipelineDslError`
    /// variant — each must fail with EXACTLY that variant.
    #[test]
    fn negative_battery_covers_each_variant() {
        // BadParam: no '=' before the next whitespace/end.
        assert!(matches!(
            parse_pipeline("xor_cipher key\n"),
            Err(PipelineDslError::BadParam { .. })
        ));
        // BadParam: empty key before '='.
        assert!(matches!(
            parse_pipeline("xor_cipher =value\n"),
            Err(PipelineDslError::BadParam { .. })
        ));

        // UnterminatedString: quoted value never closes.
        assert!(matches!(
            parse_pipeline("xor_cipher key=\"unterminated\n"),
            Err(PipelineDslError::UnterminatedString { .. })
        ));

        // BadEscape: unknown escape char `\q`.
        assert!(matches!(
            parse_pipeline("xor_cipher key=\"bad\\qescape\"\n"),
            Err(PipelineDslError::BadEscape { .. })
        ));

        // UnknownPass: parses fine, build fails (empty registry).
        {
            let reg = PassRegistry::new();
            let spec = parse_pipeline("nonexistent\n").unwrap();
            assert!(matches!(
                build_lml_pipeline(&spec, &reg),
                Err(PipelineDslError::UnknownPass { .. })
            ));
        }

        // LossyPassInLml: parses fine, build refuses.
        {
            let reg = registry_with_both();
            let spec = parse_pipeline("truncate_lossy\n").unwrap();
            assert!(matches!(
                build_lml_pipeline(&spec, &reg),
                Err(PipelineDslError::LossyPassInLml { .. })
            ));
        }
    }

    /// Bounds-safety sweep (mirrors `ir.rs`'s `truncation_sweep_never_panics`):
    /// truncating a fully valid, multi-line, comment-and-param-bearing
    /// pipeline text at every byte offset `0..len` must never panic.
    #[test]
    fn truncation_sweep_never_panics() {
        let text =
            "; reversible pipeline\n# comment\nxor_cipher key=\"a\\nb\"\ndelta_filter foo=bar\n";
        assert!(text.is_ascii(), "sweep corpus must be ASCII-safe to slice");
        for k in 0..text.len() {
            let _ = parse_pipeline(&text[..k]);
        }
    }

    /// G3: a registered pass's params are validated syntactically by
    /// `parse_pipeline` (a malformed token is still a `BadParam`/
    /// `UnterminatedString`/`BadEscape` error) but are NOT bound to the
    /// built instance — the constructor always builds whatever it was
    /// registered with. Proven here: two different `key=` values in the
    /// DSL text both resolve to the SAME hardcoded `key: 0x5A` pass, so
    /// their encoded output is identical regardless of what the DSL text
    /// said.
    #[test]
    fn g3_params_validated_syntactically_but_ignored_by_ctor() {
        let reg = registry_with_both();
        let spec_a = parse_pipeline("xor_cipher key=1\n").unwrap();
        let spec_b = parse_pipeline("xor_cipher key=255\n").unwrap();

        let mut pipeline_a = build_lml_pipeline(&spec_a, &reg).unwrap();
        let mut pipeline_b = build_lml_pipeline(&spec_b, &reg).unwrap();

        let payload = vec![1u8, 2, 3];
        let out_a = pipeline_a.run(payload.clone()).unwrap();
        let out_b = pipeline_b.run(payload).unwrap();
        assert_eq!(
            out_a, out_b,
            "ctor ignores DSL params (G3): both must use the registered key=0x5A"
        );
    }

    /// M6 (ADR 0074): the positive successor to the G3 test — `register_with`
    /// BINDS the parsed params, so different values yield different passes, and a
    /// bad value fails the build instead of silently defaulting.
    #[test]
    fn m6_register_with_binds_params_through_the_dsl() {
        use crate::error::{LmlError, LmlResult};

        struct ParamXor {
            key: u8,
        }
        impl DynPass for ParamXor {
            fn name(&self) -> &str {
                "param_xor"
            }
            fn lossy(&self) -> bool {
                false
            }
            fn run(&mut self, input: ErasedPayload) -> LmlResult<ErasedPayload> {
                Ok(input.into_iter().map(|b| b ^ self.key).collect())
            }
        }

        let mut reg = PassRegistry::new();
        reg.register_with("param_xor", |params| {
            let key: u8 = params
                .iter()
                .find(|(k, _)| k == "key")
                .map(|(_, v)| v.parse())
                .transpose()
                .map_err(|_| LmlError::InvalidHeader("param_xor: key must be a u8".into()))?
                .unwrap_or(0);
            Ok(Box::new(ParamXor { key }))
        });

        // Via the DSL: different key values → different output (contrast G3, where
        // both produced identical output because the ctor ignored the params).
        let mut pa =
            build_lml_pipeline(&parse_pipeline("param_xor key=1\n").unwrap(), &reg).unwrap();
        let mut pb =
            build_lml_pipeline(&parse_pipeline("param_xor key=255\n").unwrap(), &reg).unwrap();
        let out_a = pa.run(vec![0u8, 0, 0]).unwrap();
        let out_b = pb.run(vec![0u8, 0, 0]).unwrap();
        assert_eq!(out_a, vec![1u8, 1, 1]);
        assert_eq!(out_b, vec![255u8, 255, 255]);
        assert_ne!(
            out_a, out_b,
            "M6: params must BIND — different key ⇒ different output"
        );

        // A bad param VALUE is a build error (BadParamValue), not a silent default.
        let bad = build_lml_pipeline(&parse_pipeline("param_xor key=notau8\n").unwrap(), &reg);
        assert!(
            matches!(bad, Err(PipelineDslError::BadParamValue { .. })),
            "a bad param value must fail the build, got {bad:?}"
        );
    }
}
