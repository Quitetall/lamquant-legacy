# lamquant-legacy — API reference

The rule this file answers to is set by the meta index (`API.md` at the root of
the LamQuant meta-repository): *if a tool isn't documented in the owning repo's
`API.md`, it isn't a supported surface.* Until this file existed, that made a
load-bearing surface undocumented — the LML conformance suite drives the
adapter below, and CI builds it on every run.

**This repository has exactly one supported surface.** Everything else here is
preserved-for-recovery, not offered for use. See "Not supported surfaces".

---

## `lamquant-legacy-adapter` — the process-isolated retired-format reader

`crates/lamquant-legacy-adapter`, binary `lamquant-legacy-adapter`.

Decode-forever for retired LamQuant wire formats. It runs as a **separate
process** and speaks JSON over stdio, which is the whole point: main's codec
refuses legacy magics by design, so the ability to read them lives outside the
main dependency graph and cannot weaken it.

### Invocation

    lamquant-legacy-adapter  <  request.json  >  response.json

One request per process. stdin is read to EOF and **capped at 1 MiB**
(`MAX_PROCESS_REQUEST_BYTES`); a larger request is rejected with
`invalid-protocol` rather than truncated.

### Request — one JSON object, tagged by `operation`

`#[serde(tag = "operation", rename_all = "kebab-case")]`

| `operation` | fields | purpose |
|---|---|---|
| `manifest` | — | report the adapter's capability manifest |
| `inspect` | `source` (path), `max_source_bytes` (u64) | identify a retired container without converting it |
| `convert-forensic` | *ConvertRequest* | convert a retired blob into governed ABIR |
| `import-semantic` | *SemanticImportRequest* | import a retired semantic record |
| `export-semantic` | *SemanticExportRequest* | export to a retired semantic record |

### Response — one JSON object, tagged by `status` with the payload under `value`

`#[serde(tag = "status", content = "value", rename_all = "kebab-case")]`

| `status` | payload |
|---|---|
| `ok-manifest` | `CapabilityManifest` |
| `ok-inspection` | `Inspection` |
| `ok-conversion` | `ConvertReceipt` |
| `ok-semantic-import` | `SemanticImportReceipt` |
| `ok-semantic-export` | `SemanticExportReceipt` |
| `error` | `{ "code": <string>, "message": <string>}` |

### The exit code is ALWAYS 0 — parse the response, never the status code

`main` builds a `ProcessResponse` (an `Error` variant on any failure), prints
it, and returns unit. There is no `process::exit` anywhere in the crate. A
caller that branches on the exit code therefore reads **every failure as a
success**. Branch on `status`, and on `code` within it.

### Error codes — the closed set

These are the `code` values `LegacyError::code()` can emit. They are a typed
vocabulary, not prose, precisely so a caller can map them without matching on
human-readable text:

    acceptance-required        crc-mismatch             decoded-output-too-large
    destination-conflict       invalid-protocol         io
    key-unavailable            malformed-container      payload-identity-mismatch
    semantic-export-unsupported                         semantic-import-unsupported
    semantic-validation        source-too-large         truncated
    unknown-magic              unsafe-source

The meta's conformance harness (`docs/specs/conformance/verify.py`) consumes
three of them directly, and the mapping is why the taxonomy must stay typed —
a corrupted payload and an incomplete file have to remain distinguishable:

| adapter `code` | conformance kind |
|---|---|
| `crc-mismatch` | `CrcMismatch` |
| `truncated` | `Truncated` |
| `unknown-magic` | `InvalidMagic` |

### Who drives it

Not hypothetical consumers — these run today:

- `docs/specs/conformance/verify.py` (meta) — `--binary` points here; the
  vectors carry frozen legacy magics that only this adapter may decode
- `.github/workflows/ci.yml:273` and `tools/ci_local.py:392` — both build
  `--release --bin lamquant-legacy-adapter` before the conformance leg
- `training/cookbooks/lamquant/python/lamquant/dataset/preprocess.py` — imports
  retired sources through the adapter as governed ABIR

---

## Not supported surfaces

Documented so the omission is a decision rather than an oversight. None of
these is offered for use; all are kept because this tree retires by sequester.

- **`lamquant-runtime-legacy`** (binary `lamquant-runtimed`, lib
  `lamquant_runtime`) — its own manifest calls it "Compatibility-only LamQuant
  Source, Sink, and WindowBatch runtime". Nothing outside this repository
  references it. The daemon that ships is the meta's `lamquant-runtimed`
  (`crates/lamquant-runtime`, ADR 0135); this is the superseded one, and the
  name collision is exactly why it is called out here.
- **`lamquant-stage-pass-legacy`** — a compatibility facade preserving the
  final public `Stage`/`Pass`/pipeline-DSL API at LamQuant-Lossless revision
  `db7ff36aff529886195e067ea9628d3e7a08cd84`. A facade over an immutable
  source revision, not a second implementation. Current code must use validated
  ABIR nodes, kernels, compiled plans and execution receipts.
- **`lamquant-abir-bridge`, `lamquant-legacy-ir`, `lamquant-lma-training-legacy`,
  `lamquant-lml-legacy`, `lamquant-op-event-legacy`** — internal crates of the
  adapter and the facades above. Not independent surfaces.

This repository is **not** a library dependency of ABIR, BLUT, firmware,
training, or the main LamQuant runtime, and must not become one. Its isolation
is the contract.
