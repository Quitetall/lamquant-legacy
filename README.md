# lamquant-legacy

Optional, process-isolated adapters for retired LamQuant wire formats. This
repository is not a library dependency of ABIR, BLUT, firmware, training, or
the main LamQuant runtime.

## Retired Rust composition API

`lamquant-stage-pass-legacy` preserves the exact final public
`Stage`/`Pass`/pipeline-DSL/codec-stage API from LamQuant Lossless revision
`db7ff36aff529886195e067ea9628d3e7a08cd84`. It is a compatibility facade over
that immutable source revision, not a second implementation. Migration tools
may link it in their isolated process; current LamQuant code must use validated
ABIR Nodes, kernels, compiled plans, and execution receipts instead.

The facade is tested as part of this workspace and is intentionally
`publish = false`. It exists to keep historical source consumers recoverable
without retaining the superseded abstraction in the main dependency graph.

## Retired runtime object model

`lamquant-runtime-legacy` preserves the final `Source`, `Sink`, and
`WindowBatch` object family, its nine concrete implementations, manifest
engine, daemon, and control CLI from LamQuant revision
`93119e4e25402b2c27a15518f1d2399a98990257`. Its codec and LSL dependencies
are pinned to exact compatible source revisions. Features remain opt-in and
the package is `publish = false`.

The copy records that source snapshot, then applies three bounded safety
repairs: stream-layout changes fail without mutating buffered signal, every
manifest is constructed before any pipeline task starts, and the daemon never
replaces or cleans up a control path it did not create. These repairs preserve
compatibility behavior while preventing inherited panic, task-leak, and
destructive-path failure modes.

Current LamQuant execution uses capability-driven ABIR Nodes, compiled plans,
implementation-bound kernels, and transactional receipts. Compatibility tools
may run the retired runtime in isolation; mainline code must not depend on it.

## Retired operation-event protocol

`lamquant-op-event-legacy` preserves the final `OpEvent` enum, process runner,
bounded channel sink, SSH transport, launcher registry, JSON Schema, fixture,
Python emitter, and TypeScript binding. Rust sources come from LamQuant revision
`4ea94a8499f3493127d38d9ba9380ac54b578d2d`; codec artifacts come from
LamQuant Lossless revision `9c923f743def0eebef6394486d2c436ae240f2b3`;
the TypeScript binding comes from LamQuant Vision revision
`d3dd2d00369e31387b297c29eee6d6261c517b87`.

Current front ends consume projections bound to compiled graph, plan,
invocation, kernel, and implementation identities. Only the supervising
`PlanExecutor` creates terminal receipt/failure projections. Migration tools may
use the retired event protocol in isolation; production code must not depend on
it.

The adapter provides bounded inspection and exact, non-destructive forensic
conversion for every retired profile. BCS1, LML1, and LQTP1 additionally
support a bounded `import-semantic` process operation. That operation decodes
stored values, constructs and validates a current `AbirDataset`, writes its
canonical JSON and content-addressed payload, and retains the complete
source as an exact capsule. It also writes explicit mapping and fidelity
reports. Unversioned legacy metadata is quarantined, so the resulting coverage
is honestly reported as `projected-semantic`, not full semantic equivalence.

BCS1 and LML1 also expose `export-semantic`. It accepts canonical ABIR JSON and
an explicit list of content-addressed payload files, admits only aligned,
uniform-rate signed integer signal blocks, and writes an atomically committed
legacy wire plus receipt. The process decodes its own output and compares every
sample before commit. The receipt claims exact sample values, not full semantic
equivalence; callers must explicitly accept that projection.

LQTP1 import validates the complete fixed-stride pack, dequantizes every BFP
window into a `[window, channel, sample]` `f32` tensor, and retains the ordered
manifest SHA-256 as an exact source key. The pack contains neither row
identities nor a sampling clock or modality, so those meanings remain explicit
projection gaps. LQTP1 export restores a hash-bound source capsule exactly and
rejects missing or changed capsule bytes before writing.

The process accepts one JSON request on stdin and returns one JSON response on
stdout. A semantic request has this shape:

```json
{
  "operation": "import-semantic",
  "source": "/input/recording.lml",
  "destination": "/output/import",
  "accept_fidelity": true,
  "max_source_bytes": 1073741824,
  "max_decoded_bytes": 4294967296
}
```

The two limits are enforced before signal decoding. The destination is created
atomically and never replaces an existing, different result. Repeating an
identical request verifies every artifact and returns the same receipt.

An export request names every payload rather than trusting directory layout:

```json
{
  "operation": "export-semantic",
  "format": "bcs1",
  "dataset": "/input/dataset.json",
  "payloads": [
    {"content_id": "<abir-content-id>", "path": "/input/payload.i64le"}
  ],
  "destination": "/output/export",
  "accept_fidelity": true,
  "max_dataset_bytes": 1048576,
  "max_payload_bytes": 4294967296,
  "max_output_bytes": 4294967296,
  "window_size": 2500
}
```

During the alpha integration phase this workspace uses the ADR 0139-permitted
local mounts of ABIR and LamQuant Lossless. A standalone release must replace
those paths with reviewed, exact source revisions without changing the process
protocol or capability claims.
