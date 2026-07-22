# lamquant-legacy

Optional, process-isolated adapters for retired LamQuant wire formats. This
repository is not a library dependency of ABIR, BLUT, firmware, training, or
the main LamQuant runtime.

The adapter provides bounded inspection and exact, non-destructive forensic
conversion for every retired profile. BCS1 and LML1 containers additionally
support a bounded `import-semantic` process operation. That operation decodes
integer samples, constructs and validates a current `AbirDataset`, writes its
canonical JSON and content-addressed `i64` payload, and retains the complete
source as an exact capsule. It also writes explicit mapping and fidelity
reports. Unversioned legacy metadata is quarantined, so the resulting coverage
is honestly reported as `projected-semantic`, not full semantic equivalence.

BCS1 and LML1 also expose `export-semantic`. It accepts canonical ABIR JSON and
an explicit list of content-addressed payload files, admits only aligned,
uniform-rate signed integer signal blocks, and writes an atomically committed
legacy wire plus receipt. The process decodes its own output and compares every
sample before commit. The receipt claims exact sample values, not full semantic
equivalence; callers must explicitly accept that projection.

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
