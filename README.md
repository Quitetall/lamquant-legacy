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

During the alpha integration phase this workspace uses the ADR 0139-permitted
local mounts of ABIR and LamQuant Lossless. A standalone release must replace
those paths with reviewed, exact source revisions without changing the process
protocol or capability claims.
