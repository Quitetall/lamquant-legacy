# lamquant-runtime-legacy

Compatibility-only crate preserving the legacy runtime object model (`Source`,
`Sink`, and `WindowBatch`) plus pre-graph implementations.

Source snapshot: `93119e4e25402b2c27a15518f1d2399a98990257`.
The sequestered copy adds fail-closed compatibility hardening for stream-layout
changes, manifest construction, and Unix control-socket ownership. Those
changes are intentionally limited to preventing panics, detached tasks, and
destructive path replacement; they do not create a second production runtime.

Mainline runtime now uses ABIR capability-based Nodes, plans, kernels, and
receipts for execution and provenance. This crate is retained only for migration
and compatibility use and is **not for production use**.
