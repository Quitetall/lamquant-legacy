# lamquant-op-event-legacy

Compatibility-only crate preserving the retired `OpEvent` protocol. This
package is non-publishable and must never enter the production dependency
graph.

## Source provenance

- Rust protocol source:
  `4ea94a8499f3493127d38d9ba9380ac54b578d2d`
- JSON Schema and Python emitter:
  `9c923f743def0eebef6394486d2c436ae240f2b3`
- TypeScript binding:
  `d3dd2d00369e31387b297c29eee6d6261c517b87`

## Compatibility scope

Preserved surface includes the enum, runner, sink, SSH transport, launcher
registry, schema fixture, Python emitter, and TypeScript binding. New code uses
compiled-plan projections and graph-core execution receipts.
