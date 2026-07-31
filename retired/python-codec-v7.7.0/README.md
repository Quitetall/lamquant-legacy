# Retired `lamquant-codec` Python reference implementation

This directory preserves the final production-shipped Python source from
LamQuant Lossless revision
`f9b915466e67a87ad8d290a9793d349df250c9fb`.

It is retained for audit, historical-file recovery, and controlled rollback.
It is not a production dependency and is deliberately non-publishable:

- no `pyproject.toml`, `setup.py`, or `setup.cfg` exists here;
- current LamQuant packages must not add this directory to `PYTHONPATH`;
- current code must use ABIR, `lamquant-core`, `lamquant-neural`, and owned
  training adapters instead;
- rollback execution must use an exact verified source checkout in a supervised,
  isolated process. This snapshot is evidence, not an ambient import fallback.

`source-manifest.json` records every preserved file, byte length, SHA-256, and
the aggregate tree identity. Validate it with:

```sh
python3 tools/verify_retired_python_codec.py
```

To prove byte identity against the originating repository:

```sh
python3 tools/verify_retired_python_codec.py \
  --source-repo /path/to/LamQuant-Lossless
```

