"""Legacy LML iteration decoders — opt-in only.

This package contains decoders for earlier development iterations of the LML
wire format (LMQ4, LMQ5, LML ). Nothing in production code imports from this
package. Importers must reach in by hand:

    from lamquant_codec.legacy.lossless_legacy import _decompress_legacy_bytes_ref

These decoders exist so that bytes saved during pre-v1 development can still
be read manually. They will be removed once we are confident no such bytes
remain in the field.
"""
