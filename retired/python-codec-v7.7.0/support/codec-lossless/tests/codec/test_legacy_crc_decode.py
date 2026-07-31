"""LML legacy (pre-a81cd04) CRC back-compat decode gate — PYTHON parity.

Mirror of the Rust gate `lamquant-lossless/tests/legacy_crc_decode.rs`.

ROOT CAUSE: commit a81cd04 (2026-05-11, "fix(lml): CRC covers packet header
to detect single-byte header corruption") widened the LML1 per-window CRC-32
scope from ``crc32(lpc_meta || payload)`` (legacy, payload-only) to
``crc32(header[4..18] || lpc_meta || payload)`` (modern) on BOTH encode and
decode, with no version field in the LML1 header and no back-compat read path.
Every file written before a81cd04 fails CRC under the current reader even
though its bytes are perfectly intact.

THE FIX (decode-side only, in ``lossless._verify_packet_crc``): on a CRC miss
against the modern scope, recompute the legacy payload-only scope; if THAT
matches, the packet is a valid pre-a81cd04 packet — accept it, latch
``lossless.SAW_LEGACY_CRC``, and warn once. If both scopes miss, raise
``LmlCrcError`` (genuine corruption). The encoder is untouched.

Both decode paths share the helper:
  * ``lossless._decompress_bytes_ref`` — reference (pure-Python) path.
  * ``ops.fused_lml.fused_decompress`` — numba fused path, reached via the
    ``lossless._decompress_bytes`` dispatcher when numba is present.

FIXTURE: ``lamquant-lossless/tests/fixtures/legacy_payload_crc.lml`` is the
SAME real pre-a81cd04 container the Rust agent froze — a verbatim carve of the
physionet ``mental_arithmetic`` ``Subject33_2.edf``. It is a 32-byte-header
container; we carve window 0's per-window LML1 packet (the unit the CRC
fallback operates on) and feed it to both decoders. Every per-window packet in
this file uses the legacy payload-only scope, so this exercises the fallback on
real wire bytes, not a synthetic re-encode.
"""
import hashlib
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

import lamquant_codec.lossless as lossless
from lamquant_codec.errors import LmlCrcError

# Shared with the Rust gate's FIXTURE constant — the same frozen file.
# Resolve relative to THIS file so the test is portable across CI runners
# and contributor checkouts (mirrors the Rust gate's CARGO_MANIFEST_DIR).
# This file lives at <repo>/tests/codec/; the fixture at
# <repo>/lamquant-lossless/tests/fixtures/.
FIXTURE = str(
    Path(__file__).resolve().parents[2]
    / "lamquant-lossless"
    / "tests"
    / "fixtures"
    / "legacy_payload_crc.lml"
)

# sha256 over window-0's decoded samples (channel-major, each i64 little-endian).
# Frozen golden — futureproof-tests style: a fixed hash, NOT a value derived
# from the decoder under test. Regenerate ONLY if the fixture is replaced, and
# say why in the commit message. (Distinct from the Rust golden, which hashes
# the FULL multi-window container; this hashes window 0 only.)
WINDOW0_SAMPLES_SHA256 = (
    "b295f4e331d6f710805eed8357b6fc23099e41a02cc98c15bc3e9da5cb995ba7"
)


def _carve_first_window_packet(buf: bytes) -> bytes:
    """Return window 0's per-window LML1 packet from a 32-byte-header container.

    Mirrors ``first_window_packet`` in the Rust gate and the container framing
    in ``container.rs``:
      * 32-byte header: ``n_windows = u16 @8``, ``meta_len = u32 @22``.
      * index of ``n_windows × u32`` relative offsets begins at ``32 + meta_len``.
      * ``payload_start = 32 + meta_len + n_windows*4``; window 0's block sits
        at ``payload_start + index[0]`` as ``[u32 len LE][packet bytes]``.
    """
    assert buf[:4] == b"LML1", "container starts with LML1 magic"
    assert buf[20] in (16, 24, 32), "expected a 16/24/32-byte container header"
    n_windows = struct.unpack("<H", buf[8:10])[0]
    assert n_windows >= 1, "container has at least one window"
    meta_len = struct.unpack("<I", buf[22:26])[0]
    index_start = 32 + meta_len
    payload_start = index_start + n_windows * 4
    rel_off = struct.unpack("<I", buf[index_start:index_start + 4])[0]
    block_pos = payload_start + rel_off
    length = struct.unpack("<I", buf[block_pos:block_pos + 4])[0]
    packet = buf[block_pos + 4:block_pos + 4 + length]
    assert b"LML1" in packet, "window packet contains LML1 magic after ASCII prefix"
    return packet


def _samples_sha256(signal) -> str:
    """Channel-major, every sample as 8 little-endian bytes (mirror Rust)."""
    h = hashlib.sha256()
    for ch in np.asarray(signal):
        h.update(np.asarray(ch, dtype="<i8").tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def legacy_packet() -> bytes:
    with open(FIXTURE, "rb") as fh:
        buf = fh.read()
    return _carve_first_window_packet(buf)


@pytest.fixture(autouse=True)
def _reset_legacy_latch():
    """Reset the process-wide latch so each test observes its own effect."""
    saved_saw = lossless.SAW_LEGACY_CRC
    saved_warned = lossless._WARNED_LEGACY_CRC
    lossless.SAW_LEGACY_CRC = False
    lossless._WARNED_LEGACY_CRC = False
    yield
    lossless.SAW_LEGACY_CRC = saved_saw
    lossless._WARNED_LEGACY_CRC = saved_warned


def test_carved_packet_is_actually_legacy_scope(legacy_packet):
    """Sanity: the fixture's window-0 packet really uses the legacy scope.

    If this fails the fixture was regenerated with a modern encoder and the
    whole test is testing nothing.
    """
    import zlib
    mo = legacy_packet.find(b"LML1")
    inner = legacy_packet[mo:]
    (_, n_ch, T, n_levels, flags,
     lpc_len, sub_len, crc_exp) = struct.unpack("<4sHHBBIII", inner[:22])
    payload = inner[22:22 + lpc_len + sub_len]
    header_var = inner[4:18]
    crc_modern = zlib.crc32(header_var + payload) & 0xFFFFFFFF
    crc_legacy = zlib.crc32(payload) & 0xFFFFFFFF
    assert crc_modern != crc_exp, "fixture packet must NOT match the modern scope"
    assert crc_legacy == crc_exp, "fixture packet must match the legacy payload-only scope"


@pytest.mark.parametrize("decode", [
    pytest.param(lossless._decompress_bytes, id="dispatcher_fused"),
    pytest.param(lossless._decompress_bytes_ref, id="reference"),
])
def test_legacy_packet_decodes_via_fallback(decode, legacy_packet):
    """POSITIVE: a legacy payload-only-CRC packet decodes via the fallback,
    ``SAW_LEGACY_CRC`` latches, and the samples match the frozen golden."""
    assert lossless.SAW_LEGACY_CRC is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        signal = decode(legacy_packet)

    assert signal.shape[0] > 0 and signal.shape[1] > 0, "decoded signal has data"
    assert lossless.SAW_LEGACY_CRC is True, (
        "SAW_LEGACY_CRC must latch: the fixture is a legacy payload-only-CRC file"
    )
    # The fallback warns at least once on the first legacy packet of a run.
    assert any("payload-only CRC" in str(w.message) for w in caught), (
        "the legacy fallback must emit a warn-once notice"
    )
    assert _samples_sha256(signal) == WINDOW0_SAMPLES_SHA256, (
        "decoded samples drifted from the frozen golden (sha256). If the "
        "fixture changed intentionally, update WINDOW0_SAMPLES_SHA256 and say "
        "why in the commit message."
    )


def test_both_paths_agree_bit_exact(legacy_packet):
    """The fused dispatcher and the reference path must decode the legacy
    packet to bit-identical int64 samples (cross-path parity of the fix)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig_disp = lossless._decompress_bytes(legacy_packet)
        sig_ref = lossless._decompress_bytes_ref(legacy_packet)
    assert np.array_equal(
        np.asarray(sig_disp, dtype=np.int64),
        np.asarray(sig_ref, dtype=np.int64),
    ), "fused and reference legacy-fallback decodes diverged"


@pytest.mark.parametrize("decode", [
    pytest.param(lossless._decompress_bytes, id="dispatcher_fused"),
    pytest.param(lossless._decompress_bytes_ref, id="reference"),
])
def test_flipped_payload_byte_still_rejected(decode, legacy_packet):
    """NEGATIVE: flipping a payload byte makes BOTH scopes miss, so decode
    raises ``LmlCrcError`` and the legacy latch stays clear (the fallback does
    not weaken genuine-corruption detection)."""
    packet = bytearray(legacy_packet)
    mo = packet.find(b"LML1")
    # 22-byte header + 64 bytes into the payload region: squarely inside the
    # bytes covered by BOTH the modern and the legacy CRC scopes.
    flip_at = mo + 22 + 64
    assert flip_at < len(packet), "packet large enough to corrupt payload"
    packet[flip_at] ^= 0x01

    with pytest.raises(LmlCrcError) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decode(bytes(packet))

    assert lossless.SAW_LEGACY_CRC is False, (
        "a genuinely corrupt packet must NOT latch the legacy path"
    )
    assert "payload-only" in str(exc.value), (
        "corruption message should note both scopes were tried"
    )
