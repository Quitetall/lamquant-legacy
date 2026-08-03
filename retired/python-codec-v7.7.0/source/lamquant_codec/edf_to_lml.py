#!/usr/bin/env python3
"""edf_to_lml.py — Convert EDF database to LML lossless format.

Compresses an entire EDF directory into organized LML files with
complete metadata preservation. Achieves ~3:1 lossless compression
vs raw EDF while keeping all original data bit-exact recoverable.

Output structure:
  output_dir/
    {dataset}/
      {patient_id}/
        {session}_{recording}.lml
    manifest.json    — index of all files with metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


# ============================================================
# LML file container format (v1)
#
# Layout:
#   [20 bytes]  header (magic 'LML1', version, n_ch, n_windows,
#                       total_samples, window_size, metadata_len)
#   [variable]  metadata JSON (UTF-8)
#   [4*n_win]   window index (uint32 LE offsets into payload section)
#   [variable]  window payloads (each: uint32 LE length + LML packet)
#
# Each LML per-window packet has its own CRC-32.
# The file-level SHA-256 (in convert_edf_to_lml) is the ultimate check.
# ============================================================

LML_MAGIC = b'LML1'
_LML_HEADER_SIZE = 32  # magic(4) + ver(1+1) + n_ch(2) + n_win(2) + total(4) + ws(2) + sr_mhz(4) + bit_depth(1) + flags(1) + meta_len(4) + reserved(6)
_MAX_META_LEN = 0  # no hard cap — validated against file size at read time
_MAX_PAYLOAD_LEN = 256 * 1024 * 1024   # 256 MB sanity cap
_MAX_CHANNELS = 1024                    # no EEG system has more
# Nominal 250 Hz window size. write_lml_file scales by actual_sr/250 → 10 s
# at any sample rate. Must match `lml::cmd_encode`'s window_size in the Rust
# canonical encoder so cross-language output stays byte-comparable.
NOMINAL_WINDOW_SAMPLES = 2500


def write_lml_file(output_path: str, signal_int: np.ndarray,
                   metadata: dict, window_size: int = 2500,
                   noise_bits: int = 0) -> dict:
    """Write a complete LML v1 file from integer signal + metadata."""
    metadata = metadata or {}
    C, T = signal_int.shape
    if C == 0:
        raise ValueError("Cannot compress 0-channel signal.")
    if T == 0:
        raise ValueError("Cannot compress 0-sample signal.")

    sr = float(metadata.get('sample_rate', 250))
    actual_window = int(window_size * sr / 250)

    n_windows = max(1, (T + actual_window - 1) // actual_window)
    meta_json = json.dumps(metadata, default=str).encode('utf-8')

    # LML container header: 32 bytes (matches Rust binary)
    sr_mhz = int(sr * 1000)
    header = struct.pack('<4sBBHHIHIBBI2x4x',
                         LML_MAGIC,
                         1, 0,           # version major, minor
                         C,
                         n_windows,
                         T,
                         actual_window,
                         sr_mhz,         # sample_rate * 1000
                         16,             # bit_depth
                         0,              # flags
                         len(meta_json), # meta_len
                         )

    window_payloads = []
    window_offsets = []
    current_offset = 0

    for w in range(n_windows):
        start = w * actual_window
        end = min(start + actual_window, T)
        payload = _compress_bytes(signal_int[:, start:end].astype(np.float64),
                                  n_levels=3, noise_bits=noise_bits)
        window_payloads.append(payload)
        window_offsets.append(current_offset)
        current_offset += len(payload) + 4

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        f.write(header)
        f.write(meta_json)
        for offset in window_offsets:
            f.write(struct.pack('<I', offset))
        for payload in window_payloads:
            f.write(struct.pack('<I', len(payload)))
            f.write(payload)
    os.replace(tmp_path, output_path)

    compressed_size = os.path.getsize(output_path)
    raw_size = C * T * 2
    return {
        'n_windows': n_windows,
        'compressed_size': compressed_size,
        'raw_size': raw_size,
        'cr': raw_size / compressed_size if compressed_size > 0 else 0,
        'n_channels': C,
        'total_samples': T,
        'duration_s': T / sr,
    }


def _read_header(f, file_size: int):
    """Read and validate LML header. Supports v0 (18B), v1 (20B), and v1.x (32B).

    Returns (n_ch, n_windows, total_samples, window_size, meta_len).
    Raises ValueError with actionable messages on any violation.
    """
    magic = f.read(4)
    if len(magic) < 4:
        raise ValueError(
            f"File too small ({file_size} bytes). Not an LML file.")
    if magic[:3] != b'LML':
        raise ValueError(
            f"Not an LML file (magic: {magic!r}). "
            f"Check that this is a .lml file produced by LamQuant.")
    if magic != LML_MAGIC:
        lml_ver = magic[3:4]
        if lml_ver in (b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'):
            raise ValueError(
                f"LML container version {lml_ver.decode()} is newer than "
                f"this reader supports (max version 1). Update LamQuant.")
        raise ValueError(
            f"Invalid LML version byte: {magic!r}. File may be corrupt.")

    # Auto-detect header size: 32-byte (Rust), 20-byte (Python v1), 18-byte (v0)
    # Read enough for the largest header
    peek = f.read(28)  # 32 - 4 (magic already read)
    if len(peek) < 14:
        raise ValueError("Truncated LML header.")
    probe = struct.unpack('<H', peek[0:2])[0]

    if probe == 1:
        # Has version field. Check for 32-byte header (bit_depth at byte 20)
        if len(peek) >= 28 and peek[16] in (16, 24, 32):
            # 32-byte header: ver(2) + n_ch(2) + n_win(2) + total(4) + ws(2) +
            #                 sr_mhz(4) + bit_depth(1) + flags(1) + meta(4) + reserved(6)
            n_ch = struct.unpack('<H', peek[2:4])[0]
            n_windows = struct.unpack('<H', peek[4:6])[0]
            total_samples = struct.unpack('<I', peek[6:10])[0]
            window_size = struct.unpack('<H', peek[10:12])[0]
            meta_len = struct.unpack('<I', peek[18:22])[0]
            # File position is now at 32
        else:
            # 20-byte header: ver(2) + n_ch(2) + n_win(2) + total(4) + ws(2) + meta(4)
            n_ch, n_windows, total_samples, window_size, meta_len = \
                struct.unpack('<HHIHI', peek[2:16])
            # Seek back to 20 (we read too much)
            f.seek(20)
    else:
        # 18-byte header (no version): n_ch(2) + n_win(2) + total(4) + ws(2) + meta(4)
        n_ch = probe
        n_windows, total_samples, window_size, meta_len = \
            struct.unpack('<HIHI', peek[2:14])
        f.seek(18)

    # Sanity guards
    if n_ch == 0 or n_ch > _MAX_CHANNELS:
        raise ValueError(
            f"Invalid channel count {n_ch} (expected 1-{_MAX_CHANNELS}). "
            f"File is corrupt.")
    if total_samples == 0:
        raise ValueError("File declares 0 samples. Corrupt or empty.")
    if n_windows == 0:
        raise ValueError("File declares 0 windows. Corrupt.")
    if meta_len > file_size:
        raise ValueError(
            f"Metadata length {meta_len} exceeds file size {file_size}. Corrupt.")
    if window_size == 0:
        raise ValueError("Window size is 0. Corrupt.")

    return n_ch, n_windows, total_samples, window_size, meta_len


def read_lml_file(lml_path: str) -> tuple[np.ndarray, dict]:
    """Read an LML file. Returns (signal_int64, metadata_dict).

    Streaming: decompresses window-by-window, never loads entire file.
    Validates bounds, CRC (via LMQ5 per-window), and rejects corrupt
    files with actionable error messages.
    """
    file_size = os.path.getsize(lml_path)

    with open(lml_path, 'rb') as f:
        n_ch, n_windows, total_samples, window_size, meta_len = \
            _read_header(f, file_size)

        meta_raw = f.read(meta_len)
        if len(meta_raw) < meta_len:
            raise ValueError(
                f"Truncated metadata: expected {meta_len} bytes, "
                f"got {len(meta_raw)}. File is incomplete.")
        try:
            metadata = json.loads(meta_raw.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"Corrupt metadata: {e}. Header is valid but "
                f"metadata block is damaged.") from e

        idx_size = n_windows * 4
        idx_raw = f.read(idx_size)
        if len(idx_raw) < idx_size:
            raise ValueError(
                f"Truncated window index: expected {idx_size} bytes, "
                f"got {len(idx_raw)}.")

        signal = np.zeros((n_ch, total_samples), dtype=np.int64)
        for w in range(n_windows):
            len_raw = f.read(4)
            if len(len_raw) < 4:
                raise ValueError(
                    f"Truncated at window {w}/{n_windows}: "
                    f"missing payload length. File is incomplete.")

            payload_len = struct.unpack('<I', len_raw)[0]
            if payload_len > _MAX_PAYLOAD_LEN:
                raise ValueError(
                    f"Window {w} payload length {payload_len} exceeds "
                    f"maximum ({_MAX_PAYLOAD_LEN}). File is corrupt.")
            if payload_len == 0:
                raise ValueError(
                    f"Window {w} has 0-byte payload. File is corrupt.")

            payload = f.read(payload_len)
            if len(payload) < payload_len:
                raise ValueError(
                    f"Truncated at window {w}/{n_windows}: "
                    f"expected {payload_len} bytes, got {len(payload)}. "
                    f"File is incomplete — re-run compression.")

            window = _decompress_bytes(payload)

            start = w * window_size
            end = min(start + window_size, total_samples)
            w_ch = min(window.shape[0], n_ch)
            w_t = min(window.shape[1], end - start)
            signal[:w_ch, start:start + w_t] = \
                window[:w_ch, :w_t].astype(np.int64)

    return signal, metadata


def reconstruct_edf(lml_path: str, output_path: str) -> bool:
    """Reconstruct the original EDF file from an LML archive.

    If the LML file contains the preserved EDF header (edf_header field),
    the output will be bit-exact with the original. Otherwise, a best-effort
    EDF is synthesized from the metadata fields.

    Returns True if the original header was used (bit-exact), False if
    synthesized (signal-exact but header may differ).
    """
    import base64, zstandard

    signal, meta = read_lml_file(lml_path)
    C, T = signal.shape
    _dctx = zstandard.ZstdDecompressor()

    edf_header_b64 = meta.get('edf_header')
    if edf_header_b64:
        # Bit-exact reconstruction from preserved header
        raw_compressed = base64.b64decode(edf_header_b64)
        # Handle both content-sized and streaming zstd frames
        try:
            edf_header = _dctx.decompress(raw_compressed)
        except zstandard.ZstdError:
            edf_header = _dctx.decompress(raw_compressed, max_output_size=16 * 1024 * 1024)

        # Verify EDF header integrity if hash field present (FDA provenance).
        # Legacy files without the field are accepted but flagged in stats.
        expected_hdr_sha = meta.get('edf_header_sha256')
        if expected_hdr_sha:
            actual_hdr_sha = hashlib.sha256(edf_header).hexdigest()
            if actual_hdr_sha != expected_hdr_sha:
                raise ValueError(
                    f"EDF header SHA-256 mismatch: expected {expected_hdr_sha}, "
                    f"got {actual_hdr_sha}. Header has been tampered with or corrupted."
                )
        main_hdr = edf_header[:256]
        sig_hdr = edf_header[256:]
        n_signals = int(main_hdr[252:256].strip())
        n_data_records = meta.get('n_data_records', int(main_hdr[236:244].strip()))
        dur_record = meta.get('record_duration', float(main_hdr[244:252].strip()))

        # Parse ns_per_rec from the stored header
        all_ns = meta.get('all_ns_per_rec', [])
        if not all_ns:
            # Fall back to parsing from header
            off_ns = sum([16, 80, 8, 8, 8, 8, 8, 80]) * n_signals
            for i in range(n_signals):
                s = sig_hdr[off_ns + i*8:off_ns + (i+1)*8]
                all_ns.append(int(s.decode('ascii', errors='replace').strip()))

        eeg_idx = meta.get('eeg_channel_indices',
                          [i for i in range(n_signals)
                           if i not in meta.get('annotation_channel_indices', [])])
        ann_idx = meta.get('annotation_channel_indices', [])

        # Rebuild the data block
        total_per_rec = sum(all_ns)
        is_bdf = main_hdr[0:1] == b'\xff'
        bps = 3 if is_bdf else 2
        mode_ns = all_ns[eeg_idx[0]] if eeg_idx else int(T / n_data_records)

        data_block = bytearray(n_data_records * total_per_rec * bps)

        eeg_idx_map = {ch: j for j, ch in enumerate(eeg_idx)}
        for r in range(n_data_records):
            pos = 0
            for ch in range(n_signals):
                ns = all_ns[ch]
                rec_offset = r * total_per_rec * bps + pos * bps
                if ch in eeg_idx:
                    j = eeg_idx_map[ch]
                    samples = signal[j, r * mode_ns:(r + 1) * mode_ns]
                    if is_bdf:
                        # Vectorized int24 packing (replaces per-sample Python loop)
                        vals = samples[:ns].astype(np.int32)
                        vals[vals < 0] += 1 << 24
                        packed = np.zeros(ns * 3, dtype=np.uint8)
                        packed[0::3] = vals & 0xFF
                        packed[1::3] = (vals >> 8) & 0xFF
                        packed[2::3] = (vals >> 16) & 0xFF
                        data_block[rec_offset:rec_offset + ns*3] = packed.tobytes()
                    else:
                        chunk = samples[:ns].astype(np.int16)
                        data_block[rec_offset:rec_offset + ns*2] = chunk.tobytes()
                else:
                    # Restore non-EEG channel data from archive
                    non_eeg = meta.get('non_eeg_channels', {})
                    ch_key = str(ch)
                    if ch_key in non_eeg:
                        ch_raw = base64.b64decode(non_eeg[ch_key])
                        try:
                            ch_bytes = _dctx.decompress(ch_raw)
                        except zstandard.ZstdError:
                            ch_bytes = _dctx.decompress(ch_raw, max_output_size=64 * 1024 * 1024)
                        chunk_size = ns * bps
                        start = r * chunk_size
                        end = start + chunk_size
                        if end <= len(ch_bytes):
                            data_block[rec_offset:rec_offset + chunk_size] = \
                                ch_bytes[start:end]
                pos += ns

        # Append trailing partial record data if preserved
        trailing = b''
        trail_b64 = meta.get('trailing_data')
        if trail_b64:
            trail_compressed = base64.b64decode(trail_b64)
            try:
                trailing = _dctx.decompress(trail_compressed)
            except zstandard.ZstdError:
                trailing = _dctx.decompress(trail_compressed, max_output_size=64*1024*1024)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(main_hdr)
            f.write(sig_hdr)
            f.write(bytes(data_block))
            if trailing:
                f.write(trailing)
        return True

    else:
        # Synthesize EDF header from metadata (signal-exact, header best-effort)
        sr = meta.get('sample_rate', 250)
        channels = meta.get('channels', [f'EEG{i}' for i in range(C)])
        phys_mins = meta.get('phys_min', [-32768.0] * C)
        phys_maxs = meta.get('phys_max', [32767.0] * C)
        dig_mins = meta.get('dig_min', [-32768] * C)
        dig_maxs = meta.get('dig_max', [32767] * C)
        patient_id = meta.get('patient_id', '')
        recording_info = meta.get('recording_info', '')
        startdate = meta.get('startdate', '')

        ns_per_rec = int(sr)  # 1-second records
        n_data_records = T // ns_per_rec
        if n_data_records * ns_per_rec < T:
            n_data_records += 1

        # Main header (256 bytes)
        main_hdr = bytearray(256)
        main_hdr[0:8] = b'0       '
        main_hdr[8:88] = f'{patient_id:<80s}'.encode('ascii')[:80]
        main_hdr[88:168] = f'{recording_info:<80s}'.encode('ascii')[:80]
        parts = startdate.split()
        main_hdr[168:176] = f'{parts[0] if parts else "":<8s}'.encode('ascii')[:8]
        main_hdr[176:184] = f'{parts[1] if len(parts) > 1 else "":<8s}'.encode('ascii')[:8]
        hdr_bytes = 256 + C * 256
        main_hdr[184:192] = f'{hdr_bytes:<8d}'.encode('ascii')[:8]
        main_hdr[192:236] = b' ' * 44
        main_hdr[236:244] = f'{n_data_records:<8d}'.encode('ascii')[:8]
        main_hdr[244:252] = f'{1.0:<8f}'.encode('ascii')[:8]
        main_hdr[252:256] = f'{C:<4d}'.encode('ascii')[:4]

        # Signal headers (C * 256 bytes, field-by-field)
        fields = []
        widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
        for fi, w in enumerate(widths):
            for ch in range(C):
                if fi == 0:
                    val = channels[ch][:16]
                elif fi == 1:
                    val = ''
                elif fi == 2:
                    val = meta.get('phys_dim', 'uV')
                elif fi == 3:
                    val = f'{phys_mins[ch]}'
                elif fi == 4:
                    val = f'{phys_maxs[ch]}'
                elif fi == 5:
                    val = f'{dig_mins[ch]}'
                elif fi == 6:
                    val = f'{dig_maxs[ch]}'
                elif fi == 7:
                    val = ''
                elif fi == 8:
                    val = f'{ns_per_rec}'
                else:
                    val = ''
                fields.append(f'{val:<{w}s}'.encode('ascii')[:w])
        sig_hdr = b''.join(fields)

        # Data block
        data = np.zeros((n_data_records, ns_per_rec * C), dtype=np.int16)
        for r in range(n_data_records):
            for ch in range(C):
                start = r * ns_per_rec
                end = min(start + ns_per_rec, T)
                n = end - start
                data[r, ch * ns_per_rec:ch * ns_per_rec + n] = \
                    signal[ch, start:end].astype(np.int16)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(bytes(main_hdr))
            f.write(sig_hdr)
            f.write(data.tobytes())
        return False


# ============================================================
# Fast EDF reader — 24× faster than pyedflib
# ============================================================

def _parse_tal(raw_bytes: bytes) -> list:
    """Parse EDF+ Time-stamped Annotation List (TAL) from annotation channel.

    TAL format per data record:
      +onset\x15duration\x14description\x14\x00
      +onset\x14description\x14\x00          (duration optional)

    \x15 = duration separator, \x14 = annotation separator, \x00 = end.
    Multiple annotations per record are separated by \x14.
    """
    annotations = []
    # Split on \x00 (TAL terminator), each chunk is one TAL
    for tal in raw_bytes.split(b'\x00'):
        tal = tal.strip(b'\x00\x14\x15')
        if not tal or tal[0:1] not in (b'+', b'-'):
            continue
        # Split onset+duration from descriptions
        parts = tal.split(b'\x14')
        if not parts:
            continue
        # First part: +onset or +onset\x15duration
        time_part = parts[0]
        dur_split = time_part.split(b'\x15')
        try:
            onset = float(dur_split[0])
            duration = float(dur_split[1]) if len(dur_split) > 1 else 0.0
        except (ValueError, IndexError):
            continue
        # Remaining parts are description strings
        for desc_bytes in parts[1:]:
            desc = desc_bytes.decode('utf-8', errors='replace').strip()
            if desc:
                annotations.append({
                    'onset': onset,
                    'duration': duration,
                    'description': desc,
                })
    return annotations


def _unpack_int24(raw: bytes, n_samples: int) -> np.ndarray:
    """Unpack packed little-endian 24-bit signed integers (BDF format).

    BDF stores each sample as 3 bytes LE. We unpack to int32 with sign extension.
    """
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n_samples, 3)
    # Little-endian: byte0 is LSB, byte2 is MSB (with sign)
    result = (b[:, 0].astype(np.int32)
              | (b[:, 1].astype(np.int32) << 8)
              | (b[:, 2].astype(np.int32) << 16))
    # Sign-extend from 24-bit: if bit 23 is set, value is negative
    result[result >= 0x800000] -= 0x1000000
    return result


def read_edf_digital(edf_path: str):
    """Read EDF/EDF+/BDF file as digital values. No float conversion.

    Supports the complete EDF family:
      - EDF (1992): int16, continuous
      - EDF+ continuous (EDF+C): int16, TAL annotations
      - EDF+ discontinuous (EDF+D): int16, TAL annotations, non-contiguous records
      - BDF (BioSemi): int24, continuous

    24× faster than pyedflib for EDF. Bulk data read + numpy deinterleave.

    Returns:
        signal_int: [C, T] int64 (digital values, bit-exact)
        metadata: dict with channels, sample rate, patient info, annotations, etc.
    """
    from collections import Counter

    file_size = os.path.getsize(edf_path)
    if file_size < 256:
        raise ValueError(f"File too small ({file_size} bytes) for EDF/BDF header.")

    with open(edf_path, 'rb') as f:
        hdr = f.read(256)

        # ---- Detect format: EDF vs BDF ----
        version_byte = hdr[0:1]
        is_bdf = (version_byte == b'\xff')  # BDF magic
        bytes_per_sample = 3 if is_bdf else 2
        sample_dtype = None if is_bdf else np.int16  # BDF needs manual unpack

        # ---- Detect EDF+C vs EDF+D ----
        reserved_field = hdr[192:236].decode('ascii', errors='replace').strip()
        is_edfplus = 'EDF+' in reserved_field
        is_discontinuous = 'EDF+D' in reserved_field

        try:
            n_data_records = int(hdr[236:244].strip())
            dur_record = float(hdr[244:252].strip())
            n_signals = int(hdr[252:256].strip())
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid EDF/BDF header: {e}") from e

        if n_signals <= 0:
            raise ValueError(f"File declares {n_signals} signals.")
        if n_data_records <= 0:
            raise ValueError(f"File declares {n_data_records} data records.")

        # ---- Signal headers ----
        sig_hdr_size = 256 * n_signals
        sig_hdr = f.read(sig_hdr_size)
        if len(sig_hdr) < sig_hdr_size:
            raise ValueError("Truncated signal headers.")

        field_widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
        offsets = [0]
        for w in field_widths[:-1]:
            offsets.append(offsets[-1] + w * n_signals)

        def _str(fi, si):
            o = offsets[fi]
            w = field_widths[fi]
            return sig_hdr[o + si * w:o + (si + 1) * w] \
                .decode('ascii', errors='replace').strip()

        def _int(fi, si):
            return int(_str(fi, si))

        def _float(fi, si):
            return float(_str(fi, si))

        labels = [_str(0, i) for i in range(n_signals)]
        phys_dims = [_str(2, i) for i in range(n_signals)]
        phys_mins = [_float(3, i) for i in range(n_signals)]
        phys_maxs = [_float(4, i) for i in range(n_signals)]
        dig_mins = [_int(5, i) for i in range(n_signals)]
        dig_maxs = [_int(6, i) for i in range(n_signals)]
        ns_per_rec = [_int(8, i) for i in range(n_signals)]

        # ---- Identify annotation vs EEG channels ----
        ann_idx = [i for i in range(n_signals)
                   if 'annotation' in labels[i].lower()]
        eeg_idx = [i for i in range(n_signals)
                   if 'annotation' not in labels[i].lower()]
        if not eeg_idx:
            raise ValueError("No EEG channels (all are annotations).")

        # Pick the rate group with the most TOTAL DATA, not the most channels.
        # PSG and similar multi-rate EDFs carry 3 high-rate EEG/EOG channels
        # alongside 4 low-rate sensors (Resp, EMG, Temp, Marker at ~1 Hz).
        # Counting channels picks 1 Hz as the mode and silently drops the
        # 3 real EEG channels into the zstd fallback. Weight each rate by
        # samples-per-record (channels at the same rate accumulate) so the
        # rate carrying the most data wins. Matches the Rust encoder fix in
        # `lamquant-core/src/edf.rs::mode_ns selection`.
        sr_weights = {}
        for i in eeg_idx:
            sr_weights[ns_per_rec[i]] = sr_weights.get(ns_per_rec[i], 0) + ns_per_rec[i]
        # Ties on equal data-volume resolve to the first rate group seen in
        # iteration order. Channel order in EDF headers is stable across
        # reads, so this is deterministic. Higher-rate-on-tie is not chosen
        # explicitly: a tie means both groups carry the same total data, and
        # either is equally well-served by the DWT pipeline.
        mode_ns = max(sr_weights.items(), key=lambda kv: kv[1])[0]
        eeg_idx = [i for i in eeg_idx if ns_per_rec[i] == mode_ns]

        C = len(eeg_idx)
        T = mode_ns * n_data_records
        sr = mode_ns / dur_record if dur_record > 0 else 250.0

        if T < 4:
            raise ValueError(f"Recording too short ({T} samples).")

        # ---- Bulk data read ----
        total_per_rec = sum(ns_per_rec)
        expected_bytes = n_data_records * total_per_rec * bytes_per_sample
        raw = f.read(expected_bytes)
        trailing_bytes = b''
        if len(raw) < expected_bytes:
            usable = len(raw) // (total_per_rec * bytes_per_sample)
            if usable == 0:
                raise ValueError("Data block is empty or truncated.")
            usable_bytes = usable * total_per_rec * bytes_per_sample
            trailing_bytes = raw[usable_bytes:]
            raw = raw[:usable_bytes]
            n_data_records = usable
            T = mode_ns * n_data_records

    # ---- Compute per-signal offsets within each record ----
    rec_offsets = []
    pos = 0
    for i in range(n_signals):
        rec_offsets.append(pos)
        pos += ns_per_rec[i]

    # ---- Deinterleave: extract EEG channels ----
    if is_bdf:
        # BDF: 3 bytes/sample, manual int24 unpack per channel
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
        rec_bytes = total_per_rec * 3
        signal_int = np.zeros((C, T), dtype=np.int64)
        for j, ch in enumerate(eeg_idx):
            o = rec_offsets[ch]
            ns = ns_per_rec[ch]
            for r in range(n_data_records):
                rec_start = r * rec_bytes + o * 3
                chunk = _unpack_int24(
                    raw[rec_start:rec_start + ns * 3], ns)
                signal_int[j, r * ns:(r + 1) * ns] = chunk
    else:
        # EDF: 2 bytes/sample, fast numpy reshape
        data = np.frombuffer(raw, dtype=np.int16)
        records = data.reshape(n_data_records, total_per_rec)
        signal_int = np.zeros((C, T), dtype=np.int64)
        for j, ch in enumerate(eeg_idx):
            o = rec_offsets[ch]
            ns = ns_per_rec[ch]
            signal_int[j] = records[:, o:o + ns].ravel()[:T]

    # ---- Preserve raw non-EEG channel data for bit-exact roundtrip ----
    # This includes annotation channels and any other non-EEG signals.
    import base64 as _b64, zstandard as _zstd
    _cctx = _zstd.ZstdCompressor(level=9)
    non_eeg_data = {}
    for ch in range(n_signals):
        if ch in eeg_idx:
            continue
        ns = ns_per_rec[ch]
        ch_data = bytearray()
        for r in range(n_data_records):
            if is_bdf:
                rec_start = r * total_per_rec * 3 + rec_offsets[ch] * 3
                ch_data.extend(raw[rec_start:rec_start + ns * 3])
            else:
                rec_start = r * total_per_rec * 2 + rec_offsets[ch] * 2
                ch_data.extend(raw[rec_start:rec_start + ns * 2])
        compressed = _cctx.compress(bytes(ch_data))
        non_eeg_data[str(ch)] = _b64.b64encode(compressed).decode('ascii')

    # ---- EDF+ annotations (TAL) ----
    annotations = []
    if ann_idx:
        for ann_ch in ann_idx:
            o = rec_offsets[ann_ch]
            ns = ns_per_rec[ann_ch]
            for r in range(n_data_records):
                if is_bdf:
                    rec_start = r * total_per_rec * 3 + o * 3
                    ann_raw = raw[rec_start:rec_start + ns * 3]
                else:
                    rec_start = r * total_per_rec * 2 + o * 2
                    ann_raw = raw[rec_start:rec_start + ns * 2]
                annotations.extend(_parse_tal(ann_raw))

    # ---- EDF+D: check for discontinuities ----
    is_continuous = True
    if is_discontinuous and annotations:
        # The first annotation in each record is the record onset time.
        # If gaps exist between records, flag it.
        record_onsets = []
        for ann in annotations:
            if ann['description'] == '' and ann['duration'] == 0.0:
                record_onsets.append(ann['onset'])
        if len(record_onsets) >= 2:
            expected_gap = dur_record
            for k in range(1, len(record_onsets)):
                actual_gap = record_onsets[k] - record_onsets[k - 1]
                if abs(actual_gap - expected_gap) > 0.001:
                    is_continuous = False
                    break

    # ---- Patient metadata ----
    try:
        patient_raw = hdr[8:88].decode('ascii', errors='replace').strip()
        recording_raw = hdr[88:168].decode('ascii', errors='replace').strip()
        startdate_raw = hdr[168:176].decode('ascii', errors='replace').strip()
        starttime_raw = hdr[176:184].decode('ascii', errors='replace').strip()
    except Exception:
        patient_raw = recording_raw = startdate_raw = starttime_raw = ''

    # Filter out empty record-onset annotations (TAL bookkeeping, not clinical)
    clinical_annotations = [a for a in annotations if a['description']]

    fmt = 'BDF' if is_bdf else ('EDF+D' if is_discontinuous
                                 else 'EDF+C' if is_edfplus else 'EDF')

    # ---- Preserve raw EDF header for perfect round-trip ----
    # The full EDF header = 256 bytes (main) + n_signals * 256 bytes (per-signal)
    import base64, zstandard as _zstd2
    full_header_bytes = hdr + sig_hdr
    edf_header_compressed = _zstd2.ZstdCompressor(level=9).compress(full_header_bytes)
    edf_header_b64 = base64.b64encode(edf_header_compressed).decode('ascii')
    edf_header_sha256 = hashlib.sha256(full_header_bytes).hexdigest()

    # Encoder provenance — required for FDA traceability.
    try:
        from lamquant_codec import __version__ as _lq_version
    except Exception:
        _lq_version = "unknown"
    encoder_version = f"lamquant_codec/{_lq_version}"

    # ---- Store all signal channel info (including annotation channels) ----
    all_channel_labels = labels
    all_ns_per_rec = ns_per_rec
    all_phys_dims = phys_dims
    transducers = [_str(1, i) for i in range(n_signals)]
    prefilterings = [_str(7, i) for i in range(n_signals)]

    metadata = {
        'source_file': os.path.basename(edf_path),
        'source_path': str(edf_path),
        'format': fmt,
        'channels': [labels[i] for i in eeg_idx],
        'n_channels': C,
        'n_signals_total': n_signals,
        'sample_rate': sr,
        'bits_per_sample': 24 if is_bdf else 16,
        'n_data_records': n_data_records,
        'record_duration': dur_record,
        'phys_min': [phys_mins[i] for i in eeg_idx],
        'phys_max': [phys_maxs[i] for i in eeg_idx],
        'dig_min': [dig_mins[i] for i in eeg_idx],
        'dig_max': [dig_maxs[i] for i in eeg_idx],
        'phys_dim': phys_dims[eeg_idx[0]] if eeg_idx else 'uV',
        'all_labels': all_channel_labels,
        'all_ns_per_rec': all_ns_per_rec,
        'all_phys_dims': all_phys_dims,
        'transducers': transducers,
        'prefilterings': prefilterings,
        'eeg_channel_indices': eeg_idx,
        'annotation_channel_indices': ann_idx,
        'duration_s': float(T / sr),
        'continuous': is_continuous,
        'annotations': clinical_annotations,
        'patient_id': patient_raw,
        'recording_info': recording_raw,
        'startdate': f'{startdate_raw} {starttime_raw}'.strip(),
        'edf_header': edf_header_b64,
        'edf_header_sha256': edf_header_sha256,
        'encoder_version': encoder_version,
        'non_eeg_channels': non_eeg_data,
        'conversion_date': datetime.now(timezone.utc).isoformat(),
    }

    # Preserve trailing partial record data for bit-exact roundtrip
    if trailing_bytes:
        compressed_trail = _zstd2.ZstdCompressor(level=9).compress(trailing_bytes)
        metadata['trailing_data'] = base64.b64encode(compressed_trail).decode('ascii')
        metadata['trailing_data_size'] = len(trailing_bytes)

    return signal_int, metadata


# ============================================================
# EDF → LML conversion
# ============================================================

def convert_edf_to_lml(edf_path: str, output_path: str,
                       verify: bool = False, noise_bits: int = 0) -> dict:
    """Convert a single EDF file to LML format.

    Uses read_edf_digital (direct numpy, 24× faster than pyedflib).
    Reads int16 digital values directly — no float conversion.
    Mandatory SHA-256 verify-after-write roundtrip.
    """
    try:
        signal_int, metadata = read_edf_digital(edf_path)
    except Exception as e:
        return {'error': str(e)[:200]}

    sr = metadata['sample_rate']

    # SHA-256 of the input signal — ground truth fingerprint
    signal_hash = hashlib.sha256(signal_int.tobytes()).hexdigest()
    metadata['signal_sha256'] = signal_hash

    # Compress.
    # `window_size` is the NOMINAL 250 Hz window — write_lml_file scales it
    # by sr/250 internally. Passing the already-scaled `10 * sr` here led to
    # double-scaling (e.g. 50 Hz files used 100-sample windows instead of
    # 500), inflating per-window overhead and dropping CR from ~1.97 to
    # ~1.43 on sleep-cassette / 50 Hz PSG. The Rust encoder uses 2500 as
    # the nominal too — keep both reference and canonical encoder aligned.
    stats = write_lml_file(output_path, signal_int, metadata,
                           window_size=NOMINAL_WINDOW_SAMPLES,
                           noise_bits=noise_bits)

    # MANDATORY verification: decompress and compare SHA-256.
    # If this fails, the file is deleted — no silent corruption on disk.
    recon, _ = read_lml_file(output_path)
    C, T = signal_int.shape
    recon_hash = hashlib.sha256(recon[:C, :T].tobytes()).hexdigest()
    if recon_hash != signal_hash:
        os.remove(output_path)
        return {'error': f'SHA-256 MISMATCH: wrote {signal_hash[:16]}... '
                         f'read {recon_hash[:16]}... — file deleted'}
    stats['verified'] = True
    stats['signal_sha256'] = signal_hash
    stats['source'] = os.path.basename(edf_path)
    stats['sample_rate'] = sr
    return stats


# ============================================================
# Batch conversion
# ============================================================

def find_edf_files(input_dir: str) -> List[str]:
    """Find all EDF files recursively."""
    edfs = []
    for ext in ('*.edf', '*.EDF'):
        edfs.extend(str(p) for p in Path(input_dir).rglob(ext))
    return sorted(set(edfs))


def detect_dataset(edf_path: str) -> str:
    p = edf_path.lower()
    if 'chb' in p: return 'chbmit'
    if 'tueg' in p or 'tuh' in p: return 'tuh'
    if 'siena' in p: return 'siena'
    if 'eegmmi' in p: return 'eegmmidb'
    if 'sleep' in p: return 'sleep'
    return 'generic'


def make_output_path(edf_path: str, output_dir: str) -> str:
    """Generate organized output path: dataset/patient/session.lml"""
    dataset = detect_dataset(edf_path)
    base = Path(edf_path).stem
    parts = Path(edf_path).parts
    patient = 'unknown'
    for part in parts:
        if len(part) == 8 and part.startswith('aaaaaa'):
            patient = part; break
        if part.startswith('chb'):
            patient = part; break
    return os.path.join(output_dir, dataset, patient, f'{base}.lml')
