"""lma.py — LamQuant Archive (.lma) container format.

A single .lma file preserves an entire directory tree:
  - EDF/BDF files → LML lossless codec (domain-specific, ~2.5x CR)
  - Text files (.csv, .tse, .lbl, .txt, .json) → zstd (general, ~5-10x CR)
  - Binary files → zstd
  - Directory structure preserved exactly

The archive is self-describing, streamable, and every entry has
SHA-256 integrity. You can reconstruct the original directory
byte-for-byte from a .lma file.

Format layout:
    [4 bytes]   Magic: b'LMA1'
    [4 bytes]   Version: uint32 LE (1)
    [4 bytes]   Number of entries: uint32 LE
    [4 bytes]   Manifest length: uint32 LE
    [variable]  Manifest: zstd-compressed JSON
    [variable]  Entry payloads (concatenated)
    [32 bytes]  Archive SHA-256 (of everything before this)

Each manifest entry:
    {
        "path": "relative/path/to/file.edf",
        "original_size": 12345678,
        "compressed_size": 5678901,
        "method": "lml" | "zstd" | "store",
        "sha256": "abc123...",
        "offset": 123456,       # byte offset into payload section
    }

Usage:
    from lamquant_codec.lma import pack_lma, unpack_lma, list_lma

    # Archive a directory
    pack_lma("/data/tuh_eeg/patient001/", "patient001.lma")

    # List contents
    for entry in list_lma("patient001.lma"):
        print(entry["path"], entry["original_size"], entry["method"])

    # Extract (bit-exact reconstruction)
    unpack_lma("patient001.lma", "/data/restored/")
"""
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

LMA_MAGIC = b'LMA1'
LMA_VERSION = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# Pluggable secondary compressor — typed interface
# ============================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressedBlock:
    """Output of a secondary compression operation.

    Every compressor returns this type. Every decompressor accepts
    the raw bytes and returns plain bytes. The dataclass ensures
    a stable contract across implementations and years of changes.
    """
    data: bytes
    original_size: int
    compressor: str
    level: int


class Compressor(ABC):
    """Base class for secondary compressors.

    Subclass and implement compress() and decompress().
    Register with register_compressor().

    This typed interface ensures all compressors share the same
    contract regardless of the underlying algorithm.
    """
    name: str = "none"
    max_level: int = 9

    @abstractmethod
    def compress(self, data: bytes, level: int) -> CompressedBlock:
        """Compress data. Returns a CompressedBlock."""
        ...

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data. Returns original bytes."""
        ...


class ZstdCompressor(Compressor):
    name = "zstd"
    max_level = 22

    def compress(self, data: bytes, level: int) -> CompressedBlock:
        import zstandard
        out = zstandard.ZstdCompressor(level=level).compress(data)
        return CompressedBlock(data=out, original_size=len(data),
                               compressor=self.name, level=level)

    def decompress(self, data: bytes) -> bytes:
        import zstandard
        # max_output_size needed when frame header lacks content size
        return zstandard.ZstdDecompressor().decompress(
            data, max_output_size=len(data) * 20)


class LzmaCompressor(Compressor):
    name = "lzma"
    max_level = 9

    def compress(self, data: bytes, level: int) -> CompressedBlock:
        import lzma
        out = lzma.compress(data, preset=min(level, 9))
        return CompressedBlock(data=out, original_size=len(data),
                               compressor=self.name, level=level)

    def decompress(self, data: bytes) -> bytes:
        import lzma
        return lzma.decompress(data)


# Registry
_COMPRESSORS: Dict[str, Compressor] = {}
_ACTIVE_COMPRESSOR = "zstd"
_COMPRESSOR_LEVEL = 9


def register_compressor(compressor: Compressor):
    """Register a secondary compressor."""
    _COMPRESSORS[compressor.name] = compressor


def set_compressor(name: str, level: int = 9):
    """Set the active compressor for new archives."""
    global _ACTIVE_COMPRESSOR, _COMPRESSOR_LEVEL
    if name not in _COMPRESSORS:
        raise ValueError(f"Unknown compressor '{name}'. "
                         f"Available: {list(_COMPRESSORS.keys())}")
    _ACTIVE_COMPRESSOR = name
    _COMPRESSOR_LEVEL = level


def _compress_secondary(data: bytes, level: int = None) -> bytes:
    """Compress using the active secondary compressor."""
    level = level if level is not None else _COMPRESSOR_LEVEL
    block = _COMPRESSORS[_ACTIVE_COMPRESSOR].compress(data, level)
    return block.data


def _decompress_secondary(data: bytes, compressor: str = None) -> bytes:
    """Decompress using the named (or active) compressor."""
    name = compressor or _ACTIVE_COMPRESSOR
    if name not in _COMPRESSORS:
        raise ValueError(f"Archive uses compressor '{name}' which is not "
                         f"registered. Install the required package.")
    return _COMPRESSORS[name].decompress(data)


class NoneCompressor(Compressor):
    """Store data uncompressed."""
    name = "none"
    max_level = 0

    def compress(self, data: bytes, level: int) -> CompressedBlock:
        return CompressedBlock(data=data, original_size=len(data),
                               compressor=self.name, level=0)

    def decompress(self, data: bytes) -> bytes:
        return data


# Register built-ins
register_compressor(ZstdCompressor())
register_compressor(LzmaCompressor())
register_compressor(NoneCompressor())


def _choose_method(path: str) -> str:
    """Choose compression method based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.edf', '.bdf'):
        return 'lml'
    if ext in ('.lml', '.lmq', '.lma', '.gz', '.zst', '.zip', '.7z',
               '.png', '.jpg', '.jpeg', '.mp4', '.avi'):
        return 'store'  # already compressed
    return 'secondary'  # uses the active pluggable compressor


def pack_lma(input_dir: str, output_path: str, *,
             compressor: str = None,
             level: int = None,
             verbose: bool = True,
             progress_fn=None) -> dict:
    """Archive a directory into a .lma file.

    Returns summary dict with counts, sizes, timing.
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Not a directory: {input_dir}")

    # Discover all files
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, input_dir)
            all_files.append((full, rel))

    if not all_files:
        raise ValueError(f"No files found in {input_dir}")

    comp_name = compressor or _ACTIVE_COMPRESSOR
    comp_level = level if level is not None else _COMPRESSOR_LEVEL
    if comp_name not in _COMPRESSORS:
        raise ValueError(f"Unknown compressor '{comp_name}'")

    if verbose:
        print(f"  Archiving {len(all_files)} files from {input_dir}")
        print(f"  Secondary compressor: {comp_name} (level {comp_level})")

    t0 = time.time()
    manifest = []
    total_original = 0
    counts = {'lml': 0, 'secondary': 0, 'store': 0}

    # Stream payloads to temp file (constant memory)
    import tempfile as _tf
    payload_tmp = _tf.NamedTemporaryFile(delete=False, suffix='.lma_payload')
    payload_tmp_path = payload_tmp.name
    payload_offset = 0

    for i, (full_path, rel_path) in enumerate(all_files):
        with open(full_path, 'rb') as f:
            raw = f.read()

        original_size = len(raw)
        total_original += original_size
        file_hash = _sha256(raw)
        method = _choose_method(rel_path)

        if method == 'lml':
            try:
                from lamquant_codec.edf_to_lml import read_edf_digital, write_lml_file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.lml', delete=False) as tmp:
                    tmp_path = tmp.name

                signal_int, metadata = read_edf_digital(full_path)
                write_lml_file(tmp_path, signal_int, metadata,
                               window_size=2500)  # 10s at 250Hz ref; writer scales for actual sr
                del signal_int, metadata

                with open(tmp_path, 'rb') as f:
                    compressed = f.read()
                os.unlink(tmp_path)
            except Exception as e:
                compressed = _compress_secondary(raw, comp_level)
                method = 'secondary'
                print(f"  WARNING: LML failed for {rel_path}: {e}, "
                      f"falling back to {comp_name}", file=sys.stderr)

        elif method == 'secondary':
            compressed = _compress_secondary(raw, comp_level)

        else:  # store
            compressed = raw

        del raw
        compressed_size = len(compressed)
        offset = payload_offset
        payload_tmp.write(compressed)
        payload_offset += compressed_size
        del compressed

        manifest.append({
            'path': rel_path,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'method': method,
            'sha256': file_hash,
            'offset': offset,
        })
        counts[method] += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(all_files)} files...")
        if progress_fn:
            progress_fn(i + 1, len(all_files), rel_path)

    payload_tmp.flush()
    payload_tmp_path = payload_tmp.name
    payload_tmp.close()

    # Wrap manifest with archive metadata
    archive_manifest = {
        'compressor': comp_name,
        'compressor_level': comp_level,
        'files': manifest,
    }
    manifest_json = json.dumps(archive_manifest, separators=(',', ':')).encode('utf-8')
    # Manifest always zstd — ensures cross-language compatibility (Rust reader)
    import zstandard
    manifest_compressed = zstandard.ZstdCompressor(level=comp_level).compress(manifest_json)

    # Write archive: header + manifest + payloads (streamed from temp)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    hasher = hashlib.sha256()

    with open(output_path, 'wb') as f:
        header = struct.pack('<4sIII',
                             LMA_MAGIC, LMA_VERSION,
                             len(manifest), len(manifest_compressed))
        f.write(header)
        hasher.update(header)

        f.write(manifest_compressed)
        hasher.update(manifest_compressed)

        # Stream payloads from temp file in chunks
        with open(payload_tmp_path, 'rb') as ptf:
            while True:
                chunk = ptf.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)

        f.write(hasher.digest())

    try:
        os.unlink(payload_tmp_path)
    except OSError:
        pass

    elapsed = time.time() - t0
    archive_size = os.path.getsize(output_path)

    summary = {
        'files': len(manifest),
        'original_bytes': total_original,
        'archive_bytes': archive_size,
        'cr': total_original / archive_size if archive_size > 0 else 0,
        'elapsed_s': elapsed,
        'counts': counts,
    }

    if verbose:
        print(f"  {len(manifest)} files archived")
        print(f"  Original:  {total_original / 2**20:,.1f} MiB")
        print(f"  Archive:   {archive_size / 2**20:,.1f} MiB  ({summary['cr']:.2f}x)")
        print(f"  Methods:   {counts['lml']} LML, {counts['secondary']} zstd, {counts['store']} stored")
        print(f"  Time:      {elapsed:.1f}s")

    return summary


def _read_lma_manifest(archive_path: str) -> dict:
    """Read and decompress the archive manifest. Returns full manifest dict."""
    with open(archive_path, 'rb') as f:
        magic = f.read(4)
        if magic != LMA_MAGIC:
            raise ValueError(f"Not an LMA archive (magic: {magic!r})")

        version, n_entries, manifest_len = struct.unpack('<III', f.read(12))
        if version > LMA_VERSION:
            raise ValueError(f"LMA version {version} not supported (max {LMA_VERSION})")

        manifest_compressed = f.read(manifest_len)

    # Try each registered compressor until one works
    for name in [_ACTIVE_COMPRESSOR] + list(_COMPRESSORS.keys()):
        try:
            raw = _COMPRESSORS[name].decompress(manifest_compressed)
            return json.loads(raw)
        except Exception:
            continue
    raise ValueError("Cannot decompress manifest — no compatible compressor found")


def list_lma(archive_path: str) -> List[dict]:
    """List contents of an LMA archive without extracting."""
    manifest = _read_lma_manifest(archive_path)
    # Handle both old format (list) and new format (dict with 'files')
    if isinstance(manifest, list):
        return manifest
    return manifest.get('files', [])


def unpack_lma(archive_path: str, output_dir: str, *,
               dataset: str = None,
               training_split: bool = False,
               verify: bool = True,
               verbose: bool = True) -> dict:
    """Extract an LMA archive, reconstructing the original directory tree.

    EDF files are reconstructed from LML (bit-exact).
    All other files are decompressed from zstd or stored raw.
    Every file is SHA-256 verified against the manifest.

    Args:
        dataset: Extract only this dataset subdirectory (e.g. "tusz").
                 None = extract everything.
        training_split: If True, deduplicate by content hash across all
                        datasets — each unique file appears once. Useful
                        for building a training memmap where duplicate
                        recordings would bias the model. Files are placed
                        in a flat structure under output_dir/.
        verify: SHA-256 verify each extracted file.
    """
    with open(archive_path, 'rb') as f:
        # Read and verify archive checksum
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        archive_data = f.read(file_size - 32)
        stored_hash = f.read(32)

        if verify:
            actual_hash = hashlib.sha256(archive_data).digest()
            if actual_hash != stored_hash:
                raise ValueError("Archive SHA-256 mismatch — file is corrupted")

    # Parse header
    magic = archive_data[:4]
    if magic != LMA_MAGIC:
        raise ValueError(f"Not an LMA archive (magic: {magic!r})")

    version, n_entries, manifest_len = struct.unpack('<III', archive_data[4:16])

    # Decompress manifest
    manifest_compressed = archive_data[16:16 + manifest_len]
    archive_meta = None
    for name in [_ACTIVE_COMPRESSOR] + list(_COMPRESSORS.keys()):
        try:
            raw = _COMPRESSORS[name].decompress(manifest_compressed)
            archive_meta = json.loads(raw)
            break
        except Exception:
            continue
    if archive_meta is None:
        raise ValueError("Cannot decompress manifest")

    # Extract file list and compressor info
    if isinstance(archive_meta, list):
        # Old format
        file_entries = archive_meta
        comp_name = 'zstd'
    else:
        file_entries = archive_meta.get('files', [])
        comp_name = archive_meta.get('compressor', 'zstd')

    # Payload section starts after header + manifest
    payload_start = 16 + manifest_len

    t0 = time.time()
    extracted = verified = failed = skipped_dedup = skipped_filter = 0
    seen_hashes = set()  # for training_split dedup

    for entry in file_entries:
        rel_path = entry['path']
        method = entry['method']

        # Dataset filter: only extract files from the requested dataset
        if dataset is not None:
            # Super archives have paths like "tusz/edf/patient/file.lml"
            top_dir = rel_path.split('/')[0] if '/' in rel_path else ''
            if top_dir != dataset:
                skipped_filter += 1
                continue

        # Training split: skip duplicate content across datasets
        if training_split:
            file_hash = entry.get('sha256', '')
            if file_hash and file_hash in seen_hashes:
                skipped_dedup += 1
                continue
            if file_hash:
                seen_hashes.add(file_hash)
        offset = entry['offset']
        comp_size = entry['compressed_size']
        orig_size = entry['original_size']
        expected_hash = entry['sha256']

        compressed = archive_data[payload_start + offset:
                                  payload_start + offset + comp_size]

        # Decompress
        if method == 'lml':
            # Write temp LML, reconstruct EDF
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.lml', delete=False) as tmp:
                tmp.write(compressed)
                tmp_path = tmp.name

            try:
                from lamquant_codec.edf_to_lml import reconstruct_edf
                out_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
                reconstruct_edf(tmp_path, out_path)
            finally:
                os.unlink(tmp_path)

        elif method in ('secondary', 'zstd'):
            # 'zstd' for backward compat, 'secondary' for new archives
            raw = _decompress_secondary(compressed, comp_name)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(raw)

        else:  # store
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(compressed)

        extracted += 1

        # Verify
        if verify:
            with open(out_path, 'rb') as f:
                actual_hash = _sha256(f.read())
            if actual_hash == expected_hash:
                verified += 1
            else:
                failed += 1
                if verbose:
                    print(f"  HASH MISMATCH: {rel_path}")

        if verbose and extracted % 100 == 0:
            print(f"    {extracted}/{len(file_entries)} extracted...")

    elapsed = time.time() - t0
    summary = {
        'extracted': extracted,
        'verified': verified,
        'failed': failed,
        'skipped_dedup': skipped_dedup,
        'skipped_filter': skipped_filter,
        'elapsed_s': elapsed,
    }

    if verbose:
        parts = [f"{extracted} extracted", f"{verified} verified",
                 f"{failed} failed"]
        if skipped_filter > 0:
            parts.append(f"{skipped_filter} filtered (dataset={dataset})")
        if skipped_dedup > 0:
            parts.append(f"{skipped_dedup} deduped (training_split)")
        print(f"  {', '.join(parts)} ({elapsed:.1f}s)")

    return summary


def list_datasets(archive_path: str) -> dict:
    """List datasets in a super archive with file counts and sizes.

    Returns {name: {"files": N, "bytes": N, "unique_bytes": N}}.
    """
    entries = list_lma(archive_path)
    datasets = {}
    for e in entries:
        top = e['path'].split('/')[0] if '/' in e['path'] else '_root'
        if top not in datasets:
            datasets[top] = {'files': 0, 'bytes': 0, 'unique_bytes': 0}
        datasets[top]['files'] += 1
        datasets[top]['bytes'] += e['original_size']
        if not e.get('dedup'):
            datasets[top]['unique_bytes'] += e['compressed_size']
    return datasets


def verify_lma(archive_path: str, verbose: bool = True) -> bool:
    """Verify archive integrity without extracting."""
    with open(archive_path, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        archive_data = f.read(file_size - 32)
        stored_hash = f.read(32)

    actual_hash = hashlib.sha256(archive_data).digest()
    ok = actual_hash == stored_hash

    if verbose:
        if ok:
            entries = list_lma(archive_path)
            total_orig = sum(e['original_size'] for e in entries)
            print(f"  Archive OK: {len(entries)} files, "
                  f"{total_orig / 2**20:,.1f} MiB original, "
                  f"{file_size / 2**20:,.1f} MiB archived")
        else:
            print(f"  CORRUPT: SHA-256 mismatch")

    return ok


def pack_super_lma(input_dirs: dict, output_path: str, *,
                   compressor: str = None,
                   level: int = None,
                   dedup: bool = True,
                   verbose: bool = True) -> dict:
    """Archive multiple directories into one super .lma, deduplicating by content.

    Each input directory becomes a subdirectory in the archive. Files with
    identical SHA-256 across directories are stored once and referenced by
    all paths that share them.

    Args:
        input_dirs: {name: path} mapping, e.g. {"tueg": "/data/tueg", "tusz": "/data/tusz"}
        output_path: Output .lma file path
        dedup: If True, files with identical content are stored once (default True)

    Example:
        pack_super_lma({
            "tueg": "/data/lml/tueg_super",
            "tusz": "/data/lml/tusz",
            "chbmit": "/data/lml/chbmit",
        }, "eeg_super.lma")
    """
    comp_name = compressor or _ACTIVE_COMPRESSOR
    comp_level = level if level is not None else _COMPRESSOR_LEVEL

    # Discover all files across all directories
    all_files = []  # (full_path, archive_path)
    for name, dir_path in sorted(input_dirs.items()):
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            if verbose:
                print(f"  WARNING: {name} ({dir_path}) not found, skipping")
            continue
        for root, dirs, files in os.walk(dir_path):
            for f in sorted(files):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dir_path)
                archive_path = os.path.join(name, rel)
                all_files.append((full, archive_path))

    if not all_files:
        raise ValueError("No files found in any input directory")

    if verbose:
        print(f"  Super archive: {len(all_files):,} files from {len(input_dirs)} directories")
        if dedup:
            print(f"  Deduplication: ON (identical files stored once)")

    t0 = time.time()
    manifest = []
    total_original = 0
    counts = {'lml': 0, 'secondary': 0, 'store': 0, 'dedup': 0}

    # Dedup index: sha256 → (offset, compressed_size, method)
    seen_hashes: dict = {} if dedup else None

    # Stream payloads to a temp file instead of RAM
    import tempfile as _tf
    payload_tmp = _tf.NamedTemporaryFile(delete=False, suffix='.lma_payload')
    payload_offset = 0

    for i, (full_path, archive_path) in enumerate(all_files):
        with open(full_path, 'rb') as f:
            raw = f.read()

        original_size = len(raw)
        total_original += original_size
        file_hash = _sha256(raw)
        method = _choose_method(archive_path)

        # Dedup check
        if dedup and file_hash in seen_hashes:
            ref_offset, ref_size, ref_method = seen_hashes[file_hash]
            manifest.append({
                'path': archive_path,
                'original_size': original_size,
                'compressed_size': ref_size,
                'method': ref_method,
                'sha256': file_hash,
                'offset': ref_offset,
                'dedup': True,
            })
            counts['dedup'] += 1
            del raw
            continue

        # Compress based on method
        if method == 'lml':
            try:
                from lamquant_codec.edf_to_lml import read_edf_digital, write_lml_file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.lml', delete=False) as tmp:
                    tmp_path = tmp.name
                signal_int, metadata = read_edf_digital(full_path)
                write_lml_file(tmp_path, signal_int, metadata,
                               window_size=2500)  # 10s at 250Hz ref; writer scales for actual sr
                del signal_int, metadata
                with open(tmp_path, 'rb') as f:
                    compressed = f.read()
                os.unlink(tmp_path)
            except Exception as e:
                compressed = _compress_secondary(raw, comp_level)
                method = 'secondary'
                print(f"  WARNING: LML failed for {archive_path}: {e}",
                      file=sys.stderr)

        elif method == 'secondary':
            compressed = _compress_secondary(raw, comp_level)

        else:  # store
            compressed = raw

        del raw  # free source data immediately

        compressed_size = len(compressed)
        offset = payload_offset
        payload_tmp.write(compressed)
        payload_offset += compressed_size
        del compressed  # free compressed data immediately

        # Register in dedup index
        if dedup:
            seen_hashes[file_hash] = (offset, compressed_size, method)

        manifest.append({
            'path': archive_path,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'method': method,
            'sha256': file_hash,
            'offset': offset,
        })
        counts[method] += 1

        if verbose and (i + 1) % 500 == 0:
            dedup_saved = sum(1 for e in manifest if e.get('dedup'))
            print(f"    {i+1}/{len(all_files)} files... "
                  f"({dedup_saved} deduplicated, "
                  f"{payload_offset / 2**30:.1f} GiB written)")

    payload_tmp.flush()
    payload_tmp_path = payload_tmp.name
    payload_tmp.close()

    # Build archive: header + manifest + payloads (streamed from temp file)
    archive_manifest = {
        'compressor': comp_name,
        'compressor_level': comp_level,
        'super': True,
        'sources': {name: os.path.abspath(p) for name, p in input_dirs.items()},
        'dedup_enabled': dedup,
        'files': manifest,
    }
    manifest_json = json.dumps(archive_manifest, separators=(',', ':')).encode('utf-8')
    # Manifest always zstd — ensures cross-language compatibility (Rust reader)
    import zstandard
    manifest_compressed = zstandard.ZstdCompressor(level=comp_level).compress(manifest_json)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    hasher = hashlib.sha256()

    with open(output_path, 'wb') as f:
        header = struct.pack('<4sIII',
                             LMA_MAGIC, LMA_VERSION,
                             len(manifest), len(manifest_compressed))
        f.write(header)
        hasher.update(header)
        f.write(manifest_compressed)
        hasher.update(manifest_compressed)

        # Stream payloads from temp file in chunks (constant memory)
        with open(payload_tmp_path, 'rb') as ptf:
            while True:
                chunk = ptf.read(8 * 1024 * 1024)  # 8 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)

        f.write(hasher.digest())

    os.unlink(payload_tmp_path)  # clean up temp file

    elapsed = time.time() - t0
    archive_size = os.path.getsize(output_path)
    unique_files = counts['lml'] + counts['secondary'] + counts['store']
    dedup_count = counts['dedup']
    dedup_bytes = sum(e['original_size'] for e in manifest if e.get('dedup'))

    summary = {
        'total_files': len(manifest),
        'unique_files': unique_files,
        'deduplicated': dedup_count,
        'dedup_bytes_saved': dedup_bytes,
        'original_bytes': total_original,
        'archive_bytes': archive_size,
        'cr': total_original / archive_size if archive_size > 0 else 0,
        'elapsed_s': elapsed,
        'counts': counts,
    }

    if verbose:
        print(f"\n  === Super Archive ===")
        print(f"  Total files:    {len(manifest):,}")
        print(f"  Unique:         {unique_files:,} ({counts['lml']} LML, "
              f"{counts['secondary']} zstd, {counts['store']} stored)")
        print(f"  Deduplicated:   {dedup_count:,} "
              f"(saved {dedup_bytes / 2**20:,.1f} MiB)")
        print(f"  Original:       {total_original / 2**30:,.1f} GiB")
        print(f"  Archive:        {archive_size / 2**30:,.1f} GiB "
              f"({summary['cr']:.2f}x)")
        print(f"  Time:           {elapsed:.1f}s")

    return summary
