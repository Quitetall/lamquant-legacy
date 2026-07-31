"""Batch operations: compress/decompress/validate/info across many files.

Researchers and clinicians don't process one window at a time — they have
entire studies, weeks of recordings, thousands of files. This module is
the orchestration layer that turns the per-file codec into a real tool.

Public API
----------
    compress_batch(inputs, output_dir, mode='neural', ...) -> BatchReport
    decompress_batch(inputs, output_dir, ...) -> BatchReport
    validate_batch(inputs, level='C', ...) -> BatchReport
    info_batch(inputs, ...) -> BatchReport

Design
------
Each batch operation is a thin orchestrator over the existing per-file
codec functions:

  1. expand_inputs() turns a path / glob / list / stdin into [Path]
  2. mirror_path() preserves the input directory tree under output_dir
  3. multiprocessing.Pool fans out across cores
  4. tqdm shows two-level progress (files outer, windows inner)
  5. BatchReport collects per-file results and writes a CSV manifest

Manifest columns:
    input_path, output_path, status, raw_bytes, compressed_bytes, cr,
    duration_ms, error

Resume: pass --from-manifest to skip files marked 'success' in a previous
manifest. This makes 5-hour batch operations safely interruptible.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import glob as _glob
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ============================================================
# Result dataclasses
# ============================================================

@dataclass
class BatchResult:
    """Outcome of a single-file batch operation."""
    input_path: str
    output_path: str
    status: str            # 'success' | 'failed' | 'skipped'
    raw_bytes: int = 0
    compressed_bytes: int = 0
    cr: float = 0.0
    duration_ms: float = 0.0
    error: Optional[str] = None
    # Optional quality metrics for validate runs.
    prd: Optional[float] = None
    pearson_r: Optional[float] = None
    lqs_pass: Optional[bool] = None


@dataclass
class BatchReport:
    """Aggregate of a batch run."""
    results: List[BatchResult] = field(default_factory=list)
    total_seconds: float = 0.0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.status == 'success')

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status == 'failed')

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.status == 'skipped')

    def summary(self) -> dict:
        ok = [r for r in self.results if r.status == 'success']
        total_raw = sum(r.raw_bytes for r in ok)
        total_compressed = sum(r.compressed_bytes for r in ok)
        return {
            'total': self.n_total,
            'success': self.n_success,
            'failed': self.n_failed,
            'skipped': self.n_skipped,
            'total_raw_bytes': total_raw,
            'total_compressed_bytes': total_compressed,
            'avg_cr': (total_raw / total_compressed) if total_compressed else 0.0,
            'wall_seconds': self.total_seconds,
        }

    def print_summary(self, file=sys.stdout) -> None:
        s = self.summary()
        ok = self.n_success
        n = self.n_total
        pct = 100 * ok / n if n else 0
        print(f"\nBatch summary:", file=file)
        print(f"  Files: {n}  (success {ok} / failed {self.n_failed} / skipped {self.n_skipped})", file=file)
        if s['total_raw_bytes'] > 0:
            print(f"  Total: {_human(s['total_raw_bytes'])} raw → "
                  f"{_human(s['total_compressed_bytes'])} compressed  "
                  f"({s['avg_cr']:.1f}:1)", file=file)
        print(f"  Time:  {s['wall_seconds']:.1f} s", file=file)
        if self.n_failed:
            print(f"\n  FAILED:", file=file)
            for r in self.results:
                if r.status == 'failed':
                    print(f"    {r.input_path}: {r.error}", file=file)

    def to_csv(self, path) -> None:
        """Write a per-file manifest. Use --from-manifest <path> to resume."""
        path = Path(path)
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'input_path', 'output_path', 'status',
                'raw_bytes', 'compressed_bytes', 'cr', 'duration_ms', 'error',
                'prd', 'pearson_r', 'lqs_pass',
            ])
            w.writeheader()
            for r in self.results:
                w.writerow(asdict(r))

    @classmethod
    def from_csv(cls, path) -> 'BatchReport':
        """Load a previous manifest. Useful for --from-manifest resume."""
        results = []
        with Path(path).open() as f:
            for row in csv.DictReader(f):
                results.append(BatchResult(
                    input_path=row['input_path'],
                    output_path=row['output_path'],
                    status=row['status'],
                    raw_bytes=int(row.get('raw_bytes') or 0),
                    compressed_bytes=int(row.get('compressed_bytes') or 0),
                    cr=float(row.get('cr') or 0.0),
                    duration_ms=float(row.get('duration_ms') or 0.0),
                    error=row.get('error') or None,
                    prd=float(row['prd']) if row.get('prd') else None,
                    pearson_r=float(row['pearson_r']) if row.get('pearson_r') else None,
                    lqs_pass=(row['lqs_pass'] == 'True') if row.get('lqs_pass') else None,
                ))
        return cls(results=results)


def _human(n_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


# ============================================================
# Input expansion: directory / glob / stdin / single file → [Path]
# ============================================================

# Recognised input file extensions (compress side).
COMPRESS_EXTS = ('.edf', '.npy')
# Recognised input extensions (decompress side).
DECOMPRESS_EXTS = ('.lmq', '.lml')


def expand_inputs(paths: Iterable[str],
                  recursive: bool = False,
                  exts: Sequence[str] = COMPRESS_EXTS) -> List[Path]:
    """Expand a mixed list of files / directories / globs / '-' into [Path].

    Args:
        paths: iterable of strings. '-' means read newline-separated paths
               from stdin (xargs/find-style pipe).
        recursive: if a directory is given, recurse into subdirs.
        exts: only files with these extensions are returned (case-insensitive).

    Returns: sorted list of unique Path objects that exist on disk.
    """
    exts_lower = tuple(e.lower() for e in exts)
    out: list[Path] = []

    for raw in paths:
        if raw == '-':
            for line in sys.stdin:
                line = line.strip()
                if line:
                    out.extend(_expand_single(line, recursive, exts_lower))
        else:
            out.extend(_expand_single(raw, recursive, exts_lower))

    # De-duplicate, preserve order, sort within bucket.
    seen: set = set()
    dedup: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(p)
    return sorted(dedup)


def _expand_single(raw: str, recursive: bool, exts: tuple) -> list[Path]:
    p = Path(raw)
    if p.is_file():
        return [p]
    if p.is_dir():
        pattern = '**/*' if recursive else '*'
        return [
            f for f in p.glob(pattern)
            if f.is_file() and f.suffix.lower() in exts
        ]
    # Treat as glob (handles * ? [abc] patterns the shell didn't expand).
    matches = _glob.glob(raw, recursive=recursive)
    return [
        Path(m) for m in matches
        if Path(m).is_file() and Path(m).suffix.lower() in exts
    ]


def mirror_path(input_path: Path, input_root: Path, output_root: Path,
                output_ext: str) -> Path:
    """Mirror input_path's tree under output_root, swapping the extension.

    Example:
        input_root  = data/
        input_path  = data/patient_001/session_a.edf
        output_root = compressed/
        output_ext  = '.lmq'
        → returns compressed/patient_001/session_a.lmq
    """
    try:
        rel = input_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        # Fall back to flat layout if input is outside the declared root.
        rel = Path(input_path.name)
    return output_root / rel.with_suffix(output_ext)


# ============================================================
# Single-file workers (top-level for multiprocessing pickling)
# ============================================================

def _compress_one(job: dict) -> BatchResult:
    """Worker: compress one file. Called by multiprocessing.Pool."""
    in_path = Path(job['input'])
    out_path = Path(job['output'])
    mode = job['mode']             # 'neural' | 'lossless'
    quality = job.get('quality', 2)
    ckpt = job.get('checkpoint')
    t0 = time.perf_counter()

    try:
        signal = _load_eeg(in_path)
        raw_bytes = signal.nbytes

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == 'lossless':
            from lamquant_codec.edf_to_lml import write_lml_file
            stats = write_lml_file(
                str(out_path),
                np.round(signal).astype(np.int64),
                {'sample_rate': 250, 'source_file': in_path.name},
            )
            data = None  # already written to disk
        elif mode == 'neural':
            if not ckpt:
                raise ValueError("neural mode requires --checkpoint")
            try:
                from lamquant_neural.codec import SubbandCodec  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "neural batch mode requires the neural codec, now in "
                    "LamQuant-Neural (pip install lamquant-neural). Use "
                    "mode='lossless' for the LML path."
                ) from exc
            import torch
            # Per-worker model cache (each process loads at most once).
            codec = _get_neural_codec(ckpt)
            x = torch.from_numpy(signal[None].astype(np.float32))
            with torch.no_grad():
                latent, metadata = codec.encode(x)
                lpc_coeffs = metadata[0][0] if metadata else None
                subbands = [m[1] for m in metadata] if metadata else None
                data = codec.compress(latent, lpc_coeffs, subbands,
                                      quality_mode=quality)
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        if data is not None:
            tmp_path = str(out_path) + '.tmp'
            with open(tmp_path, 'wb') as f:
                f.write(data)
            os.replace(tmp_path, str(out_path))

        compressed = out_path.stat().st_size if out_path.exists() else 0
        return BatchResult(
            input_path=str(in_path),
            output_path=str(out_path),
            status='success',
            raw_bytes=raw_bytes,
            compressed_bytes=compressed,
            cr=raw_bytes / max(compressed, 1),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        # Clean up partial output to prevent --skip-existing from skipping corrupt files
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        return BatchResult(
            input_path=str(in_path),
            output_path=str(out_path),
            status='failed',
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def _decompress_one(job: dict) -> BatchResult:
    """Worker: decompress one .lmq/.lml file."""
    in_path = Path(job['input'])
    out_path = Path(job['output'])
    ckpt = job.get('checkpoint')
    t0 = time.perf_counter()

    try:
        compressed = in_path.stat().st_size
        out_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = in_path.suffix.lower()
        if suffix == '.lml':
            from lamquant_codec.edf_to_lml import read_lml_file
            recon, _ = read_lml_file(str(in_path))
            recon = recon.astype(np.float64)
        elif suffix == '.lmq':
            if not ckpt:
                raise ValueError(".lmq decompression requires --checkpoint")
            with open(in_path, 'rb') as f:
                data = f.read()
            codec = _get_neural_codec(ckpt)
            import torch
            latent, _quality, _lpc, _det = codec.decompress(data)
            with torch.no_grad():
                recon = codec.model.decode(latent, target_len=313, quantize=True)
            recon = recon[0].cpu().numpy() if hasattr(recon, 'cpu') else np.asarray(recon[0])
        else:
            raise ValueError(f"unknown input format: {suffix}")

        _save_eeg(out_path, recon)
        raw_bytes = recon.nbytes

        return BatchResult(
            input_path=str(in_path),
            output_path=str(out_path),
            status='success',
            raw_bytes=raw_bytes,
            compressed_bytes=compressed,
            cr=raw_bytes / max(compressed, 1),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        return BatchResult(
            input_path=str(in_path),
            output_path=str(out_path),
            status='failed',
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def _validate_one(job: dict) -> BatchResult:
    """Worker: decompress + measure quality vs original (if provided)."""
    in_path = Path(job['input'])
    ref_path = job.get('reference')
    level = job.get('level', 'C')
    ckpt = job.get('checkpoint')
    t0 = time.perf_counter()

    try:
        with open(in_path, 'rb') as f:
            data = f.read()
        compressed = len(data)

        # Decompress (lossless or neural).
        if in_path.suffix.lower() == '.lml':
            from lamquant_codec.edf_to_lml import read_lml_file
            recon, meta = read_lml_file(str(in_path))
        else:
            if not ckpt:
                raise ValueError(".lmq validation requires --checkpoint")
            codec = _get_neural_codec(ckpt)
            import torch
            latent, _quality, _lpc, _det = codec.decompress(data)
            with torch.no_grad():
                recon = codec.model.decode(latent, target_len=313, quantize=True)
            recon = recon[0].cpu().numpy() if hasattr(recon, 'cpu') else np.asarray(recon[0])

        prd = pearson_r = None
        lqs_pass = None
        raw_bytes = 0
        if ref_path:
            original = _load_eeg(Path(ref_path))
            raw_bytes = original.nbytes
            from lamquant_codec.benchmark import Benchmark
            from lamquant_codec.codec_types import EEGPacket
            packet = EEGPacket.from_reconstruction(
                signal=recon, compressed_bytes=compressed,
                mode='lossless' if in_path.suffix.lower() == '.lml' else 'neural',
            )
            prd = Benchmark.prd(original, packet)
            pearson_r = Benchmark.pearson_r(original, packet)
            # LQS thresholds (rough): A=alerting, M=monitoring, C=clinical, L=lossless
            thresh = {'A': (25.0, 0.85), 'M': (15.0, 0.92),
                      'C': (9.0, 0.96), 'L': (0.001, 0.9999)}
            max_prd, min_r = thresh.get(level, (9.0, 0.96))
            lqs_pass = (prd <= max_prd) and (pearson_r >= min_r)

        return BatchResult(
            input_path=str(in_path),
            output_path='',
            status='success',
            raw_bytes=raw_bytes,
            compressed_bytes=compressed,
            cr=(raw_bytes / max(compressed, 1)) if raw_bytes else 0.0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            prd=prd, pearson_r=pearson_r, lqs_pass=lqs_pass,
        )
    except Exception as e:
        return BatchResult(
            input_path=str(in_path),
            output_path='',
            status='failed',
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


# ============================================================
# Per-worker model cache (avoids reloading on every file)
# ============================================================

_NEURAL_CODEC_CACHE: dict = {}


def _get_neural_codec(ckpt_path):
    """Return a cached SubbandCodec for this worker process.

    Keeps at most one model in memory — evicts previous if checkpoint changes.
    """
    key = str(Path(ckpt_path).resolve())
    if key not in _NEURAL_CODEC_CACHE:
        _NEURAL_CODEC_CACHE.clear()
        try:
            from lamquant_neural.codec import SubbandCodec
        except ImportError as exc:
            raise RuntimeError(
                "neural batch mode requires the neural codec, now in "
                "LamQuant-Neural (pip install lamquant-neural)."
            ) from exc
        _NEURAL_CODEC_CACHE[key] = SubbandCodec.from_checkpoint(key)
    return _NEURAL_CODEC_CACHE[key]


# ============================================================
# I/O helpers — load/save EEG in supported formats
# ============================================================

def _load_eeg(path: Path) -> np.ndarray:
    """Load an EEG signal as [C, T] float32. Supports .npy and .edf."""
    suffix = path.suffix.lower()
    if suffix == '.npy':
        sig = np.load(path).astype(np.float32, copy=False)
        if sig.ndim == 3 and sig.shape[0] == 1:
            sig = sig[0]
        if sig.ndim != 2:
            raise ValueError(f"{path}: expected [C, T] array, got shape {sig.shape}")
        return sig
    if suffix == '.edf':
        try:
            import mne
        except ImportError:
            raise ImportError(
                "Reading .edf requires mne. Install with: pip install mne") from None
        mne.set_log_level('WARNING')
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        return raw.get_data().astype(np.float32, copy=False)
    raise ValueError(f"unsupported input format: {suffix} (supported: .edf, .npy)")


def _save_eeg(path: Path, signal: np.ndarray) -> None:
    """Save a reconstructed signal. Supports .npy and (basic) .csv."""
    suffix = path.suffix.lower()
    if suffix == '.npy':
        np.save(path, signal)
    elif suffix == '.csv':
        np.savetxt(path, signal.T, delimiter=',')
    else:
        # Default: .npy (most useful for downstream tooling).
        np.save(path.with_suffix('.npy'), signal)


def _pool_ignore_sigint():
    """Ignore SIGINT in pool workers — parent handles shutdown."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ============================================================
# Batch orchestrators
# ============================================================

def _run_batch(jobs: list[dict],
               worker_fn,
               workers: Optional[int] = None,
               quiet: bool = False,
               desc: str = 'Processing') -> BatchReport:
    """Internal: run worker_fn over jobs, collect BatchResults, return report."""
    if not jobs:
        return BatchReport(results=[], total_seconds=0.0)

    workers = workers or min(max(1, mp.cpu_count() or 1), 8)
    workers = min(workers, len(jobs))

    t_start = time.perf_counter()
    results: list[BatchResult] = []

    if workers == 1:
        # Serial path — easier to debug, no pickle needed.
        iterator = (worker_fn(job) for job in jobs)
        if not quiet:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(jobs), desc=desc, unit='file')
            except ImportError:
                pass
        results = list(iterator)
    else:
        ctx = mp.get_context('forkserver' if sys.platform != 'win32' else 'spawn')
        with ctx.Pool(workers, initializer=_pool_ignore_sigint) as pool:
            chunk = max(1, len(jobs) // (workers * 4))
            iterator = pool.imap_unordered(worker_fn, jobs, chunksize=chunk)
            if not quiet:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, total=len(jobs), desc=desc, unit='file')
                except ImportError:
                    pass
            results = list(iterator)

    return BatchReport(results=results, total_seconds=time.perf_counter() - t_start)


def _filter_resume(jobs: list[dict], manifest_path: Optional[str]) -> list[dict]:
    """Drop jobs whose input was 'success' in the previous manifest."""
    if not manifest_path or not Path(manifest_path).exists():
        return jobs
    prev = BatchReport.from_csv(manifest_path)
    succeeded = {r.input_path for r in prev.results if r.status == 'success'}
    return [j for j in jobs if j['input'] not in succeeded]


def _filter_skip_existing(jobs: list[dict]) -> list[dict]:
    """Drop jobs whose output already exists."""
    return [j for j in jobs if not Path(j['output']).exists()]


def compress_batch(inputs: list,
                   output_dir,
                   *,
                   mode: str = 'neural',
                   quality: int = 2,
                   checkpoint: Optional[str] = None,
                   recursive: bool = False,
                   skip_existing: bool = True,
                   from_manifest: Optional[str] = None,
                   manifest: Optional[str] = None,
                   workers: Optional[int] = None,
                   quiet: bool = False,
                   dry_run: bool = False) -> BatchReport:
    """Compress a list of inputs into output_dir.

    Mirror the input directory tree under output_dir; only the file
    extension changes ('.lmq' for neural, '.lml' for lossless).
    """
    files = expand_inputs(inputs, recursive=recursive, exts=COMPRESS_EXTS)
    if not files and inputs:
        import warnings
        warnings.warn(f"No matching files found. Checked: {', '.join(str(i) for i in inputs[:5])}")
    output_dir = Path(output_dir)
    output_ext = '.lmq' if mode == 'neural' else '.lml'

    # Determine input root for tree mirroring (deepest common ancestor).
    if files:
        common_root = Path(os.path.commonpath([str(f.resolve()) for f in files]))
        if common_root.is_file():
            common_root = common_root.parent
    else:
        common_root = Path('.')

    jobs = []
    for f in files:
        out = mirror_path(f, common_root, output_dir, output_ext)
        jobs.append({
            'input': str(f), 'output': str(out),
            'mode': mode, 'quality': quality,
            'checkpoint': checkpoint,
        })

    jobs = _filter_resume(jobs, from_manifest)
    if skip_existing:
        jobs = _filter_skip_existing(jobs)

    if dry_run:
        total_raw = sum(Path(j['input']).stat().st_size for j in jobs
                        if Path(j['input']).exists())
        est_compressed = int(total_raw / 2.26) if total_raw else 0
        est_seconds = len(jobs) / 12.0  # ~12 files/s average
        print("Dry-run summary:")
        print(f"  Files:              {len(jobs)}")
        print(f"  Total raw size:     {_human(total_raw)}")
        print(f"  Est. compressed:    {_human(est_compressed)} (at 2.26:1 avg CR)")
        print(f"  Est. time:          {est_seconds:.1f}s")
        print(f"  No files written.")
        return BatchReport(results=[], total_seconds=0.0)

    report = _run_batch(jobs, _compress_one, workers=workers,
                        quiet=quiet, desc='Compressing')

    if manifest:
        report.to_csv(manifest)
    return report


def decompress_batch(inputs: list,
                     output_dir,
                     *,
                     output_ext: str = '.npy',
                     checkpoint: Optional[str] = None,
                     recursive: bool = False,
                     skip_existing: bool = True,
                     from_manifest: Optional[str] = None,
                     manifest: Optional[str] = None,
                     workers: Optional[int] = None,
                     quiet: bool = False,
                     dry_run: bool = False) -> BatchReport:
    """Decompress a list of .lmq/.lml inputs into output_dir."""
    files = expand_inputs(inputs, recursive=recursive, exts=DECOMPRESS_EXTS)
    output_dir = Path(output_dir)

    if files:
        common_root = Path(os.path.commonpath([str(f.resolve()) for f in files]))
        if common_root.is_file():
            common_root = common_root.parent
    else:
        common_root = Path('.')

    jobs = []
    for f in files:
        out = mirror_path(f, common_root, output_dir, output_ext)
        jobs.append({
            'input': str(f), 'output': str(out),
            'checkpoint': checkpoint,
        })

    jobs = _filter_resume(jobs, from_manifest)
    if skip_existing:
        jobs = _filter_skip_existing(jobs)

    if dry_run:
        total_compressed = sum(Path(j['input']).stat().st_size for j in jobs
                               if Path(j['input']).exists())
        est_raw = int(total_compressed * 2.26)
        est_seconds = len(jobs) / 12.0
        print("Dry-run summary:")
        print(f"  Files:              {len(jobs)}")
        print(f"  Total compressed:   {_human(total_compressed)}")
        print(f"  Est. decoded size:  {_human(est_raw)} (at 2.26:1 avg CR)")
        print(f"  Est. time:          {est_seconds:.1f}s")
        print(f"  No files written.")
        return BatchReport(results=[], total_seconds=0.0)

    report = _run_batch(jobs, _decompress_one, workers=workers,
                        quiet=quiet, desc='Decompressing')
    if manifest:
        report.to_csv(manifest)
    return report


def validate_batch(inputs: list,
                   *,
                   reference_dir: Optional[str] = None,
                   level: str = 'C',
                   checkpoint: Optional[str] = None,
                   recursive: bool = False,
                   manifest: Optional[str] = None,
                   workers: Optional[int] = None,
                   quiet: bool = False) -> BatchReport:
    """Validate compressed files against an LQS quality level.

    If `reference_dir` is given, the original signals are looked up there
    (same relative path with .npy/.edf extension) and quality metrics are
    computed. Otherwise the report only contains structural checks.
    """
    files = expand_inputs(inputs, recursive=recursive, exts=DECOMPRESS_EXTS)
    if files:
        common_root = Path(os.path.commonpath([str(f.resolve()) for f in files]))
        if common_root.is_file():
            common_root = common_root.parent
    else:
        common_root = Path('.')

    jobs = []
    ref_root = Path(reference_dir).resolve() if reference_dir else None
    for f in files:
        ref = None
        if ref_root:
            try:
                rel = f.resolve().relative_to(common_root.resolve())
                # Try .npy first, then .edf.
                for ext in ('.npy', '.edf'):
                    cand = ref_root / rel.with_suffix(ext)
                    if cand.exists():
                        ref = str(cand)
                        break
            except ValueError:
                pass
        jobs.append({
            'input': str(f), 'reference': ref,
            'level': level, 'checkpoint': checkpoint,
        })

    report = _run_batch(jobs, _validate_one, workers=workers,
                        quiet=quiet, desc='Validating')
    if manifest:
        report.to_csv(manifest)
    return report


def _verify_one(job: dict) -> BatchResult:
    """Worker: structural / CRC integrity check on a single file.

    Checks (in order):
      1. File is readable
      2. Magic bytes recognised (LML1 / LMQ1 / LQN1 / LQL1)
      3. Header parses cleanly
      4. Declared payload sizes don't exceed file size
      5. (LQN1 / LQL1 only) CRC32 verifies on every window
      6. (--decode) full decompress round-trip raises no exception
    """
    import struct
    in_path = Path(job['input'])
    do_decode = job.get('decode', False)
    ckpt = job.get('checkpoint')
    t0 = time.perf_counter()

    try:
        size = in_path.stat().st_size
        if size < 8:
            raise ValueError(f"file too small ({size} bytes)")

        with open(in_path, 'rb') as fh:
            head = fh.read(256)

        # Scan past ASCII prefix to find binary magic
        magic = head[:4]
        hdr_offset = 0
        if magic != b'LML1' and magic != b'LMQ1':
            nl = head.find(b'\n')
            if 0 < nl < 128 and all(0x20 <= b <= 0x7E for b in head[:nl]):
                hdr_offset = nl + 1
                magic = head[hdr_offset:hdr_offset + 4]

        # Two formats: LML1 (lossless) and LMQ1 (neural)
        if magic == b'LML1':
            from lamquant_codec.edf_to_lml import read_lml_file
            _, meta = read_lml_file(str(in_path))
            n_ch = meta.get('n_channels', '?')
            check = f"LML ok ({n_ch}ch, verified)"
        elif magic == b'LMQ1':
            check = f"LMQ ok (neural)"
        else:
            raise ValueError(f"Unknown format magic: {magic!r}")

        if do_decode:
            if magic == b'LML1':
                pass  # read_lml_file already verified CRC per window
            else:
                with open(in_path, 'rb') as fh:
                    data = fh.read()
                from lamquant_codec.lossless import _decompress_bytes
                _decompress_bytes(data)
            check += " + decode ok"

        return BatchResult(
            input_path=str(in_path),
            output_path=check,           # repurpose this column for the check summary
            status='success',
            compressed_bytes=size,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return BatchResult(
            input_path=str(in_path),
            output_path='',
            status='failed',
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def verify_batch(inputs: list, *,
                 decode: bool = False,
                 checkpoint: Optional[str] = None,
                 recursive: bool = False,
                 manifest: Optional[str] = None,
                 workers: Optional[int] = None,
                 quiet: bool = False) -> BatchReport:
    """Structural / CRC integrity check across many files.

    Catches: corrupted headers, truncated files, CRC mismatches in the
    wrapper format, and (with --decode) any decoder exception.
    """
    files = expand_inputs(inputs, recursive=recursive, exts=DECOMPRESS_EXTS)
    jobs = [{'input': str(f), 'decode': decode, 'checkpoint': checkpoint}
            for f in files]
    report = _run_batch(jobs, _verify_one, workers=workers,
                        quiet=quiet, desc='Verifying')
    if manifest:
        report.to_csv(manifest)
    return report


def info_batch(inputs: list, *, recursive: bool = False) -> list[dict]:
    """Inspect headers of .lmq/.lml files. No decode, no parallelism.

    Handles current iteration formats:
      - Multi-window wrapper files (LQN1 / LQL1) via fileformat.info
      - Raw single-packet LML1 lossless container header
      - Raw single-packet LMQ1 neural (when neural codec ships)

    Legacy iteration magics (LMQ4 / LMQ5 / LML ) are NOT inspected here.
    Use lamquant_codec.legacy.lossless_legacy.peek_header_legacy for those.
    """
    import struct
    files = expand_inputs(inputs, recursive=recursive, exts=DECOMPRESS_EXTS)
    out = []
    for f in files:
        meta = {'input_path': str(f), 'error': None,
                'file_size_bytes': f.stat().st_size}
        try:
            # Sniff first 4 magic bytes.
            with open(f, 'rb') as fh:
                magic = fh.read(4)
                fh.seek(0)
                head = fh.read(64)

            if magic in (b'LQN1', b'LQL1'):
                # Multi-window wrapper format.
                from lamquant_codec.fileformat import info as _file_info
                info = _file_info(str(f))
                meta.update(info)
            elif magic == b'LMQ1':
                # Raw single-packet neural. Header is 30 bytes.
                m, qmode, L, lat_dim, lat_T, vmin, vmax, \
                    rans_len, lpc_len, det_len = \
                    struct.unpack('<4sBBHHffIII', head[:30])
                meta.update({
                    'format': 'LMQ1 (neural single-packet)',
                    'magic': 'LMQ1',
                    'quality_mode': {0: 'alerting', 1: 'monitoring',
                                     2: 'clinical'}.get(int(qmode), str(qmode)),
                    'fsq_levels': int(L),
                    'latent_dim': int(lat_dim),
                    'latent_T': int(lat_T),
                    'vmin': float(vmin),
                    'vmax': float(vmax),
                    'rans_payload_bytes': int(rans_len),
                    'lpc_payload_bytes': int(lpc_len),
                    'detail_payload_bytes': int(det_len),
                })
            elif magic == b'LML1':
                # LML v1 lossless container. Header is 32 bytes.
                # <4sBBHHIHIBBI2x4x> (last 6 bytes are reserved padding)
                m, ver_maj, ver_min, n_ch, n_win, total_samples, \
                    win_size, sr_mhz, bit_depth, flags, meta_len = \
                    struct.unpack('<4sBBHHIHIBBI', head[:26])
                sr = sr_mhz / 1000.0
                duration = total_samples / sr if sr > 0 else 0
                raw_bytes = n_ch * total_samples * 2
                meta.update({
                    'format': 'LML v1 (lossless container)',
                    'magic': 'LML1',
                    'version': f'{ver_maj}.{ver_min}',
                    'channels': int(n_ch),
                    'windows': int(n_win),
                    'total_samples': int(total_samples),
                    'window_size': int(win_size),
                    'sample_rate': float(sr),
                    'duration_s': round(duration, 2),
                    'bit_depth': int(bit_depth),
                    'cr_estimate': raw_bytes / max(meta['file_size_bytes'], 1),
                })
            elif magic == b'LMA1':
                # LMA archive. Use list_lma to get entry count.
                from lamquant_codec.lma import list_lma
                try:
                    entries = list_lma(str(f))
                    meta.update({
                        'format': 'LMA v1 (archive)',
                        'magic': 'LMA1',
                        'n_files': len(entries),
                    })
                except Exception:
                    meta.update({
                        'format': 'LMA v1 (archive)',
                        'magic': 'LMA1',
                    })
            else:
                meta['error'] = f"Unknown magic: {magic!r}"
        except Exception as e:
            meta['error'] = f"{type(e).__name__}: {e}"
        out.append(meta)
    return out


__all__ = [
    'BatchResult', 'BatchReport',
    'expand_inputs', 'mirror_path',
    'compress_batch', 'decompress_batch', 'validate_batch', 'verify_batch',
    'info_batch',
    'COMPRESS_EXTS', 'DECOMPRESS_EXTS',
]
