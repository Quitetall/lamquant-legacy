#!/usr/bin/env python3
"""estimate_noise_floor.py -- Estimate ADC noise floor for all windows in a dataset.

Reads Q31 NPZ files, windows the signal into 2500-sample segments,
estimates noise_bits per window via bit-level autocorrelation, and
writes a noise_profile.json with per-file and per-dataset statistics.

Usage:
    python scripts/estimate_noise_floor.py /path/to/q31_events/
    python scripts/estimate_noise_floor.py /path/to/q31_events/ -o out.json
    python scripts/estimate_noise_floor.py /path/to/q31_events/ -w 4
    python scripts/estimate_noise_floor.py /path/to/q31_events/ --sample 100
"""
from __future__ import annotations
import argparse, json, multiprocessing, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lamquant_codec.ops.noise import estimate_noise_bits

WINDOW_SIZE = 2500  # samples per window, matches codec


def _process_file(npz_path: str) -> Optional[Dict]:
    """Estimate noise_bits for every window in one NPZ file."""
    try:
        with np.load(npz_path) as f:
            data = f['data']  # [21, T] int32 (Q31 EEG)
    except Exception as e:
        print(f"  WARN: skipping {npz_path}: {e}", file=sys.stderr, flush=True)
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    C, T = data.shape
    n_windows = max(1, (T + WINDOW_SIZE - 1) // WINDOW_SIZE)
    per_window: List[int] = []
    for w in range(n_windows):
        start = w * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, T)
        per_window.append(estimate_noise_bits(data[:, start:end]))
    arr = np.array(per_window)
    return {
        'path': npz_path, 'n_windows': n_windows,
        'median': int(np.median(arr)), 'p5': int(np.percentile(arr, 5)),
        'p95': int(np.percentile(arr, 95)), 'per_window': per_window,
    }


def _find_npz_files(root: str) -> List[str]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix == '.npz':
        return [str(root_path)]
    return sorted(str(p) for p in root_path.rglob('*.npz'))


def _dataset_from_path(npz_path: str) -> str:
    name = Path(npz_path).stem
    if name.endswith('_q31'):
        name = name[:-4]
    for prefix in ('chbmit_', 'tuh_', 'tuep_', 'siena_', 'eegmmidb_',
                    'eegmmi_', 'mental_arithmetic_'):
        if name.startswith(prefix):
            return prefix.rstrip('_')
    return 'unknown'


def _build_profile(file_results: List[Dict], existing: Optional[Dict]) -> Dict:
    """Build the noise_profile.json structure from per-file results."""
    all_nb: List[int] = []
    for fr in file_results:
        all_nb.extend(fr['per_window'])
    all_arr = np.array(all_nb) if all_nb else np.array([0])
    dist = Counter(all_nb)

    # Group by dataset
    ds_groups: Dict[str, List[Dict]] = {}
    for fr in file_results:
        ds_groups.setdefault(_dataset_from_path(fr['path']), []).append(fr)

    per_dataset = {}
    for ds_name, ds_files in sorted(ds_groups.items()):
        ds_nb = [nb for fr in ds_files for nb in fr['per_window']]
        ds_arr = np.array(ds_nb)
        entry = {
            'estimated_noise_bits': int(np.median(ds_arr)),
            'empirical_noise_bits': None, 'override': False, 'notes': '',
            'median': int(np.median(ds_arr)),
            'p5': int(np.percentile(ds_arr, 5)),
            'p95': int(np.percentile(ds_arr, 95)),
            'n_files': len(ds_files), 'n_windows': len(ds_nb),
        }
        # Preserve manual overrides from previous run
        if existing:
            prev = existing.get('per_dataset', {}).get(ds_name, {})
            if prev.get('override'):
                entry['empirical_noise_bits'] = prev.get('empirical_noise_bits')
                entry['override'] = True
                if prev.get('notes'):
                    entry['notes'] = prev['notes']
        per_dataset[ds_name] = entry

    per_file = {
        fr['path']: {
            'n_windows': fr['n_windows'], 'median': fr['median'],
            'p5': fr['p5'], 'p95': fr['p95'], 'per_window': fr['per_window'],
        } for fr in file_results
    }

    return {
        'version': '1.0', 'method': 'bit_autocorrelation',
        'corr_threshold': 0.05,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'summary': {
            'n_files': len(file_results), 'n_windows': len(all_nb),
            'median_noise_bits': int(np.median(all_arr)),
            'p5_noise_bits': int(np.percentile(all_arr, 5)),
            'p95_noise_bits': int(np.percentile(all_arr, 95)),
            'distribution': {str(k): dist[k] for k in sorted(dist)},
        },
        'per_dataset': per_dataset, 'per_file': per_file,
    }


def _progress(done: int, total: int, n_ok: int, results: List[Dict],
              t0: float) -> None:
    """Print progress line at 5% intervals and on completion."""
    if done % max(1, total // 20) != 0 and done != total:
        return
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0
    med = int(np.median([r['median'] for r in results])) if results else 0
    print(f"  [{done}/{total}] {n_ok} files, "
          f"median_nb={med} ({rate:.1f}/s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate ADC noise floor for Q31 NPZ dataset.")
    parser.add_argument('path',
                        help="Directory of Q31 NPZ files (searched recursively)")
    parser.add_argument('-o', '--output', default=None,
                        help="Output JSON path (default: <path>/noise_profile.json)")
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument('--sample', type=int, default=0,
                        help="Process only N random files (0 = all)")
    args = parser.parse_args()

    files = _find_npz_files(args.path)
    if not files:
        print(f"No NPZ files found at: {args.path}", file=sys.stderr)
        return 1
    if args.sample > 0 and args.sample < len(files):
        random.seed(42)
        files = random.sample(files, args.sample)

    total = len(files)
    workers = max(1, min(args.workers, total))
    out_path = args.output or os.path.join(args.path, 'noise_profile.json')

    # Load existing profile for override preservation
    existing = None
    if os.path.isfile(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    print(f"[*] Noise floor estimation", flush=True)
    print(f"    Files:   {total}", flush=True)
    print(f"    Workers: {workers}", flush=True)
    print(f"    Output:  {out_path}\n", flush=True)

    t0 = time.monotonic()
    results: List[Dict] = []
    done = 0

    if workers == 1:
        for i, fpath in enumerate(files):
            result = _process_file(fpath)
            if result is not None:
                results.append(result)
            done = i + 1
            _progress(done, total, len(results), results, t0)
    else:
        ctx = multiprocessing.get_context('forkserver')
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_process_file, f): f for f in files}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=600)
                except Exception as e:
                    print(f"  WARN: worker failed for {futures[future]}: {e}",
                          file=sys.stderr, flush=True)
                    result = None
                if result is not None:
                    results.append(result)
                done += 1
                _progress(done, total, len(results), results, t0)

    elapsed = time.monotonic() - t0
    if not results:
        print("No files processed successfully.", file=sys.stderr)
        return 1

    profile = _build_profile(results, existing)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(profile, f, indent=2)

    s = profile['summary']
    print(f"\n{'=' * 60}", flush=True)
    print(f"  NOISE FLOOR ESTIMATION COMPLETE", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Files:     {s['n_files']:>8}", flush=True)
    print(f"  Windows:   {s['n_windows']:>8}", flush=True)
    print(f"  Median nb: {s['median_noise_bits']:>8}", flush=True)
    print(f"  P5/P95:    {s['p5_noise_bits']:>3} / {s['p95_noise_bits']}", flush=True)
    print(f"  Elapsed:   {elapsed:>7.1f}s", flush=True)
    print(f"  Output:    {out_path}", flush=True)
    for ds_name, ds in profile['per_dataset'].items():
        tag = ' [OVERRIDE]' if ds['override'] else ''
        print(f"    {ds_name:22} median={ds['median']} "
              f"p5={ds['p5']} p95={ds['p95']} "
              f"files={ds['n_files']}{tag}", flush=True)
    print(f"{'=' * 60}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
