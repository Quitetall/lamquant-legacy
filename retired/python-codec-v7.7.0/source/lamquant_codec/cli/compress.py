#!/usr/bin/env python3
"""
lamquant compress — convert EDF corpus to LML lossless archive.

    lamquant compress /data/tueg/ -o /data/lml/ -j 4
    lamquant compress recording.edf -o recording.lml
    lamquant compress /data/tueg/ -o /data/lml/ --skip-existing --quiet
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from lamquant_codec._paths import REPO_ROOT as _REPO

# Numba cache: use repo-local cache, not /tmp. Silent — the user never sees this.
_NUMBA_CACHE = _REPO / ".numba_cache"
if "NUMBA_CACHE_DIR" not in os.environ:
    _NUMBA_CACHE.mkdir(exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_NUMBA_CACHE)

from lamquant_codec.edf_to_lml import (
    convert_edf_to_lml, find_edf_files, make_output_path,
)
from lamquant_codec.cli.config import load_config, LamQuantConfig
from lamquant_codec.cli.state import (
    StateFile, print_recovery_summary, COMPLETED,
)
from lamquant_codec.cli.readout import (
    C, B, RunStats, FileResult, Dashboard, AuditLog,
    print_banner, print_summary, print_summary_json,
    print_file_line, write_manifest, _bytes, _ratio,
)


# ────────────────────────────────────────────────────────────────────
# Worker function (runs in subprocess for -j > 1)
# ────────────────────────────────────────────────────────────────────

def _worker_init():
    """Ignore SIGINT in workers — parent handles shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _copy_as_error(edf_path: str, out_path: str):
    """Copy original EDF to output dir with .edf.error extension.
    Ensures no data is lost even when compression fails."""
    import shutil
    error_path = out_path.replace('.lml', '.edf.error')
    try:
        os.makedirs(os.path.dirname(error_path) or '.', exist_ok=True)
        shutil.copy2(edf_path, error_path)
    except OSError:
        pass  # disk full or permissions — already logged as error


def _compress_one(args_tuple):
    edf_path, out_path, atomic, noise_bits, verify = args_tuple
    t0 = time.perf_counter()
    try:
        edf_size = os.path.getsize(edf_path)
    except OSError as e:
        return FileResult(path=edf_path, bytes_in=0,
                          status="error", error=f"Cannot stat input: {e}",
                          duration_s=time.perf_counter() - t0)

    if atomic:
        import tempfile
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".lml.tmp")
        os.close(fd)
        target = tmp
    else:
        target = out_path

    try:
        stats = convert_edf_to_lml(edf_path, target, verify=verify,
                                   noise_bits=noise_bits)
    except Exception as e:
        if atomic and os.path.exists(tmp):
            os.unlink(tmp)
        # Error: copy original EDF to output as .edf.error so nothing is lost
        _copy_as_error(edf_path, out_path)
        return FileResult(path=edf_path, bytes_in=edf_size,
                          status="error", error=str(e)[:2000],
                          duration_s=time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0

    if "error" in stats:
        if atomic and os.path.exists(tmp):
            os.unlink(tmp)
        _copy_as_error(edf_path, out_path)
        return FileResult(path=edf_path, bytes_in=edf_size,
                          status="error", error=stats["error"],
                          duration_s=elapsed)

    # Atomic rename on success
    if atomic:
        os.replace(tmp, out_path)

    # fsync
    try:
        d = os.open(os.path.dirname(out_path), os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    except OSError:
        pass

    lml_size = os.path.getsize(out_path)

    # Flag expansion: LML larger than raw signal = incompressible (noise/artifact)
    raw_signal_size = stats.get("n_channels", 0) * stats.get("total_samples", 0) * 2
    status = "ok"
    error = None
    if raw_signal_size > 0 and lml_size > raw_signal_size:
        status = "warning"
        error = (f"expansion: LML ({lml_size:,} B) > raw ({raw_signal_size:,} B). "
                 f"Likely noise/artifact. File preserved but flagged.")

    return FileResult(
        path=edf_path,
        bytes_in=edf_size,
        bytes_out=lml_size,
        duration_s=elapsed,
        sha256=stats.get("signal_sha256", ""),
        n_channels=stats.get("n_channels", 0),
        sample_rate=stats.get("sample_rate", 0),
        n_samples=stats.get("total_samples", 0),
        n_windows=stats.get("n_windows", 0),
        status=status,
        error=error,
    )


# ────────────────────────────────────────────────────────────────────
# Disk space pre-flight
# ────────────────────────────────────────────────────────────────────

def _check_disk(output_dir: Path, estimated_bytes: int) -> bool:
    try:
        usage = shutil.disk_usage(output_dir)
        # Need estimated output + 10% headroom
        needed = int(estimated_bytes * 1.1)
        if usage.free < needed:
            print(f"FATAL: Insufficient disk space on {output_dir}",
                  file=sys.stderr)
            print(f"       Need ~{_bytes(needed)}, "
                  f"have {_bytes(usage.free)}", file=sys.stderr)
            return False
    except OSError:
        pass
    return True


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="lamquant compress",
        description="Lossless EDF → LML compression with full verification.",
    )
    p.add_argument("input", help="EDF file or directory")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-j", "--workers", type=int, default=None,
                   help="Parallel workers (default: auto from CPU/RAM)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--skip-existing", action="store_true", default=None)
    p.add_argument("--no-skip-existing", action="store_true", default=False)
    p.add_argument("--quiet", "-q", action="store_true", default=False)
    p.add_argument("--verbose", "-v", action="store_true", default=False)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--verify-outliers", action="store_true", default=None,
                   help="Verify best/worst files against original (default: on)")
    p.add_argument("--no-verify-outliers", action="store_true", default=False)
    p.add_argument("--no-color", action="store_true", default=False)
    p.add_argument("--force-color", action="store_true", default=False)
    p.add_argument("--no-dashboard", action="store_true", default=False)
    p.add_argument("--no-resume", action="store_true", default=False)
    p.add_argument("--fail-fast", action="store_true", default=None)
    p.add_argument("--refresh-hz", type=float, default=None)
    p.add_argument("--noise-bits", type=int, default=None,
                   choices=range(0, 16),
                   metavar='{0..15}',
                   help="Strip N LSBs before compression (0=lossless, 1-15=noise floor). "
                        "Set to (bit_depth - ENOB) for your hardware. Default from config.")
    args = p.parse_args(argv)

    # ── Config ──
    overrides = {}
    if args.workers is not None:
        overrides["compute.workers"] = args.workers
    if args.no_skip_existing:
        overrides["resume.skip_existing_output"] = False
    elif args.skip_existing:
        overrides["resume.skip_existing_output"] = True
    if args.no_verify_outliers:
        overrides["integrity.verify_outliers"] = False
    elif args.verify_outliers:
        overrides["integrity.verify_outliers"] = True
    if args.fail_fast is not None:
        overrides["integrity.fail_fast"] = args.fail_fast
    if args.refresh_hz is not None:
        overrides["output.refresh_hz"] = args.refresh_hz
    if args.noise_bits is not None:
        overrides["codec.noise_bits"] = args.noise_bits

    cfg = load_config(args.config, overrides)

    # Color
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
        C.reconfigure(False)
    elif args.force_color:
        C.reconfigure(True)

    # ── Discover files ──
    if os.path.isfile(args.input):
        edfs = [args.input]
    else:
        edfs = find_edf_files(args.input)

    if not edfs:
        print("FATAL: No EDF files found.", file=sys.stderr)
        return 5

    # ── Output dir ──
    single_file = os.path.isfile(args.input)
    output_dir = Path(args.output if not single_file
                      else os.path.dirname(args.output) or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config hash ──
    config_hash = cfg.hash()

    # ── State file (resume) ──
    state = None
    recovering = False
    if cfg.resume.enabled and not args.no_resume and not single_file:
        state = StateFile(output_dir, args.input, config_hash, sys.argv)
        if state.exists():
            if state.load():
                recovering = True
                zombies = state.recover_zombies()
                if not args.quiet:
                    print_recovery_summary(state, zombies)
        # Register all discovered files
        basenames = [os.path.basename(e) for e in edfs]
        state.register_files(basenames)

    # ── Build work list ──
    work = []
    skipped = 0
    total_input_bytes = 0

    for edf in edfs:
        out = (args.output if single_file
               else make_output_path(edf, args.output))
        bn = os.path.basename(edf)

        # Skip via state file — but verify output still exists
        if state and state.is_completed(bn):
            if os.path.exists(out):
                skipped += 1
                continue
            else:
                # State says done but output is gone — reset to pending
                state.files[bn].status = "pending"
        # Skip via output file existence
        if cfg.resume.skip_existing_output and os.path.exists(out):
            skipped += 1
            if state:
                state.mark_completed(bn, out, 0, 0, "")
            continue
        # Skip by state (pending/failed = process, others = skip)
        if state and not state.should_process(bn):
            skipped += 1
            continue

        sz = os.path.getsize(edf)
        total_input_bytes += sz
        work.append((edf, out))

    if not work:
        if not args.quiet:
            print(f"  All {skipped:,} files already converted.", file=sys.stdout)
        return 0

    # ── Disk pre-flight ──
    estimated_output = total_input_bytes  # worst case: no compression
    if not _check_disk(output_dir, estimated_output):
        return 1

    # ── Stats ──
    stats = RunStats(
        files_total=len(work),
        config_hash=config_hash,
    )
    stats.files_skipped = skipped
    results = []

    # ── Workers ──
    n_workers = cfg.effective_workers() if args.workers is None else max(1, args.workers)
    atomic = cfg.output_files.atomic_writes

    # ── Audit log ──
    audit = AuditLog(output_dir / cfg.logging.audit_log)
    audit.start(sys.argv, stats)
    audit.config(config_hash)
    audit.scan(len(work), total_input_bytes)

    # ── Banner ──
    if not args.quiet and cfg.output.show_banner:
        print_banner(stats, workers=n_workers, files_total=len(work),
                     input_path=args.input, output_path=args.output)

    # ── Dashboard vs piped vs quiet ──
    interactive = (sys.stdout.isatty()
                   and not args.quiet
                   and not args.no_dashboard
                   and _term_height() >= 20)
    dash = Dashboard(stats, cfg.output.refresh_hz) if interactive else None
    if dash:
        dash.set_paths(args.input, args.output)

    # ── Signal handling ──
    shutdown_requested = False
    sigint_count = 0
    sigint_time = 0.0

    def _handle_signal(signum, frame):
        nonlocal shutdown_requested, sigint_count, sigint_time
        now = time.time()

        if signum == signal.SIGINT:
            sigint_count += 1
            if sigint_count >= 2 and (now - sigint_time) < 5.0:
                # Double SIGINT within 5s — hard exit
                if dash:
                    dash.clear()
                if state:
                    state.flush()
                print(f"\n  {C.RED}Force quit.{C.RST}", file=sys.stderr)
                sys.exit(130)
            sigint_time = now
            exit_code = 3
        elif signum == signal.SIGTERM:
            exit_code = 4
        else:
            return

        shutdown_requested = True
        if dash:
            dash.clear()
        # Full clear to remove any partial dashboard artifacts
        if sys.stdout.isatty():
            sys.stdout.write("\033[2J\033[H")
        print(f"\n  {C.YEL}Interrupted. Finishing in-flight files...{C.RST}",
              file=sys.stdout)
        sys.stdout.flush()

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (AttributeError, OSError):
        pass

    # ── Process ──
    def _handle_result(idx: int, r: FileResult):
        stats.files_done += 1
        bn = os.path.basename(r.path)

        if r.status in ("ok", "warning"):
            stats.files_ok += 1
            stats.bytes_in += r.bytes_in
            stats.bytes_out += r.bytes_out
            stats.total_samples += r.n_samples
            if r.sample_rate > 0:
                stats.total_duration_s += r.n_samples / r.sample_rate
            audit.file_ok(idx, len(work), r)
            if state:
                state.mark_completed(bn, "", r.bytes_in, r.bytes_out, r.sha256)
            if r.status == "warning":
                stats.warnings.append(r.error or "expansion")
                print(f"WARNING: {bn}: {r.error}", file=sys.stderr)
        elif r.status == "error":
            stats.files_error += 1
            audit.file_error(idx, len(work), r.path, r.error or "unknown")
            if state:
                fs = state.files.get(bn)
                if fs and fs.attempts >= cfg.resume.max_retries:
                    qdir = str(output_dir / cfg.resume.quarantine_dir)
                    os.makedirs(qdir, exist_ok=True)
                    state.quarantine(bn, qdir)
                else:
                    state.mark_failed(bn, r.error or "unknown")
            if cfg.integrity.fail_fast:
                raise SystemExit(2)
            print(f"ERROR: {bn}: {r.error}", file=sys.stderr)

        results.append(r)

        # Checkpoint state (every 100 files or on error, not every file)
        if state and (stats.files_done % 100 == 0 or r.status != "ok"):
            state.flush()

        # Display
        if dash:
            dash.tick()
        elif not args.quiet:
            if args.verbose or stats.files_done % 500 == 0 or r.status != "ok":
                print_file_line(stats.files_done, len(work), r)

    try:
        if n_workers > 1:
            nb = cfg.codec.noise_bits
            vfy = cfg.integrity.verify_after_write
            work_args = [(edf, out, atomic, nb, vfy) for edf, out in work]
            with Pool(n_workers, initializer=_worker_init) as pool:
                for idx, r in enumerate(pool.imap_unordered(
                        _compress_one, work_args, chunksize=4), 1):
                    if shutdown_requested:
                        pool.terminate()
                        break
                    if dash:
                        dash.set_file(r.path)
                    _handle_result(idx, r)
        else:
            for idx, (edf, out) in enumerate(work, 1):
                if shutdown_requested:
                    break
                if dash:
                    dash.set_file(edf)
                    dash.tick(force=True)
                if state:
                    state.mark_in_progress(os.path.basename(edf))
                r = _compress_one((edf, out, atomic, cfg.codec.noise_bits,
                                   cfg.integrity.verify_after_write))
                _handle_result(idx, r)

    except SystemExit as e:
        if dash:
            dash.clear()
        exit_code = e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        if dash:
            dash.clear()
        exit_code = 3
    except Exception as e:
        if dash:
            dash.clear()
        print(f"FATAL: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        exit_code = 1
    else:
        exit_code = 0 if stats.files_error == 0 else 2
        if shutdown_requested:
            exit_code = 3

    # ── Restore signal handlers ──
    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTERM, old_sigterm)

    # ── Finalize ──
    if dash:
        dash.clear()
        # Full clear for clean summary output
        if sys.stdout.isatty():
            sys.stdout.write("\033[2J\033[H")

    if args.quiet:
        print_summary_json(stats)
    elif cfg.output.show_summary:
        print_summary(stats, output_dir, results=results, verbose=args.verbose)

    # Manifest + state
    manifest_path = output_dir / cfg.logging.manifest
    write_manifest(manifest_path, stats, results)
    if state:
        state.flush()

    # ── Outlier verification ──
    if cfg.integrity.verify_outliers and results and exit_code in (0, 2):
        ok_results = [r for r in results if r.status in ("ok", "warning") and r.ratio > 0]
        if len(ok_results) >= 2:
            ok_sorted = sorted(ok_results, key=lambda x: x.ratio)
            outliers = []
            outliers.append(("BEST ", ok_sorted[-1]))
            outliers.append(("WORST", ok_sorted[0]))
            if len(ok_sorted) >= 3:
                outliers.append(("2ND-W", ok_sorted[1]))

            print(f"\n  {C.BLD}Outlier Verification{C.RST} "
                  f"(decompressing and comparing against original EDF)",
                  file=sys.stdout)
            print(f"  {C.DIM}{B['h'] * 72}{C.RST}", file=sys.stdout)

            for label, r in outliers:
                bn = os.path.basename(r.path)
                # Find the LML file
                lml_path = make_output_path(r.path, args.output)
                if not os.path.exists(lml_path):
                    print(f"  {label}  {bn:<40s}  LML not found", file=sys.stdout)
                    continue

                # Decompress LML
                try:
                    from lamquant_codec.edf_to_lml import read_lml_file
                    import hashlib as _hl
                    recon, meta = read_lml_file(lml_path)
                    stored_sha = meta.get("signal_sha256", "")
                    recon_sha = _hl.sha256(recon.tobytes()).hexdigest()

                    if stored_sha and recon_sha == stored_sha:
                        ck = f"{C.GRN}{B['ok']} SHA-256 match{C.RST}"
                    elif stored_sha:
                        ck = f"{C.RED}{B['no']} SHA-256 MISMATCH{C.RST}"
                    else:
                        ck = f"{C.YEL}! no stored hash{C.RST}"

                    print(f"  {label}  {bn:<40s}  {_ratio(r.ratio)}  {ck}",
                          file=sys.stdout)
                except Exception as e:
                    print(f"  {label}  {bn:<40s}  {C.RED}ERROR: {str(e)[:40]}{C.RST}",
                          file=sys.stdout)

            print(file=sys.stdout)

    # Audit
    audit.summary(stats)
    audit.end(stats, exit_code)

    # Exit message for non-zero
    if exit_code == 2:
        print(f"\n  {C.RED}{B['no']}{C.RST} Run ended with errors.",
              file=sys.stdout)
        print(f"     Files failed:      {stats.files_error:,}", file=sys.stdout)
        print(f"     See audit.log for details.", file=sys.stdout)
        print(f"     Exit code: {exit_code} (partial success)", file=sys.stdout)
        print(file=sys.stdout)
    elif exit_code == 3:
        print(f"\n  {C.YEL}!{C.RST} Run interrupted. "
              f"Use --skip-existing to resume.", file=sys.stdout)
        print(f"     Exit code: {exit_code}", file=sys.stdout)
        print(file=sys.stdout)

    return exit_code


def _term_height() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).lines
    except Exception:
        return 24


if __name__ == "__main__":
    sys.exit(main())
