"""Unified `lamquant` command-line interface.

Dispatches to existing module entry points and shell helpers so that the
whole project can be driven from a single binary installed via
``pip install lamquant``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


from lamquant_codec._paths import REPO_ROOT as PROJECT_ROOT


def _project_root() -> Path:
    return PROJECT_ROOT


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess inheriting stdio. Returns the exit code."""
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def _cmd_setup(args: argparse.Namespace, _extra: list[str]) -> int:
    from lamquant_codec import setup_cmd
    result = setup_cmd.run(yes=args.yes)
    if result == 0:
        from lamquant_codec.cli import menu
        menu.migrate_history_to_current()
    return result


def _cmd_download(_args: argparse.Namespace, extra: list[str]) -> int:
    """``lamquant download`` — sequestered shim (ADR 0018).

    The handler implementation moved to ``legacy/cli_download/`` so
    the codec wheel is pure codec. The CLI entry point stays callable
    in a full repo checkout (no surprise removals for dev workflows);
    invocation emits a ``DeprecationWarning`` and delegates to the
    sequestered module. A future sprint promotes this to a dedicated
    training-side CLI (`blut data download` or `lamquant train
    download`).

    NOTE: ``legacy/`` is repo-root, NOT inside the ``lamquant_codec``
    wheel. On a ``pip install lamquant_codec`` deployment the import
    below fails — the verb is intentionally repo-only training tooling
    and is expected to disappear from the codec CLI in a follow-up
    sprint. The ImportError path emits a clean error pointing the
    user at the training-side replacement instead of a raw traceback.
    """
    import warnings
    warnings.warn(
        "`lamquant download` is training tooling and has moved to "
        "legacy/cli_download/. The CLI entry stays for now but will be "
        "promoted to a training-side verb in a future sprint. See ADR 0018.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from legacy.cli_download.download_handler import run
    except ImportError:
        print(
            "error: `lamquant download` is only available in a full repo\n"
            "       checkout (legacy/cli_download/ is not part of the\n"
            "       lamquant_codec wheel). Run from the LamQuant source\n"
            "       tree, or wait for the training-side CLI replacement.\n"
            "       See ADR 0018.",
            file=sys.stderr,
        )
        return 1
    return run(extra)


def _cmd_init(args: argparse.Namespace, _extra: list[str]) -> int:
    """Run the first-run wizard."""
    from lamquant_codec import init_wizard
    result = init_wizard.run(force=args.force, quiet=args.quiet)
    if args.as_json:
        import json
        # Strip non-JSON-serialisable fields.
        out = {
            'initialised': result.initialised,
            'skipped': result.skipped,
            'sections': result.sections,
        }
        print(json.dumps(out, indent=2, default=str))
    return 0 if (result.initialised or result.skipped) else 1


def _cmd_doctor(_args: argparse.Namespace, _extra: list[str]) -> int:
    """Run the install validator (Python imports, Rust binary, npm, smoke tests)."""
    root = _project_root()
    sys.path.insert(0, str(root / "installer"))
    try:
        validator = importlib.import_module("install_validate")
    except ImportError as exc:
        print(f"error: cannot import install_validate: {exc}", file=sys.stderr)
        return 1
    main = getattr(validator, "main", None)
    if main is None:
        print("error: install_validate.main not found", file=sys.stderr)
        return 1
    main()
    return 0


def _forward(module: str, script_name: str, extra: list[str]) -> int:
    """Forward to an existing module's ``main()`` by setting argv."""
    mod = importlib.import_module(module)
    main = getattr(mod, "main", None)
    if main is None:
        print(f"error: {module}.main not found", file=sys.stderr)
        return 1
    saved_argv = sys.argv
    sys.argv = [script_name, *extra]
    try:
        rc = main()
    finally:
        sys.argv = saved_argv
    return int(rc) if isinstance(rc, int) else 0


def _cmd_decode(_args: argparse.Namespace, extra: list[str]) -> int:
    return _forward("lamquant_codec.cli_codec", "lamquant-decode", extra)


# ============================================================
# Batch operations: compress / decompress / validate / info
# ============================================================

def _add_batch_flags(p: argparse.ArgumentParser) -> None:
    """Common flags shared by every batch subcommand."""
    p.add_argument('inputs', nargs='+',
                   help="Input file(s), directory, glob pattern, or '-' for stdin")
    p.add_argument('-o', '--output', dest='output_dir', default='.',
                   help='Output directory (default: current)')
    p.add_argument('-r', '--recursive', action='store_true',
                   help='Recurse into input directories')
    p.add_argument('-w', '--workers', type=int, default=None,
                   help='Parallel workers (default: cpu_count)')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip files whose output already exists (default on)')
    p.add_argument('--no-skip-existing', dest='skip_existing',
                   action='store_false',
                   help='Always re-process, even if output exists')
    p.add_argument('--manifest', metavar='FILE',
                   help='Write a per-file CSV manifest (success / failed / cr / error)')
    p.add_argument('--from-manifest', metavar='FILE',
                   help='Resume from a previous manifest — skip already-succeeded files')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress progress bars (for cron / scripts)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be done without doing it')
    p.add_argument('--json', dest='as_json', action='store_true',
                   help='Print summary as JSON instead of human format')


def _cmd_compress(args: argparse.Namespace, _extra: list[str]) -> int:
    from lamquant_codec.batch import compress_batch
    report = compress_batch(
        inputs=args.inputs,
        output_dir=args.output_dir,
        mode=args.mode,
        quality={'alerting': 0, 'monitoring': 1, 'clinical': 2}[args.quality],
        checkpoint=args.checkpoint,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        from_manifest=args.from_manifest,
        manifest=args.manifest,
        workers=args.workers,
        quiet=args.quiet,
        dry_run=args.dry_run,
    )
    if args.as_json:
        import json
        print(json.dumps(report.summary(), indent=2))
    else:
        report.print_summary()
    return 0 if report.n_failed == 0 else 1


def _cmd_decompress(args: argparse.Namespace, _extra: list[str]) -> int:
    from lamquant_codec.batch import decompress_batch
    report = decompress_batch(
        inputs=args.inputs,
        output_dir=args.output_dir,
        output_ext=args.output_ext,
        checkpoint=args.checkpoint,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        from_manifest=args.from_manifest,
        manifest=args.manifest,
        workers=args.workers,
        quiet=args.quiet,
        dry_run=args.dry_run,
    )
    if args.as_json:
        import json
        print(json.dumps(report.summary(), indent=2))
    else:
        report.print_summary()
    return 0 if report.n_failed == 0 else 1


def _cmd_validate(args: argparse.Namespace, _extra: list[str]) -> int:
    """LQS quality compliance check across an archive."""
    from lamquant_codec.batch import validate_batch
    report = validate_batch(
        inputs=args.inputs,
        reference_dir=args.reference,
        level=args.level,
        checkpoint=args.checkpoint,
        recursive=args.recursive,
        manifest=args.manifest,
        workers=args.workers,
        quiet=args.quiet,
    )
    if args.report_html:
        from lamquant_codec.report import write_html_report
        path = write_html_report(report, args.report_html, level=args.level)
        if not args.quiet:
            print(f"\n  HTML report written to: {path}")

    if args.as_json:
        import json
        print(json.dumps(report.summary(), indent=2))
    else:
        report.print_summary()
        # Per-level pass rate (if quality was measured).
        with_quality = [r for r in report.results if r.lqs_pass is not None]
        if with_quality:
            n_pass = sum(1 for r in with_quality if r.lqs_pass)
            print(f"\n  LQS Level {args.level}: {n_pass}/{len(with_quality)} pass "
                  f"({100*n_pass/len(with_quality):.1f}%)")
            if n_pass < len(with_quality):
                print("  Top failures (by PRD):")
                fails = sorted([r for r in with_quality if not r.lqs_pass],
                               key=lambda r: -(r.prd or 0))[:5]
                for r in fails:
                    print(f"    {r.input_path}: PRD={r.prd:.1f}%, R={r.pearson_r:.4f}")
    return 0 if report.n_failed == 0 else 1


def _cmd_verify(args: argparse.Namespace, _extra: list[str]) -> int:
    """Structural / CRC integrity check across many files."""
    from lamquant_codec.batch import verify_batch
    report = verify_batch(
        inputs=args.inputs,
        decode=args.decode,
        checkpoint=args.checkpoint,
        recursive=args.recursive,
        manifest=args.manifest,
        workers=args.workers,
        quiet=args.quiet,
    )
    if args.as_json:
        import json
        print(json.dumps(report.summary(), indent=2))
    else:
        report.print_summary()
        if not args.quiet and report.n_success and not args.report_html:
            # Surface the per-file 'check' summary stored in output_path.
            print()
            for r in report.results:
                if r.status == 'success':
                    print(f"  ✓ {r.input_path}: {r.output_path}")
    if args.report_html:
        from lamquant_codec.report import write_html_report
        path = write_html_report(report, args.report_html, title='LamQuant Integrity Report')
        if not args.quiet:
            print(f"\n  HTML report written to: {path}")
    return 0 if report.n_failed == 0 else 1


def _cmd_info(args: argparse.Namespace, _extra: list[str]) -> int:
    from lamquant_codec.batch import info_batch
    metas = info_batch(args.inputs, recursive=args.recursive)
    if args.as_json:
        import json
        print(json.dumps(metas, indent=2, default=str))
    else:
        for m in metas:
            path = m.pop('input_path')
            print(f"\n{path}")
            for k, v in m.items():
                if v is not None:
                    print(f"  {k}: {v}")
    return 0


def _cmd_verify_manifest(args: argparse.Namespace, _extra: list[str]) -> int:
    """Verify a manifest.lml.json: check existence, size, SHA-256 of each file."""
    import hashlib
    import json
    import struct

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_dir = manifest_path.parent
    files = manifest.get('files', [])
    total = len(files)
    passed = 0
    failed = 0
    missing = 0

    for entry in files:
        out_rel = entry.get('output', '')
        expected_size = entry.get('compressed_bytes', 0)
        expected_sha = entry.get('sha256', '')

        full_path = manifest_dir / out_rel
        if not full_path.exists():
            print(f"  MISSING {out_rel}")
            missing += 1
            continue

        # Check file size
        actual_size = full_path.stat().st_size
        if expected_size and actual_size != expected_size:
            print(f"  SIZE MISMATCH {out_rel}: expected {expected_size} got {actual_size}")
            failed += 1
            continue

        # Decode and verify SHA-256
        if expected_sha:
            try:
                from lamquant_codec.edf_to_lml import read_lml_file
                signal, _meta = read_lml_file(str(full_path))
                import numpy as np
                hasher = hashlib.sha256()
                for ch in range(signal.shape[0]):
                    hasher.update(signal[ch].astype(np.int64).tobytes())
                actual_sha = hasher.hexdigest()
                if actual_sha != expected_sha:
                    print(f"  SHA256 MISMATCH {out_rel}: expected {expected_sha[:16]}.. got {actual_sha[:16]}..")
                    failed += 1
                    continue
            except Exception as e:
                print(f"  DECODE FAIL {out_rel}: {e}")
                failed += 1
                continue

        passed += 1

    print(f"\nManifest verification: {passed}/{total} passed, {failed} failed, {missing} missing")
    return 1 if (failed > 0 or missing > 0) else 0


def _cmd_stats(args: argparse.Namespace, _extra: list[str]) -> int:
    """Show per-channel signal statistics for LML/LMQ file(s)."""
    import numpy as np
    from lamquant_codec.batch import expand_inputs, DECOMPRESS_EXTS

    files = expand_inputs(args.inputs, recursive=args.recursive, exts=DECOMPRESS_EXTS)
    if not files:
        print("error: no LML/LMQ files found", file=sys.stderr)
        return 1

    is_dir = len(files) > 1

    if is_dir:
        # CSV summary mode
        print("file,channels,samples,duration_s,sample_rate,file_bytes")
        for f in files:
            try:
                from lamquant_codec.edf_to_lml import read_lml_file
                signal, meta = read_lml_file(str(f))
                n_ch = signal.shape[0]
                t = signal.shape[1] if signal.ndim == 2 else 0
                sr = meta.get('sample_rate', 250)
                duration = t / sr if sr else 0
                file_size = f.stat().st_size
                print(f"{f},{n_ch},{t},{duration:.1f},{sr},{file_size}")
            except Exception as e:
                print(f"# FAIL {f}: {e}", file=sys.stderr)
    else:
        # Single file detailed stats
        f = files[0]
        try:
            from lamquant_codec.edf_to_lml import read_lml_file
            signal, meta = read_lml_file(str(f))
        except Exception as e:
            print(f"error: cannot read {f}: {e}", file=sys.stderr)
            return 1

        n_ch = signal.shape[0]
        t = signal.shape[1] if signal.ndim == 2 else 0
        sr = meta.get('sample_rate', 250)
        duration = t / sr if sr else 0
        file_size = f.stat().st_size
        raw_size = n_ch * t * 2

        print(f"File:        {f}")
        print(f"Channels:    {n_ch}")
        print(f"Samples:     {t} ({duration:.1f}s @ {sr:.0f} Hz)")
        print(f"File size:   {file_size} bytes ({raw_size / max(file_size,1):.2f}:1 CR)")
        print()
        print(f"{'Ch':<6} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'Samples':>12} {'NoiseFloor':>12}")
        print("-" * 84)

        for i in range(n_ch):
            ch = signal[i].astype(np.float64)
            if ch.size == 0:
                continue
            min_v = int(np.min(ch))
            max_v = int(np.max(ch))
            mean_v = float(np.mean(ch))
            std_v = float(np.std(ch))
            # MAD noise floor estimate
            median_v = float(np.median(ch))
            mad = float(np.median(np.abs(ch - median_v)))
            print(f"{i:<6} {min_v:>12} {max_v:>12} {mean_v:>12.1f} {std_v:>12.1f} {ch.size:>12} {mad:>12.1f}")

    return 0


def _cmd_export(_args: argparse.Namespace, extra: list[str]) -> int:
    return _forward("lamquant_codec.export", "lamquant-export", extra)


def _cmd_visualize(_args: argparse.Namespace, extra: list[str]) -> int:
    root = _project_root()
    sys.path.insert(0, str(root))
    return _forward("tools.visualize_pipeline", "lamquant-visualize", extra)


def _cmd_train(_args: argparse.Namespace, extra: list[str]) -> int:
    """Launch training cockpit (interactive or CLI-driven)."""
    root = _project_root()
    cockpit = root / "training_cockpit.py"
    if not cockpit.exists():
        print(f"error: {cockpit} not found", file=sys.stderr)
        return 1
    return _run([sys.executable, str(cockpit), *extra], cwd=root)


def _cmd_gui(_args: argparse.Namespace, _extra: list[str]) -> int:
    """Launch OpenHuman Vision (Tauri release binary or dev mode)."""
    root = _project_root()
    candidates = [
        root / "gui" / "src-tauri" / "target" / "release" / "lamquant-gui",
        root / "gui" / "src-tauri" / "target" / "release" / "lamquant-gui.exe",
    ]
    for binary in candidates:
        if binary.exists():
            print(f"Launching OpenHuman Vision: {binary}")
            return _run([str(binary)])
    print("Release binary not found; falling back to dev mode (npx tauri dev)...")
    gui_dir = root / "gui"
    if not gui_dir.exists():
        print(f"error: {gui_dir} not found", file=sys.stderr)
        return 1
    return _run(["npx", "tauri", "dev"], cwd=gui_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lamquant",
        description="LamQuant — EEG neural codec CLI (OpenHuman Portal / Vision / Eagle)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p_setup = sub.add_parser("setup", help="Install dependencies via OpenHuman Portal (wraps install.sh / install.ps1)")
    p_setup.add_argument("--yes", "-y", action="store_true", help="Non-interactive: install everything")
    p_setup.set_defaults(func=_cmd_setup)

    p_dl = sub.add_parser("download", help="Download datasets")
    p_dl.set_defaults(func=_cmd_download)

    p_doc = sub.add_parser("doctor", help="Run installation validator (Python imports, Rust binary, npm)")
    p_doc.set_defaults(func=_cmd_doctor)

    p_init = sub.add_parser(
        "init",
        help="First-run wizard (hardware detect + smoke test + next-step hints)",
        description="Runs once on a fresh install. Detects CPU/GPU/RAM, verifies "
                    "dependencies, runs an end-to-end lossless smoke test, and "
                    "prints the most useful CLI commands. Idempotent — re-runs "
                    "are no-ops unless --force is passed.",
    )
    p_init.add_argument('--force', action='store_true',
                        help='Re-run even if already initialised')
    p_init.add_argument('--quiet', action='store_true',
                        help='Suppress prompts (for CI / scripted installs)')
    p_init.add_argument('--json', dest='as_json', action='store_true',
                        help='Emit machine-readable JSON instead of human output')
    p_init.set_defaults(func=_cmd_init)

    p_dec = sub.add_parser("decode", help="Encode/decode EEG files (forwards to lamquant-decode)",
                           add_help=False)
    p_dec.set_defaults(func=_cmd_decode)

    # ---------- Batch commands ----------
    p_compress = sub.add_parser(
        "compress",
        help="Compress one or many EEG files (.edf / .npy → .lmq / .lml)",
        description="Batch compression with per-file manifest, parallel workers, "
                    "tree mirroring, and resume support.",
    )
    _add_batch_flags(p_compress)
    p_compress.add_argument('--mode', choices=['neural', 'lossless'],
                            default='lossless',
                            help='Codec mode (default: lossless — no checkpoint needed)')
    p_compress.add_argument('--quality', choices=['alerting', 'monitoring', 'clinical'],
                            default='clinical',
                            help='Neural quality mode (default: clinical)')
    p_compress.add_argument('-c', '--checkpoint',
                            help='Encoder checkpoint (required for --mode neural)')
    p_compress.set_defaults(func=_cmd_compress)

    p_decompress = sub.add_parser(
        "decompress",
        help="Decompress .lmq / .lml files in parallel",
    )
    _add_batch_flags(p_decompress)
    p_decompress.add_argument('--output-ext', default='.npy',
                              help='Output file extension (.npy or .csv, default: .npy)')
    p_decompress.add_argument('-c', '--checkpoint',
                              help='Encoder checkpoint (required for .lmq inputs)')
    p_decompress.set_defaults(func=_cmd_decompress)

    p_validate = sub.add_parser(
        "validate",
        help="Validate compressed files against an LQS quality level (PRD/R/CR)",
        description="Decompress files and (with --reference) compare against "
                    "originals. Generates a CSV manifest and optional HTML "
                    "dashboard suitable for IRB / hospital IT review.",
    )
    _add_batch_flags(p_validate)
    p_validate.add_argument('--level', choices=['A', 'M', 'C', 'L'],
                            default='C', help='LQS level (default: C — clinical)')
    p_validate.add_argument('--reference', metavar='DIR',
                            help='Directory containing the original .npy/.edf '
                                 'files (required for quality metrics)')
    p_validate.add_argument('-c', '--checkpoint',
                            help='Encoder checkpoint (required for .lmq inputs)')
    p_validate.add_argument('--report-html', metavar='FILE',
                            help='Write self-contained HTML dashboard '
                                 '(charts + per-file table)')
    p_validate.set_defaults(func=_cmd_validate)

    p_verify = sub.add_parser(
        "verify",
        help="Structural / CRC integrity check (no quality metrics)",
        description="Walk each file's headers and (for the LQN1/LQL1 wrapper "
                    "format) verify CRC32 on every window. Catches truncation, "
                    "header corruption, CRC mismatches. Add --decode for a full "
                    "round-trip integrity test.",
    )
    _add_batch_flags(p_verify)
    p_verify.add_argument('--decode', action='store_true',
                          help='Also run a full decompress round-trip per file '
                               '(catches subtle payload corruption that CRC misses)')
    p_verify.add_argument('-c', '--checkpoint',
                          help='Encoder checkpoint (required for --decode on .lmq)')
    p_verify.add_argument('--report-html', metavar='FILE',
                          help='Write self-contained HTML dashboard')
    p_verify.set_defaults(func=_cmd_verify)

    p_info = sub.add_parser(
        "info",
        help="Show .lmq / .lml file metadata (no decode)",
    )
    p_info.add_argument('inputs', nargs='+',
                        help='Input file(s), directory, glob pattern')
    p_info.add_argument('-r', '--recursive', action='store_true')
    p_info.add_argument('--json', dest='as_json', action='store_true')
    p_info.set_defaults(func=_cmd_info)

    # ---------- New commands: verify-manifest, stats ----------
    p_vman = sub.add_parser(
        "verify-manifest",
        help="Verify a manifest.lml.json: check file existence, size, and SHA-256",
    )
    p_vman.add_argument('manifest', help='Path to manifest.lml.json')
    p_vman.set_defaults(func=_cmd_verify_manifest)

    p_stats = sub.add_parser(
        "stats",
        help="Show per-channel signal statistics for LML/LMQ file(s)",
    )
    p_stats.add_argument('inputs', nargs='+',
                         help='Input file(s), directory, glob pattern')
    p_stats.add_argument('-r', '--recursive', action='store_true',
                         help='Recurse into subdirectories')
    p_stats.set_defaults(func=_cmd_stats)

    p_exp = sub.add_parser("export", help="Export model to firmware headers (forwards to lamquant-export)",
                           add_help=False)
    p_exp.set_defaults(func=_cmd_export)

    p_viz = sub.add_parser("visualize", help="Generate diagnostic plots (forwards to lamquant-visualize)",
                           add_help=False)
    p_viz.set_defaults(func=_cmd_visualize)

    p_train = sub.add_parser("train", help="Training pipeline (interactive cockpit)",
                             add_help=False)
    p_train.set_defaults(func=_cmd_train)

    p_gui = sub.add_parser("gui", help="Launch OpenHuman Vision desktop GUI")
    p_gui.set_defaults(func=_cmd_gui)

    return parser


def main() -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args()
    return args.func(args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
