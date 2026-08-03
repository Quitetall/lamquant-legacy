"""Compression backend status and product-operation broker.

Configured Python and custom implementations remain visible for diagnostics,
library calls, and research. Product TUI encode/decode always cross the Rust
``lamquant op run`` broker so checked history, canonical operation identity,
compiled containment, cancellation, and executor-issued receipts cannot be
bypassed by one frontend.

Custom product execution stays fail-closed until a signed capability manifest
and versioned control protocol are verified by the broker.
"""
import os
import shutil
import signal
import subprocess
import sys
from typing import Optional


def detect_backend(config) -> str:
    """Detect and return the active backend name: 'rust', 'python', or 'custom'."""
    return config.backend.resolve()


def get_backend_version(config) -> str:
    """Get the version string of the active backend."""
    backend = config.backend.resolve()
    if backend == "rust":
        binary = _resolve_binary(config)
        if binary:
            try:
                r = subprocess.run([binary, "--version"],
                                   capture_output=True, text=True, timeout=5)
                return r.stdout.strip() if r.returncode == 0 else "unknown"
            except Exception:
                return "unknown"
    elif backend == "custom":
        binary = config.backend.custom_binary
        try:
            r = subprocess.run([binary, "--version"],
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    else:
        from lamquant_codec import __version__
        return f"lamquant-codec {__version__} (Python)"
    return "unknown"


def run_encode(config, input_path: str, output_path: str, *,
               recursive: bool = True, verify: bool = True,
               skip_existing: bool = True, workers: int = 0,
               noise_bits: int = 0) -> int:
    """Run canonical ``encode_lma`` through the Rust operation broker."""
    if not recursive or not verify or not skip_existing or workers != 0 or noise_bits != 0:
        print(
            "error: Python TUI accepts only the canonical encode_lma contract; "
            "use the non-product codec CLI for experimental overrides",
            file=sys.stderr,
        )
        return 2
    return _run_product_operation(config, "encode_lma", input_path, output_path)


def run_decode(config, input_path: str, output_path: str, *,
               recursive: bool = True, skip_existing: bool = True,
               workers: int = 0) -> int:
    """Run canonical ``decode`` through the Rust operation broker."""
    if not recursive or not skip_existing or workers != 0:
        print(
            "error: Python TUI accepts only the canonical decode contract; "
            "use the non-product codec CLI for experimental overrides",
            file=sys.stderr,
        )
        return 2
    return _run_product_operation(config, "decode", input_path, output_path)


def _run_product_operation(
    config, operation: str, input_path: str, output_path: str
) -> int:
    backend = config.backend.resolve()
    if backend == "python":
        print(
            "error: Python TUI product execution requires the Rust operation broker; "
            "Python codec functions remain available as library/research APIs",
            file=sys.stderr,
        )
        return 2
    if backend == "custom":
        print(
            "error: custom product backend refused without a broker-verified "
            "signed capability manifest and versioned control protocol",
            file=sys.stderr,
        )
        return 2
    broker = _resolve_broker()
    if not broker:
        print(
            "error: LamQuant operation broker not found; install the `lamquant` "
            "or `lq` product binary",
            file=sys.stderr,
        )
        return 1
    command = [
        broker,
        "op",
        "run",
        operation,
        "--input",
        input_path,
        "--output",
        output_path,
    ]
    process = None
    try:
        process = subprocess.Popen(command)
        return process.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        if process is not None:
            _stop_broker(process, interrupt=False)
        print("error: operation broker timed out after 2 hours", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"error: operation broker not found: {broker}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if process is not None:
            _stop_broker(process, interrupt=True)
        return 130


def _stop_broker(process, *, interrupt: bool) -> None:
    """Request exact broker cancellation, then escalate after a bounded grace."""
    if process.poll() is not None:
        return
    try:
        if interrupt and os.name != "nt":
            process.send_signal(signal.SIGINT)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait()


def _resolve_binary(config) -> Optional[str]:
    """Resolve the Rust binary path from config."""
    from lamquant_codec.cli.config import _find_rust_binary
    return _find_rust_binary(config.backend.rust_binary)


def _resolve_broker() -> Optional[str]:
    """Resolve the product operation broker, preferring long-form name."""
    return shutil.which("lamquant") or shutil.which("lq")
