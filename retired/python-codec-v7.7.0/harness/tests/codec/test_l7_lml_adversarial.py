"""Adversarial test suite for LML lossless codec — clinical-grade verification.

Tests every known failure mode, edge case, and attack vector.
Both Python and Rust paths are tested for cross-language consistency.

AUDIT NOTE (2026-04-27): Converted from standalone script to proper pytest suite.
Previous version used a module-level `run_test()` wrapper that:
  1. Executed ALL tests at import time (before pytest collection)
  2. Caught ALL exceptions silently into a `results` list
  3. Only printed pass/fail to stdout — pytest never saw failures
  4. Created/destroyed temp dirs at import time, outside pytest isolation

Now:
  - Every test is a proper pytest method with fixture-managed temp dirs
  - Failures propagate to pytest as AssertionError (visible in CI)
  - Rust binary absence causes skip (not silent pass)
  - subprocess failures include stdout/stderr in assertion messages
"""
import sys, os, struct, hashlib, tempfile, subprocess, shutil
from pathlib import Path
import numpy as np
import pytest

# AUDIT (2026-04-28): Replaced hardcoded /mnt/4tb/LamQuant with relative path.
# The old absolute paths broke CI and other dev machines.
# Updated 2026-05-05: file moved tests/ → tests/codec/, so resolve up two levels.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

RUST_BIN = str(_REPO / 'target' / 'release' / 'lml')

_HAVE_RUST_BIN = os.path.exists(RUST_BIN)
_skip_no_rust = pytest.mark.skipif(
    not _HAVE_RUST_BIN,
    reason=f"Rust binary not found at {RUST_BIN} — build with `cargo build --release`"
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rust_archive_extract(src_dir, verify=True):
    """Archive with Rust, extract with Rust, return extracted dir path.

    AUDIT: Previously used check=True which raised CalledProcessError —
    a valid exception, but it was swallowed by run_test(). Now we assert
    explicitly so the failure message includes stdout/stderr for diagnosis.
    """
    lma = tempfile.mktemp(suffix='.lma')
    out = tempfile.mkdtemp()
    result = subprocess.run(
        [RUST_BIN, 'archive', src_dir, '-o', lma],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"Rust archive failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    cmd = [RUST_BIN, 'extract', lma, '-o', out]
    if verify:
        cmd.append('--verify')
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Rust extract failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    os.unlink(lma)
    return out


def compare_dirs(orig, restored):
    """Compare two directories: content, size, mtime. Returns list of failures."""
    failures = []
    orig_files = set()
    for root, dirs, files in os.walk(orig):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, orig)
            orig_files.add(rel)
            extr = os.path.join(restored, rel)

            if not os.path.exists(extr):
                failures.append(f"MISSING: {rel}")
                continue

            o_data = open(full, 'rb').read()
            e_data = open(extr, 'rb').read()
            if o_data != e_data:
                failures.append(f"CONTENT: {rel} (orig={len(o_data)}B sha={sha256(o_data)[:12]}, "
                              f"extr={len(e_data)}B sha={sha256(e_data)[:12]})")
                continue

            o_mtime = os.stat(full).st_mtime
            e_mtime = os.stat(extr).st_mtime
            if abs(o_mtime - e_mtime) > 1.0:  # 1-second tolerance for filesystem
                failures.append(f"MTIME: {rel} (orig={o_mtime}, extr={e_mtime})")

    # Check for extra files in restored
    for root, dirs, files in os.walk(restored):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), restored)
            if rel not in orig_files:
                failures.append(f"EXTRA: {rel}")

    return failures


def create_edf(path, n_channels=21, n_records=10, sample_rate=250,
               samples=None, patient_id="Test Patient", is_bdf=False,
               annotation_channels=0, extra_trailing=b''):
    """Create a synthetic EDF/BDF file with exact control over every byte."""
    bps = 3 if is_bdf else 2
    ns_per_rec = sample_rate  # 1-second records
    n_signals = n_channels + annotation_channels

    # Main header (256 bytes)
    hdr = bytearray(256)
    if is_bdf:
        hdr[0:1] = b'\xff'
        hdr[1:8] = b'BIOSEMI'
    else:
        hdr[0:8] = b'0       '
    hdr[8:88] = f'{patient_id:<80s}'.encode('ascii')[:80]
    hdr[88:168] = f'{"Startdate 01-JAN-2024 Test":<80s}'.encode('ascii')[:80]
    hdr[168:176] = b'01.01.24'
    hdr[176:184] = b'00.00.00'
    total_hdr = 256 + 256 * n_signals
    hdr[184:192] = f'{total_hdr:<8d}'.encode('ascii')[:8]
    hdr[192:236] = b'EDF+C' + b' ' * 39
    hdr[236:244] = f'{n_records:<8d}'.encode('ascii')[:8]
    hdr[244:252] = f'{"1":<8s}'.encode('ascii')[:8]
    hdr[252:256] = f'{n_signals:<4d}'.encode('ascii')[:4]

    # Signal headers
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    sig_hdr = bytearray(256 * n_signals)

    for fi, w in enumerate(widths):
        for si in range(n_signals):
            off = sum(widths[:fi]) * n_signals + si * w
            if fi == 0:  # label
                if si >= n_channels:
                    val = 'EDF Annotations'
                else:
                    val = f'EEG Ch{si}'
            elif fi == 2:  # physical dimension
                val = 'uV'
            elif fi == 3:  # physical min
                val = '-3200' if not is_bdf else '-3200000'
            elif fi == 4:  # physical max
                val = '3200' if not is_bdf else '3200000'
            elif fi == 5:  # digital min
                val = '-32768' if not is_bdf else '-8388608'
            elif fi == 6:  # digital max
                val = '32767' if not is_bdf else '8388607'
            elif fi == 8:  # ns_per_rec
                val = str(ns_per_rec) if si < n_channels else str(ns_per_rec // 2)
            else:
                val = ''
            sig_hdr[off:off+w] = f'{val:<{w}s}'.encode('ascii')[:w]

    # Data records
    if samples is None:
        # Generate realistic-looking EEG
        rng = np.random.RandomState(42)
        samples = np.zeros((n_channels, ns_per_rec * n_records), dtype=np.int64)
        for ch in range(n_channels):
            t = np.arange(ns_per_rec * n_records) / sample_rate
            signal = (100 * np.sin(2 * np.pi * 10 * t + ch) +
                     rng.randn(len(t)) * 50).astype(np.int64)
            if is_bdf:
                signal = np.clip(signal, -8388608, 8388607)
            else:
                signal = np.clip(signal, -32768, 32767)
            samples[ch] = signal

    data = bytearray()
    for r in range(n_records):
        for ch in range(n_channels):
            chunk = samples[ch, r*ns_per_rec:(r+1)*ns_per_rec]
            if is_bdf:
                for v in chunk:
                    v = int(v)
                    if v < 0: v += (1 << 24)
                    data.extend(struct.pack('<I', v)[:3])
            else:
                data.extend(chunk.astype(np.int16).tobytes())
        # Annotation channels (zeros)
        for ac in range(annotation_channels):
            data.extend(b'\x00' * (ns_per_rec // 2) * bps)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(sig_hdr)
        f.write(data)
        f.write(extra_trailing)


# ============================================================
# FIXTURES — proper pytest lifecycle for temp dirs
# ============================================================

@pytest.fixture
def edf_dir():
    """Create and clean up a temp directory for EDF test files.

    AUDIT: Replaces ad-hoc tempfile.mkdtemp + shutil.rmtree in each test.
    pytest manages the lifecycle so cleanup happens even on test failure.
    """
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 1. Standard format roundtrips
# ============================================================

@_skip_no_rust
class TestStandardFormats:
    """Standard EDF/BDF roundtrip through Rust archive/extract."""

    def test_standard_edf(self, edf_dir):
        """21ch, 250Hz, 10s standard EDF roundtrip."""
        create_edf(os.path.join(edf_dir, 'test.edf'))
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Standard EDF roundtrip failures: {fails}"

    def test_bdf_24bit(self, edf_dir):
        """32ch, 512Hz BDF (24-bit) roundtrip."""
        create_edf(os.path.join(edf_dir, 'test.bdf'), is_bdf=True, n_channels=32, sample_rate=512)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"BDF 24-bit roundtrip failures: {fails}"

    def test_edf_with_annotations(self, edf_dir):
        """EDF with annotation channel roundtrip."""
        create_edf(os.path.join(edf_dir, 'annotated.edf'), annotation_channels=1)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"EDF annotation roundtrip failures: {fails}"

    def test_trailing_partial_record(self, edf_dir):
        """EDF with 500 bytes trailing data — must be preserved byte-exact."""
        create_edf(os.path.join(edf_dir, 'trailing.edf'), extra_trailing=b'\x42' * 500)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"EDF trailing data roundtrip failures: {fails}"


# ============================================================
# 2. Boundary signal values
# ============================================================

@_skip_no_rust
class TestBoundaryValues:
    """Boundary signal values that stress integer arithmetic in the codec."""

    def test_edf_boundary_values(self, edf_dir):
        """int16 min/max, alternating, impulse — worst case for delta coding."""
        n_ch, n_rec, sr = 4, 5, 250
        samples = np.zeros((n_ch, sr * n_rec), dtype=np.int64)
        samples[0, :] = -32768  # all min
        samples[1, :] = 32767   # all max
        samples[2, :] = np.tile([-32768, 32767], sr * n_rec // 2)  # alternating
        samples[3, :] = 0; samples[3, sr*2] = 32767  # impulse
        create_edf(os.path.join(edf_dir, 'boundary.edf'), n_channels=n_ch, n_records=n_rec,
                   sample_rate=sr, samples=samples)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"EDF boundary value failures: {fails}"

    def test_bdf_boundary_values(self, edf_dir):
        """BDF int24 min (-8388608) / max (8388607)."""
        n_ch, n_rec, sr = 2, 3, 256
        samples = np.zeros((n_ch, sr * n_rec), dtype=np.int64)
        samples[0, :] = -8388608  # BDF min
        samples[1, :] = 8388607   # BDF max
        create_edf(os.path.join(edf_dir, 'bdf_boundary.bdf'), n_channels=n_ch, n_records=n_rec,
                   sample_rate=sr, samples=samples, is_bdf=True)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"BDF boundary value failures: {fails}"

    def test_negative_mean_signal(self, edf_dir):
        """Negative-mean ramp — tests bias floor division (found bug 2026-04)."""
        n_ch, n_rec, sr = 8, 10, 250
        samples = np.zeros((n_ch, sr * n_rec), dtype=np.int64)
        for ch in range(n_ch):
            samples[ch] = np.arange(-20000, -20000 + sr * n_rec, dtype=np.int64) + ch * 100
        create_edf(os.path.join(edf_dir, 'negative.edf'), n_channels=n_ch, n_records=n_rec,
                   sample_rate=sr, samples=np.clip(samples, -32768, 32767))
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Negative-mean signal failures: {fails}"

    def test_dc_offset_signal(self, edf_dir):
        """Constant DC signal (16384 on all channels) — zero detail energy."""
        n_ch, n_rec, sr = 4, 5, 250
        samples = np.full((n_ch, sr * n_rec), 16384, dtype=np.int64)
        create_edf(os.path.join(edf_dir, 'dc.edf'), n_channels=n_ch, n_records=n_rec,
                   sample_rate=sr, samples=samples)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"DC offset signal failures: {fails}"


# ============================================================
# 3. Channel count and duration variants
# ============================================================

@_skip_no_rust
class TestChannelVariants:
    """Different channel counts and sample rates."""

    def test_minimal_channels(self, edf_dir):
        """2 channels, 128 Hz — minimum viable EDF."""
        create_edf(os.path.join(edf_dir, 'single.edf'), n_channels=2, sample_rate=128)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Minimal channel count failures: {fails}"

    def test_high_channel_count(self, edf_dir):
        """128 channels, 256 Hz — high-density research EEG."""
        create_edf(os.path.join(edf_dir, 'many.edf'), n_channels=128, n_records=3, sample_rate=256)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"High channel count failures: {fails}"

    def test_short_recording(self, edf_dir):
        """Single record (1 second) — minimum duration."""
        create_edf(os.path.join(edf_dir, 'short.edf'), n_records=1, n_channels=8)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Short recording failures: {fails}"


# ============================================================
# 4. Edge cases — bugs found in production
# ============================================================

@_skip_no_rust
class TestEdgeCases:
    """Edge cases that have caused real bugs in production."""

    def test_backslash_in_patient_id(self, edf_dir):
        r"""Patient ID with backslashes — regression test for JSON escape bug."""
        create_edf(os.path.join(edf_dir, 'backslash.edf'), patient_id='John\\Doe #\\#####')
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Backslash patient ID failures: {fails}"

    def test_unicode_filename(self, edf_dir):
        """Unicode filename (German umlaut) — tests path encoding."""
        create_edf(os.path.join(edf_dir, 'data.edf'), n_channels=4)
        with open(os.path.join(edf_dir, 'notes_über.txt'), 'w') as f:
            f.write('German notes\n')
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Unicode filename failures: {fails}"

    def test_empty_file_in_directory(self, edf_dir):
        """Directory containing an empty (0-byte) file — must not be dropped."""
        create_edf(os.path.join(edf_dir, 'data.edf'), n_channels=4, n_records=2)
        open(os.path.join(edf_dir, 'empty.txt'), 'w').close()
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Empty file roundtrip failures: {fails}"

    def test_hidden_files(self, edf_dir):
        """Hidden files (.hidden, .gitignore) must be preserved."""
        create_edf(os.path.join(edf_dir, 'data.edf'), n_channels=4)
        with open(os.path.join(edf_dir, '.hidden'), 'w') as f:
            f.write('hidden config\n')
        with open(os.path.join(edf_dir, '.gitignore'), 'w') as f:
            f.write('*.pyc\n')
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Hidden file roundtrip failures: {fails}"


# ============================================================
# 5. Mixed content directories
# ============================================================

@_skip_no_rust
class TestMixedContent:
    """Directories with mixed file types — the real-world scenario."""

    def test_mixed_directory(self, edf_dir):
        """EDF + TXT + CSV + binary + nested subdirectories."""
        create_edf(os.path.join(edf_dir, 'data.edf'))
        with open(os.path.join(edf_dir, 'README.txt'), 'w') as f:
            f.write('This is a test dataset.\n')
        with open(os.path.join(edf_dir, 'annotations.csv'), 'w') as f:
            f.write('onset,duration,label\n1.0,2.0,seizure\n')
        with open(os.path.join(edf_dir, 'RECORDS'), 'w') as f:
            f.write('data.edf\n')
        os.makedirs(os.path.join(edf_dir, 'sub', 'dir'), exist_ok=True)
        with open(os.path.join(edf_dir, 'sub', 'dir', 'nested.txt'), 'w') as f:
            f.write('nested file\n')
        with open(os.path.join(edf_dir, 'binary.bin'), 'wb') as f:
            f.write(os.urandom(1024))
        # Set specific mtimes for deterministic comparison
        for root, dirs, files in os.walk(edf_dir):
            for f in files:
                p = os.path.join(root, f)
                os.utime(p, (1700000000, 1700000000))

        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Mixed directory roundtrip failures: {fails}"


# ============================================================
# 6. Stress test — large files
# ============================================================

@_skip_no_rust
@pytest.mark.slow
class TestLargeFile:
    """Stress test with large files."""

    def test_large_edf(self, edf_dir):
        """64ch, 512Hz, 10min (~75MB) — memory and throughput stress test."""
        create_edf(os.path.join(edf_dir, 'large.edf'), n_channels=64, n_records=600, sample_rate=512)
        out = rust_archive_extract(edf_dir)
        fails = compare_dirs(edf_dir, out)
        shutil.rmtree(out)
        assert not fails, f"Large file roundtrip failures: {fails}"


# ============================================================
# 7. Cross-language parity — Python encode, Rust decode
# ============================================================

@_skip_no_rust
class TestCrossDecoder:
    """Python LMA pack -> Rust extract — cross-language byte parity."""

    def test_python_pack_rust_extract(self, edf_dir):
        """Python pack_lma() output must extract identically via Rust CLI."""
        from lamquant_codec.lma import pack_lma
        create_edf(os.path.join(edf_dir, 'cross.edf'), n_channels=21, sample_rate=256)

        lma = os.path.join(edf_dir, 'cross.lma')
        pack_lma(edf_dir, lma, verbose=False)

        out = tempfile.mkdtemp()
        result = subprocess.run(
            [RUST_BIN, 'extract', lma, '-o', out, '--verify'],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"Rust extract of Python-packed LMA failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Compare the EDF byte-for-byte
        orig = open(os.path.join(edf_dir, 'cross.edf'), 'rb').read()
        extr_path = os.path.join(out, 'cross.edf')
        assert os.path.exists(extr_path), (
            f"Extracted file not found at {extr_path}. "
            f"Extract dir contents: {os.listdir(out)}"
        )
        extr = open(extr_path, 'rb').read()
        shutil.rmtree(out)
        assert orig == extr, (
            f"Cross-decode content mismatch: "
            f"orig={len(orig)}B sha={sha256(orig)[:16]}, "
            f"extr={len(extr)}B sha={sha256(extr)[:16]}"
        )
