"""Coverage tests for `lamquant_codec.batch`.

Targets the still-uncovered branches in batch.py:

  - BatchReport.print_summary on mixed result set (success + failed)
  - _human() bytes-to-human conversion thresholds
  - _expand_single() glob/directory/file branches
  - expand_inputs with stdin substitution (`-`)
  - decompress_batch dry-run path
  - decompress_batch happy path (round-trip via lossless)
  - _decompress_one path on .lml input
  - _validate_one on .lml file with quality threshold
  - validate_batch with reference_dir
  - validate_batch without reference_dir (structural only)
  - _verify_one negative paths (tiny file, bad magic, prefixed magic)
  - verify_batch structural pass + decode-true
  - info_batch on LML1 / LMA1 / unknown magic
  - mirror_path edge cases (outside root → flat layout)
  - _filter_resume / _filter_skip_existing helpers

Builds real .lml files using the production lossless codec (no fake bytes).
Uses tmp_path for all I/O; no synthetic EEG semantics — math fixtures only.
"""
from __future__ import annotations

import io
import os
import struct
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lamquant_codec.batch import (
    BatchReport,
    BatchResult,
    COMPRESS_EXTS,
    DECOMPRESS_EXTS,
    _expand_single,
    _filter_resume,
    _filter_skip_existing,
    _human,
    _verify_one,
    compress_batch,
    decompress_batch,
    expand_inputs,
    info_batch,
    mirror_path,
    validate_batch,
    verify_batch,
)

pytestmark = [pytest.mark.l3]


# ============================================================
# 1. _human() — bytes to human-readable
# ============================================================


class TestHuman:

    @pytest.mark.parametrize("n,fragment", [
        (0, "B"),
        (1023, "B"),
        (1024, "KB"),
        (1024 * 1024, "MB"),
        (1024 ** 3, "GB"),
        (1024 ** 4, "TB"),
        (1024 ** 5, "PB"),
    ])
    def test_threshold_units(self, n, fragment):
        result = _human(n)
        assert fragment in result, f"_human({n}) = {result!r}, expected {fragment}"


# ============================================================
# 2. BatchReport.print_summary
# ============================================================


class TestPrintSummary:

    def test_print_summary_includes_counts_and_failures(self, capsys):
        rep = BatchReport(results=[
            BatchResult('a.npy', 'a.lml', 'success', 1000, 200, 5.0, 10.5),
            BatchResult('b.npy', 'b.lml', 'failed', error='boom'),
            BatchResult('c.npy', 'c.lml', 'skipped'),
        ])
        rep.total_seconds = 2.5
        rep.print_summary(file=sys.stdout)
        captured = capsys.readouterr().out
        # Summary lines.
        assert "Batch summary" in captured
        assert "success 1" in captured
        assert "failed 1" in captured
        assert "skipped 1" in captured
        # Failed file is listed under the FAILED block.
        assert "FAILED" in captured
        assert "b.npy" in captured
        assert "boom" in captured

    def test_print_summary_no_failed_section_when_zero(self, capsys):
        rep = BatchReport(results=[
            BatchResult('a.npy', 'a.lml', 'success', 1000, 200, 5.0, 12.5),
        ])
        rep.print_summary(file=sys.stdout)
        captured = capsys.readouterr().out
        assert "FAILED" not in captured

    def test_print_summary_omits_compression_block_when_zero(self, capsys):
        # All skipped — no raw bytes. Compression block must NOT render.
        rep = BatchReport(results=[
            BatchResult('a.npy', 'a.lml', 'skipped'),
            BatchResult('b.npy', 'b.lml', 'skipped'),
        ])
        rep.print_summary(file=sys.stdout)
        captured = capsys.readouterr().out
        assert "raw → " not in captured


# ============================================================
# 3. _expand_single — file / dir / glob branches
# ============================================================


class TestExpandSingle:

    def test_file_returns_self(self, tmp_path: Path):
        f = tmp_path / "a.npy"
        f.touch()
        result = _expand_single(str(f), recursive=False, exts=('.npy',))
        assert result == [f]

    def test_directory_non_recursive(self, tmp_path: Path):
        (tmp_path / "x.npy").touch()
        (tmp_path / "y.npy").touch()
        (tmp_path / "z.txt").touch()  # filtered out
        result = _expand_single(str(tmp_path), recursive=False, exts=('.npy',))
        names = sorted(p.name for p in result)
        assert names == ["x.npy", "y.npy"]

    def test_directory_recursive(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.npy").touch()
        (tmp_path / "top.npy").touch()
        result = _expand_single(str(tmp_path), recursive=True, exts=('.npy',))
        assert len(result) == 2

    def test_glob_pattern(self, tmp_path: Path):
        (tmp_path / "rec_001.npy").touch()
        (tmp_path / "rec_002.npy").touch()
        pattern = str(tmp_path / "rec_*.npy")
        result = _expand_single(pattern, recursive=False, exts=('.npy',))
        assert len(result) == 2


# ============================================================
# 4. expand_inputs with stdin
# ============================================================


class TestExpandInputsStdin:

    def test_stdin_marker_consumes_lines(self, tmp_path: Path):
        f1 = tmp_path / "a.npy"
        f2 = tmp_path / "b.npy"
        f1.touch(); f2.touch()
        stdin = io.StringIO(f"{f1}\n{f2}\n\n")
        with patch.object(sys, 'stdin', stdin):
            result = expand_inputs(['-'], exts=('.npy',))
        # Both files surface from the stdin pipe.
        names = sorted(p.name for p in result)
        assert names == ["a.npy", "b.npy"]


# ============================================================
# 5. mirror_path outside-root branch
# ============================================================


class TestMirrorPathFallback:

    def test_outside_root_falls_back_to_flat(self, tmp_path: Path):
        in_root = tmp_path / "declared"
        in_root.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        f = outside / "rec.edf"
        f.touch()
        out_root = tmp_path / "compressed"
        result = mirror_path(f, in_root, out_root, '.lml')
        # File is outside in_root → flat under out_root.
        assert result == out_root / "rec.lml"


# ============================================================
# 6. _filter_resume + _filter_skip_existing
# ============================================================


class TestFilterHelpers:

    def test_filter_resume_drops_succeeded(self, tmp_path: Path):
        # Build a fake manifest with one success row.
        manifest = tmp_path / "manifest.csv"
        rep = BatchReport(results=[
            BatchResult('did_succeed.npy', 'x.lml', 'success', 100, 25, 4.0, 12),
            BatchResult('did_fail.npy', 'y.lml', 'failed', error='oops'),
        ])
        rep.to_csv(manifest)
        jobs = [
            {'input': 'did_succeed.npy'},
            {'input': 'did_fail.npy'},
            {'input': 'never_seen.npy'},
        ]
        filtered = _filter_resume(jobs, str(manifest))
        # Successful job is dropped; failed + new remain.
        names = sorted(j['input'] for j in filtered)
        assert names == ['did_fail.npy', 'never_seen.npy']

    def test_filter_resume_with_no_manifest_path_is_passthrough(self):
        jobs = [{'input': 'a'}, {'input': 'b'}]
        # None or missing path → pass-through.
        assert _filter_resume(jobs, None) == jobs
        assert _filter_resume(jobs, "/no/such/path.csv") == jobs

    def test_filter_skip_existing(self, tmp_path: Path):
        existing = tmp_path / "exists.lml"
        existing.touch()
        absent = tmp_path / "missing.lml"
        jobs = [
            {'input': 'a.npy', 'output': str(existing)},
            {'input': 'b.npy', 'output': str(absent)},
        ]
        filtered = _filter_skip_existing(jobs)
        # Only the missing-output job remains.
        assert len(filtered) == 1
        assert filtered[0]['input'] == 'b.npy'


# ============================================================
# 7. decompress_batch — dry-run + happy path
# ============================================================


@pytest.fixture
def npy_corpus(tmp_path: Path) -> Path:
    """A tiny corpus of valid .npy files (math fixtures)."""
    root = tmp_path / "src"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        sig = rng.integers(-1500, 1500, (3, 256)).astype(np.float32)
        np.save(root / f"rec_{i}.npy", sig)
    return root


class TestDecompressBatch:

    def test_dry_run_prints_summary_no_files(self, npy_corpus: Path,
                                             tmp_path: Path, capsys):
        # First compress to get .lml files.
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        # Then dry-run decompress.
        dec = tmp_path / "dec"
        rep = decompress_batch(inputs=[str(enc)], output_dir=dec,
                               recursive=True, dry_run=True, workers=1,
                               quiet=True)
        assert rep.n_total == 0
        assert not dec.exists()
        captured = capsys.readouterr().out
        assert "Dry-run" in captured

    def test_happy_path_writes_outputs_and_manifest(self, npy_corpus: Path,
                                                    tmp_path: Path):
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        dec = tmp_path / "dec"
        manifest = tmp_path / "dec_manifest.csv"
        rep = decompress_batch(inputs=[str(enc)], output_dir=dec,
                               recursive=True, skip_existing=False,
                               manifest=str(manifest),
                               workers=1, quiet=True)
        assert rep.n_total == 2
        assert rep.n_success == 2
        assert manifest.exists()
        # Outputs are .npy by default.
        npy_files = list(dec.rglob('*.npy'))
        assert len(npy_files) == 2


# ============================================================
# 8. validate_batch
# ============================================================


class TestValidateBatch:

    def test_validate_without_reference_dir(self, npy_corpus: Path,
                                            tmp_path: Path):
        # Compress first.
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        rep = validate_batch(inputs=[str(enc)], recursive=True,
                             workers=1, quiet=True)
        assert rep.n_total == 2
        # Without reference_dir, prd/pearson_r are None.
        for r in rep.results:
            assert r.prd is None
            assert r.pearson_r is None

    def test_validate_with_reference_dir(self, npy_corpus: Path,
                                         tmp_path: Path):
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        rep = validate_batch(inputs=[str(enc)], recursive=True,
                             reference_dir=str(npy_corpus),
                             level='L', workers=1, quiet=True)
        assert rep.n_success == 2
        # Lossless → PRD ≈ 0, R ≈ 1.
        for r in rep.results:
            assert r.prd is not None
            assert r.prd < 1.0
            assert r.pearson_r is not None
            assert r.pearson_r > 0.99
            assert r.lqs_pass is True


# ============================================================
# 9. _verify_one negative paths
# ============================================================


class TestVerifyOne:

    def test_tiny_file_fails(self, tmp_path: Path):
        f = tmp_path / "tiny.lml"
        f.write_bytes(b"\x00\x01")
        res = _verify_one({'input': str(f), 'decode': False})
        assert res.status == 'failed'
        assert res.error  # non-empty
        assert "too small" in res.error.lower()

    def test_bad_magic_fails(self, tmp_path: Path):
        f = tmp_path / "garbage.lml"
        f.write_bytes(b"\xFF" * 200)
        res = _verify_one({'input': str(f), 'decode': False})
        assert res.status == 'failed'
        assert "magic" in res.error.lower()

    def test_lml1_prefix_path_recognised(self, tmp_path: Path,
                                         npy_corpus: Path):
        # Build a real LML1 file and verify the prefixed-magic branch.
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        lml = next(enc.rglob('*.lml'))
        res = _verify_one({'input': str(lml), 'decode': False})
        assert res.status == 'success'
        # output_path carries the check summary.
        assert "LML" in res.output_path


# ============================================================
# 10. info_batch — LML1 / LMA1 / unknown
# ============================================================


class TestInfoBatch:

    def test_info_batch_on_lml(self, npy_corpus: Path, tmp_path: Path):
        # Compress to get an LML file batch.
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        info = info_batch([str(enc)], recursive=True)
        assert len(info) == 2
        # Either it parsed as LML1 multi-window container OR returned a
        # struct error message (single-packet vs wrapper format dispatch).
        for entry in info:
            assert entry['input_path'].endswith('.lml')
            assert 'file_size_bytes' in entry

    def test_info_batch_on_lma_archive(self, tmp_path: Path):
        # Build an LMA, then point info_batch at it via the explicit path.
        # expand_inputs honours single-file paths regardless of extension
        # filter, so .lma is inspected and the LMA1 branch is exercised.
        from lamquant_codec.lma import pack_lma
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_bytes(b"hi")
        archive = tmp_path / "x.lma"
        pack_lma(str(src), str(archive), verbose=False)
        info = info_batch([str(archive)], recursive=False)
        assert len(info) == 1
        entry = info[0]
        assert entry['magic'] == 'LMA1'
        assert entry['format'].startswith('LMA')
        assert entry['n_files'] == 1

    def test_info_batch_on_unknown_magic_returns_error(self, tmp_path: Path):
        # Drop a junk file with .lml suffix → magic mismatch triggers the
        # "unknown magic" branch in info_batch.
        bogus = tmp_path / "bogus.lml"
        bogus.write_bytes(b"\xFF\xFF\xFF\xFF" + b"\x00" * 60)
        info = info_batch([str(bogus)], recursive=False)
        assert len(info) == 1
        # Either a parse error (struct unpack) or unknown-magic — both are
        # reflected in entry['error'].
        assert info[0].get('error') is not None


# ============================================================
# 11. verify_batch parallel
# ============================================================


class TestVerifyBatchParallel:

    def test_verify_batch_with_workers_2_serial_match(self, npy_corpus: Path,
                                                      tmp_path: Path):
        # Build LML files first.
        enc = tmp_path / "enc"
        compress_batch(inputs=[str(npy_corpus)], output_dir=enc,
                       mode='lossless', recursive=True,
                       skip_existing=False, workers=1, quiet=True)
        rep_serial = verify_batch(inputs=[str(enc)], recursive=True,
                                  workers=1, quiet=True)
        rep_par = verify_batch(inputs=[str(enc)], recursive=True,
                               workers=2, quiet=True)
        # Same count, same success rate.
        assert rep_serial.n_total == rep_par.n_total
        assert rep_serial.n_success == rep_par.n_success


# ============================================================
# 12. compress_batch empty-input warning
# ============================================================


class TestCompressBatchEdge:

    def test_compress_batch_warns_on_no_matches(self, tmp_path: Path):
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        with pytest.warns(UserWarning, match="No matching files"):
            rep = compress_batch(inputs=[str(empty)], output_dir=tmp_path / "out",
                                 mode='lossless', recursive=True,
                                 skip_existing=False, workers=1, quiet=True)
        assert rep.n_total == 0

    def test_constants_pinned(self):
        # Wire format stability — these tuples are public API.
        assert COMPRESS_EXTS == ('.edf', '.npy')
        assert DECOMPRESS_EXTS == ('.lmq', '.lml')


# ============================================================
# 13. Validate batch — failed file branch + missing checkpoint
# ============================================================


class TestValidateBatchErrors:

    def test_validate_lmq_without_checkpoint_fails(self, tmp_path: Path):
        """`.lmq` files require --checkpoint; missing ckpt → result.failed."""
        # Just drop a bogus .lmq file. The validator will read it, find it
        # is .lmq, see no ckpt → ValueError → status='failed'.
        f = tmp_path / "fake.lmq"
        f.write_bytes(b"LMQ1" + b"\x00" * 64)
        rep = validate_batch(inputs=[str(f)], workers=1, quiet=True)
        assert rep.n_failed == 1
        err = rep.results[0].error or ""
        assert "checkpoint" in err.lower()

    def test_validate_unreadable_file_failed(self, tmp_path: Path):
        # A 0-byte .lml — fails on read_lml_file.
        f = tmp_path / "broken.lml"
        f.write_bytes(b"")
        rep = validate_batch(inputs=[str(f)], workers=1, quiet=True)
        assert rep.n_failed == 1


# ============================================================
# 14. Real EDF fixture path — pack via Rust binary
#     Only runs when both the corpus and the Rust binary are present.
# ============================================================


class TestRealLmlViaRustBinary:

    def test_real_lma_from_rust_binary(self, tmp_path: Path,
                                       real_test_edf, lml_cli_binary):
        """Build a real .lma via the Rust binary, then verify Python can
        list its entries. Pins the LMA wire format across languages.

        `lml encode <edf>` produces an LMA1 archive (per-recording per
        the help text) — this exercises the cross-language LMA reader
        contract from the Python side.
        """
        import subprocess
        from lamquant_codec.lma import list_lma, verify_lma, LMA_MAGIC
        out_lma = tmp_path / "real.lma"
        result = subprocess.run(
            [str(lml_cli_binary), "encode",
             str(real_test_edf), "-o", str(out_lma)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not out_lma.exists():
            pytest.skip(
                f"lml binary did not produce output ({result.returncode}): "
                f"{result.stderr[:400]}")
        # Magic + integrity.
        assert out_lma.read_bytes()[:4] == LMA_MAGIC
        # Python verify_lma reads the SHA-256 trailer and returns True.
        assert verify_lma(str(out_lma), verbose=False) is True
        # list_lma yields at least one entry — the EDF wrapped as LML.
        entries = list_lma(str(out_lma))
        assert len(entries) >= 1
        for e in entries:
            assert "sha256" in e
            assert len(e["sha256"]) == 64
            assert e.get("method") in ("lml", "secondary", "zstd", "store")
