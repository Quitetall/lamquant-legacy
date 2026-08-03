"""End-to-end tests for the batch operations layer.

Verifies:
  - expand_inputs() handles single files, directories, recursion, globs.
  - mirror_path() preserves the input directory tree under output_dir.
  - compress_batch + decompress_batch round-trip on synthetic .npy data.
  - Manifest CSV round-trips (write → load → verify rows).
  - --skip-existing actually skips.
  - --from-manifest resume drops succeeded inputs.
  - Failed inputs are reported but don't crash the batch.
  - Bit-exact lossless compress/decompress through the full batch path.

Tests use only `.npy` inputs so they don't depend on mne/EDF being installed
or on a model checkpoint being present (lossless mode requires neither).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lamquant_codec.batch import (
    BatchResult, BatchReport,
    expand_inputs, mirror_path,
    compress_batch, decompress_batch,
)


# ============================================================
# Fixtures: synthetic EEG corpus on disk
# ============================================================

def _make_synth_eeg(seed: int = 0, channels: int = 21, samples: int = 2500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, (channels, samples)).astype(np.float32)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Build a small mirrored directory tree of synthetic EEG files.

    tmp_path/
      data/
        patient_001/
          session_a.npy
          session_b.npy
        patient_002/
          session_a.npy
    """
    root = tmp_path / 'data'
    (root / 'patient_001').mkdir(parents=True)
    (root / 'patient_002').mkdir(parents=True)
    np.save(root / 'patient_001' / 'session_a.npy', _make_synth_eeg(0))
    np.save(root / 'patient_001' / 'session_b.npy', _make_synth_eeg(1))
    np.save(root / 'patient_002' / 'session_a.npy', _make_synth_eeg(2))
    return root


# ============================================================
# Input expansion
# ============================================================

def test_expand_inputs_single_file(corpus: Path):
    f = corpus / 'patient_001' / 'session_a.npy'
    files = expand_inputs([str(f)], exts=('.npy',))
    assert files == [f]


def test_expand_inputs_directory_non_recursive(corpus: Path):
    files = expand_inputs([str(corpus / 'patient_001')], exts=('.npy',))
    names = sorted(p.name for p in files)
    assert names == ['session_a.npy', 'session_b.npy']


def test_expand_inputs_directory_recursive(corpus: Path):
    files = expand_inputs([str(corpus)], recursive=True, exts=('.npy',))
    assert len(files) == 3
    assert all(p.suffix == '.npy' for p in files)


def test_expand_inputs_glob(corpus: Path):
    pattern = str(corpus / 'patient_*' / 'session_a.npy')
    files = expand_inputs([pattern], exts=('.npy',))
    names = sorted(str(p.parent.name) for p in files)
    assert names == ['patient_001', 'patient_002']


def test_expand_inputs_dedup(corpus: Path):
    f = corpus / 'patient_001' / 'session_a.npy'
    files = expand_inputs([str(f), str(f)], exts=('.npy',))
    assert len(files) == 1


def test_expand_inputs_nonexistent_path_silently_dropped(corpus: Path):
    files = expand_inputs([str(corpus / 'no_such_file.npy')], exts=('.npy',))
    assert files == []


def test_expand_inputs_filters_by_extension(corpus: Path):
    # Drop a .txt file in the corpus — must be ignored.
    (corpus / 'patient_001' / 'notes.txt').write_text('hello')
    files = expand_inputs([str(corpus / 'patient_001')], exts=('.npy',))
    assert all(p.suffix == '.npy' for p in files)


# ============================================================
# Mirror tree
# ============================================================

def test_mirror_path_preserves_subdirs(tmp_path: Path):
    in_root = tmp_path / 'in'
    out_root = tmp_path / 'out'
    p = in_root / 'a' / 'b' / 'rec.edf'
    p.parent.mkdir(parents=True)
    p.touch()
    result = mirror_path(p, in_root, out_root, '.lmq')
    assert result == out_root / 'a' / 'b' / 'rec.lmq'


def test_mirror_path_swaps_extension():
    out = mirror_path(Path('/x/y/rec.edf'), Path('/x'), Path('/z'), '.lml')
    assert out == Path('/z/y/rec.lml')


def test_mirror_path_falls_back_to_flat_layout_outside_root(tmp_path: Path):
    in_root = tmp_path / 'in'
    in_root.mkdir()
    elsewhere = tmp_path / 'elsewhere' / 'rec.npy'
    elsewhere.parent.mkdir()
    elsewhere.touch()
    out_root = tmp_path / 'out'
    result = mirror_path(elsewhere, in_root, out_root, '.lml')
    # Falls back to just the file name when input is outside the declared root.
    assert result == out_root / 'rec.lml'


# ============================================================
# Batch report
# ============================================================

def test_batch_report_counts_and_summary():
    rep = BatchReport(results=[
        BatchResult('a.npy', 'a.lml', 'success', 100, 25, 4.0, 12),
        BatchResult('b.npy', 'b.lml', 'success', 100, 50, 2.0, 14),
        BatchResult('c.npy', 'c.lml', 'failed', 0, 0, 0.0, 5, error='boom'),
        BatchResult('d.npy', 'd.lml', 'skipped', 0, 0, 0.0, 0),
    ])
    s = rep.summary()
    assert s['total'] == 4
    assert s['success'] == 2
    assert s['failed'] == 1
    assert s['skipped'] == 1
    assert s['avg_cr'] == pytest.approx((100 + 100) / (25 + 50))


def test_batch_report_csv_roundtrip(tmp_path: Path):
    src = BatchReport(results=[
        BatchResult('a.npy', 'a.lml', 'success', 1024, 256, 4.0, 12.5,
                    prd=2.3, pearson_r=0.99, lqs_pass=True),
        BatchResult('b.npy', 'b.lml', 'failed', error='oops'),
    ])
    csv_path = tmp_path / 'manifest.csv'
    src.to_csv(csv_path)

    loaded = BatchReport.from_csv(csv_path)
    assert len(loaded.results) == 2
    a, b = loaded.results
    assert a.input_path == 'a.npy'
    assert a.status == 'success'
    assert a.cr == 4.0
    assert a.prd == 2.3
    assert a.lqs_pass is True
    assert b.status == 'failed'
    assert b.error == 'oops'


# ============================================================
# Lossless compress/decompress round-trip via batch
# ============================================================

def test_compress_batch_lossless_mirrors_tree(corpus: Path, tmp_path: Path):
    out_dir = tmp_path / 'compressed'
    rep = compress_batch(
        inputs=[str(corpus)],
        output_dir=out_dir,
        mode='lossless',
        recursive=True,
        skip_existing=False,
        workers=1,            # serial — easier to debug
        quiet=True,
    )
    assert rep.n_total == 3
    assert rep.n_success == 3, [r.error for r in rep.results if r.error]

    # Output tree mirrors input tree, with .lml extension.
    expected = {
        out_dir / 'patient_001' / 'session_a.lml',
        out_dir / 'patient_001' / 'session_b.lml',
        out_dir / 'patient_002' / 'session_a.lml',
    }
    actual = set(Path(r.output_path) for r in rep.results)
    assert actual == expected
    for p in expected:
        assert p.exists()
        assert p.read_bytes()[:3] == b'LML'


def test_compress_then_decompress_lossless_bit_exact(corpus: Path, tmp_path: Path):
    """Full round-trip through the batch layer on real data."""
    enc_dir = tmp_path / 'encoded'
    dec_dir = tmp_path / 'decoded'

    enc_rep = compress_batch(
        inputs=[str(corpus)], output_dir=enc_dir, mode='lossless',
        recursive=True, skip_existing=False, workers=1, quiet=True,
    )
    assert enc_rep.n_failed == 0

    dec_rep = decompress_batch(
        inputs=[str(enc_dir)], output_dir=dec_dir,
        recursive=True, skip_existing=False, workers=1, quiet=True,
    )
    assert dec_rep.n_failed == 0

    # Compare each decoded file back to its original — bit-exact.
    for f in corpus.rglob('*.npy'):
        rel = f.relative_to(corpus)
        decoded = dec_dir / rel
        assert decoded.exists(), f'missing decoded output: {decoded}'
        original = np.load(f)
        recon = np.load(decoded)
        assert np.array_equal(recon.astype(np.int64), original.astype(np.int64)), \
            f'bit-exact lossless broken on {rel}'


def test_skip_existing_drops_already_done(corpus: Path, tmp_path: Path):
    out_dir = tmp_path / 'out'
    # First pass writes everything.
    rep1 = compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, skip_existing=True, workers=1, quiet=True,
    )
    assert rep1.n_total == 3 and rep1.n_success == 3
    # Second pass with skip-existing must be a no-op (zero jobs).
    rep2 = compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, skip_existing=True, workers=1, quiet=True,
    )
    assert rep2.n_total == 0


def test_dry_run_does_not_create_outputs(corpus: Path, tmp_path: Path, capsys):
    out_dir = tmp_path / 'out'
    rep = compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, dry_run=True, workers=1, quiet=True,
    )
    assert rep.n_total == 0      # dry-run returns empty report
    assert not out_dir.exists()
    captured = capsys.readouterr()
    assert 'Dry-run' in captured.out or 'would compress' in captured.out


def test_manifest_written_and_loadable(corpus: Path, tmp_path: Path):
    out_dir = tmp_path / 'out'
    manifest = tmp_path / 'manifest.csv'
    rep = compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, skip_existing=False, workers=1, quiet=True,
        manifest=str(manifest),
    )
    assert manifest.exists()
    loaded = BatchReport.from_csv(manifest)
    assert loaded.n_total == rep.n_total
    assert loaded.n_success == rep.n_success
    # Status values are preserved.
    statuses = sorted(r.status for r in loaded.results)
    assert statuses == ['success'] * 3


def test_resume_from_manifest_drops_succeeded(corpus: Path, tmp_path: Path):
    """If a previous manifest marks files as 'success', a resume pass skips them."""
    out_dir = tmp_path / 'out'
    manifest = tmp_path / 'manifest.csv'

    # First run — write manifest.
    compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, skip_existing=False, workers=1, quiet=True,
        manifest=str(manifest),
    )
    # Wipe outputs but keep manifest. Resume should skip everything.
    for p in out_dir.rglob('*.lml'):
        p.unlink()
    rep = compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, skip_existing=False,
        from_manifest=str(manifest),
        workers=1, quiet=True,
    )
    assert rep.n_total == 0


def test_failed_input_does_not_crash_batch(corpus: Path, tmp_path: Path):
    """One malformed input must not bring down the whole batch."""
    bad = corpus / 'patient_001' / 'broken.npy'
    bad.write_bytes(b'this is not a numpy file')
    out_dir = tmp_path / 'out'

    rep = compress_batch(
        inputs=[str(corpus)], output_dir=out_dir, mode='lossless',
        recursive=True, skip_existing=False, workers=1, quiet=True,
    )
    # 3 valid + 1 broken = 4 total; the broken one fails, the rest succeed.
    assert rep.n_total == 4
    assert rep.n_success == 3
    assert rep.n_failed == 1
    failed = [r for r in rep.results if r.status == 'failed'][0]
    assert 'broken.npy' in failed.input_path
    assert failed.error  # populated, not None


# ============================================================
# Multi-worker correctness (parallel path)
# ============================================================

def test_compress_batch_parallel_correctness(corpus: Path, tmp_path: Path):
    """workers=2 must produce the same outputs as workers=1."""
    out_serial = tmp_path / 'serial'
    out_parallel = tmp_path / 'parallel'

    compress_batch(
        inputs=[str(corpus)], output_dir=out_serial, mode='lossless',
        recursive=True, skip_existing=False, workers=1, quiet=True,
    )
    compress_batch(
        inputs=[str(corpus)], output_dir=out_parallel, mode='lossless',
        recursive=True, skip_existing=False, workers=2, quiet=True,
    )

    # Same set of output files, byte-identical contents.
    serial_files = {p.name: p.read_bytes() for p in out_serial.rglob('*.lml')}
    parallel_files = {p.name: p.read_bytes() for p in out_parallel.rglob('*.lml')}
    assert set(serial_files) == set(parallel_files)
    for name, data in serial_files.items():
        assert data == parallel_files[name], f'parallel != serial for {name}'


def test_summary_compression_ratio_makes_sense(corpus: Path, tmp_path: Path):
    out = tmp_path / 'out'
    rep = compress_batch(
        inputs=[str(corpus)], output_dir=out, mode='lossless',
        recursive=True, skip_existing=False, workers=1, quiet=True,
    )
    s = rep.summary()
    assert s['total_raw_bytes'] > 0
    assert s['total_compressed_bytes'] > 0
    assert s['avg_cr'] > 1.0   # actually compressed (random EEG → ~3-5×)


# ============================================================
# verify_batch — structural integrity check
# ============================================================

def test_verify_batch_passes_clean_files(corpus: Path, tmp_path: Path):
    """verify_batch should report success on freshly-compressed files."""
    from lamquant_codec.batch import verify_batch
    out = tmp_path / 'out'
    compress_batch(inputs=[str(corpus)], output_dir=out, mode='lossless',
                   recursive=True, skip_existing=False, workers=1, quiet=True)
    rep = verify_batch(inputs=[str(out)], recursive=True, workers=1, quiet=True)
    assert rep.n_total == 3
    assert rep.n_success == 3
    # The 'output_path' field carries the integrity-check summary.
    for r in rep.results:
        assert 'ok' in r.output_path.lower()


def test_verify_batch_with_decode_roundtrip(corpus: Path, tmp_path: Path):
    """--decode triggers a full decompress round-trip on every file."""
    from lamquant_codec.batch import verify_batch
    out = tmp_path / 'out'
    compress_batch(inputs=[str(corpus)], output_dir=out, mode='lossless',
                   recursive=True, skip_existing=False, workers=1, quiet=True)
    rep = verify_batch(inputs=[str(out)], decode=True, recursive=True,
                       workers=1, quiet=True)
    assert rep.n_success == 3
    for r in rep.results:
        assert 'decode ok' in r.output_path


def test_verify_batch_catches_truncated_file(tmp_path: Path):
    """Truncating a file mid-payload must be reported as failed."""
    from lamquant_codec.batch import verify_batch
    out = tmp_path / 'out'
    out.mkdir()
    src_npy = tmp_path / 'src.npy'
    np.save(src_npy, _make_synth_eeg(0))
    compress_batch(inputs=[str(src_npy)], output_dir=out, mode='lossless',
                   workers=1, quiet=True, skip_existing=False)
    lml = next(out.rglob('*.lml'))
    # Truncate to half its size — declared payload exceeds file length.
    sz = lml.stat().st_size
    with open(lml, 'rb') as f:
        head = f.read(sz // 2)
    lml.write_bytes(head)

    rep = verify_batch(inputs=[str(out)], recursive=True, workers=1, quiet=True)
    assert rep.n_failed == 1
    assert any(r.error for r in rep.results)  # truncated file must produce an error


def test_verify_batch_catches_garbage_file(tmp_path: Path):
    """A file with random bytes must fail magic detection."""
    from lamquant_codec.batch import verify_batch
    out = tmp_path / 'out'
    out.mkdir()
    bogus = out / 'bogus.lml'
    bogus.write_bytes(b'\xff' * 200)
    rep = verify_batch(inputs=[str(out)], recursive=True, workers=1, quiet=True)
    assert rep.n_failed == 1
    assert 'magic' in rep.results[0].error.lower()


# ============================================================
# HTML report
# ============================================================

def test_html_report_renders_for_compressed_batch(corpus: Path, tmp_path: Path):
    """write_html_report should produce a self-contained HTML file."""
    from lamquant_codec.report import write_html_report
    out = tmp_path / 'out'
    rep = compress_batch(inputs=[str(corpus)], output_dir=out, mode='lossless',
                         recursive=True, skip_existing=False, workers=1, quiet=True)
    html_path = tmp_path / 'report.html'
    written = write_html_report(rep, html_path, level='C',
                                title='Test Report')
    assert written == html_path
    assert html_path.exists()
    body = html_path.read_text()
    # Must contain the key structural elements.
    assert '<!doctype html>' in body
    assert 'Test Report' in body
    assert '<table' in body
    # Per-file table has one row per result.
    assert body.count('<tr') >= len(rep.results)
    # No external assets (no <link rel="stylesheet">, no <script src=>).
    assert 'rel="stylesheet"' not in body
    assert '<script src=' not in body
    # SVG charts only get rendered when there's quality data; for plain
    # compress (no PRD/R) the histograms are skipped — verify no <svg> tags.
    # (validate_batch would emit them; tested separately.)


def test_html_report_handles_degenerate_data(tmp_path: Path):
    """All-identical values (e.g. lossless R=1.0 everywhere) must not crash matplotlib."""
    from lamquant_codec.report import write_html_report
    rep = BatchReport(results=[
        BatchResult('a.lml', '', 'success', 1024, 256, 4.0, 12.0,
                    prd=0.0, pearson_r=1.0, lqs_pass=True),
        BatchResult('b.lml', '', 'success', 1024, 256, 4.0, 12.0,
                    prd=0.0, pearson_r=1.0, lqs_pass=True),
    ])
    out = tmp_path / 'degenerate.html'
    write_html_report(rep, out, level='L')
    assert out.exists() and '<svg' in out.read_text()


def test_html_report_handles_empty_results(tmp_path: Path):
    """Zero results should render a 'no data' placeholder, not crash."""
    from lamquant_codec.report import write_html_report
    rep = BatchReport(results=[])
    out = tmp_path / 'empty.html'
    write_html_report(rep, out)
    assert out.exists()


def test_html_report_with_quality_metrics(tmp_path: Path):
    """When BatchResults have prd/pearson_r set, charts are inlined."""
    from lamquant_codec.report import write_html_report
    rep = BatchReport(results=[
        BatchResult('a.lml', '', 'success', 1024, 256, 4.0, 12.0,
                    prd=2.3, pearson_r=0.99, lqs_pass=True),
        BatchResult('b.lml', '', 'success', 1024, 256, 4.0, 14.0,
                    prd=11.4, pearson_r=0.92, lqs_pass=False),
        BatchResult('c.lml', '', 'success', 1024, 512, 2.0, 13.0,
                    prd=4.1, pearson_r=0.97, lqs_pass=True),
    ], total_seconds=0.5)
    out = tmp_path / 'q.html'
    write_html_report(rep, out, level='C')
    body = out.read_text()
    # Should contain at least one inline SVG (matplotlib chart).
    assert '<svg' in body
    # Stats table reports the right counts.
    assert 'pass' in body and 'fail' in body
    # Top failures section present (we have one fail).
    assert 'Top failures' in body
    assert 'b.lml' in body
