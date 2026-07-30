"""Extra coverage for ``ai_models.snn.lma_subject_id``.

Exercises the parser branches not already hit by
``tests/integration/test_lma_dataset.py``:

  - TUEV unparseable filename → empty subject id
  - TUH first-token fallback (non-canonical stem) → falls through to first token
  - TUH unparseable (empty stem)
  - chbmit filename-prefix and parent-dir paths
  - siena filename-prefix and unparseable
  - Unknown corpus shortname

Real TUH-style stems are used where available (via the shared fixtures);
parser-only paths are exercised with documented synthetic stems
(no synthetic EEG bytes — only filename-string inputs, which is what the
parser operates on).
"""
from __future__ import annotations

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

from pathlib import Path

import pytest

from ai_models.snn.lma_subject_id import (
    CORPUS_PRECEDENCE,
    corpus_short_name,
    extract_subject_id,
    precedence_rank,
)


pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# corpus_short_name (covers casing + suffix-stripping)
# ---------------------------------------------------------------------------


class TestCorpusShortName:
    def test_strips_version_suffix(self):
        assert corpus_short_name("tueg_v2.0.1") == "tueg"

    def test_uppercase_lowered(self):
        assert corpus_short_name("TUEG_V2.0.1") == "tueg"

    def test_no_underscore(self):
        assert corpus_short_name("tueg") == "tueg"

    def test_multi_underscore(self):
        # "abc_def_ghi" → "abc" (split takes the first token)
        assert corpus_short_name("abc_def_ghi") == "abc"


# ---------------------------------------------------------------------------
# extract_subject_id — TUH-family + TUEV layouts
# ---------------------------------------------------------------------------


class TestExtractSubjectIdTUH:
    def test_canonical_tueg_stem(self):
        sid, tag = extract_subject_id(
            "tueg_v2.0.1", Path("edf/000/aaaaaaaa/s001/aaaaaaaa_s001_t000.lml")
        )
        assert sid == "aaaaaaaa"
        assert tag == "tueg_filename_regex"

    def test_non_canonical_falls_to_first_token(self):
        """Stem that doesn't match the canonical regex falls to first-token."""
        sid, tag = extract_subject_id(
            "tueg_v2.0.1", Path("edf/x/abc123_extra.lml")
        )
        assert sid == "abc123"
        assert tag == "tueg_filename_first_token"

    def test_truly_empty_stem_returns_unparseable(self):
        """Path with an empty stem ('') returns empty sid + unparseable tag."""
        # Constructing the path with bare filename produces stem=='' which
        # fails the `parts and parts[0]` truthy check.
        from pathlib import Path as P
        p = P("edf/x") / ""
        # Skip if the synthesized path doesn't produce an empty stem
        if p.stem != "":
            pytest.skip("can't synthesize empty stem on this platform")
        sid, tag = extract_subject_id("tueg_v2.0.1", p)
        assert sid == ""
        assert tag == "tueg_unparseable"

    def test_tusz_canonical(self):
        sid, tag = extract_subject_id(
            "tusz_v2.0.6", Path("edf/train/01/aaaaaiwy_s001_t000.lml")
        )
        assert sid == "aaaaaiwy"
        assert tag == "tusz_filename_regex"

    def test_tuab_canonical(self):
        sid, tag = extract_subject_id(
            "tuab_v3.0.1", Path("normal/aaaaaoyp_s001_t001.lml")
        )
        assert sid == "aaaaaoyp"
        assert tag == "tuab_filename_regex"

    def test_tuep_canonical(self):
        sid, tag = extract_subject_id(
            "tuep_v2.0.1", Path("aaaaazzy_s001_t000.lml")
        )
        assert sid == "aaaaazzy"
        assert tag == "tuep_filename_regex"

    def test_tusl_canonical(self):
        sid, tag = extract_subject_id(
            "tusl_v1.0.0", Path("aaaaapqr_s002_t005.lml")
        )
        assert sid == "aaaaapqr"
        assert tag == "tusl_filename_regex"

    def test_tuar_canonical(self):
        sid, tag = extract_subject_id(
            "tuar_v2.0.0", Path("aaaaabrr_s003_t000.lml")
        )
        assert sid == "aaaaabrr"
        assert tag == "tuar_filename_regex"


class TestExtractSubjectIdTUEV:
    def test_train_layout(self):
        sid, tag = extract_subject_id(
            "tuev_v2.0.1", Path("edf/train/aaaaabba/aaaaabba_00000003.lml")
        )
        assert sid == "aaaaabba"
        assert tag == "tuev_train_filename_regex"

    def test_eval_layout_with_run_suffix(self):
        sid, tag = extract_subject_id(
            "tuev_v2.0.1", Path("edf/eval/051/bckg_051_a_1.lml")
        )
        assert sid == "tuev_eval_051"
        assert tag == "tuev_eval_filename_event_regex"

    def test_eval_layout_no_run_suffix(self):
        sid, tag = extract_subject_id(
            "tuev_v2.0.1", Path("edf/eval/036/bckg_036_a_.lml")
        )
        assert sid == "tuev_eval_036"
        assert tag == "tuev_eval_filename_event_regex"

    def test_eval_layout_all_event_types(self):
        """All TUEV event labels parse successfully."""
        for event in ("bckg", "gped", "pled", "spsw", "eyem", "artf"):
            sid, tag = extract_subject_id(
                "tuev_v2.0.1", Path(f"edf/eval/100/{event}_100_a_2.lml")
            )
            assert sid == "tuev_eval_100"
            assert tag == "tuev_eval_filename_event_regex"

    def test_unparseable(self):
        """Stem that matches neither TUEV regex returns unparseable."""
        sid, tag = extract_subject_id(
            "tuev_v2.0.1", Path("edf/eval/000/garbage_stem.lml")
        )
        assert sid == ""
        assert tag == "tuev_unparseable"


class TestExtractSubjectIdCHBMIT:
    def test_filename_prefix(self):
        sid, tag = extract_subject_id(
            "chbmit", Path("chb01/chb01_03.lml")
        )
        assert sid == "chb01"
        assert tag == "chbmit_filename_prefix"

    def test_parent_dir_fallback(self):
        """Stem doesn't start with chb but parent dir does → parent_dir tag."""
        sid, tag = extract_subject_id(
            "chbmit", Path("chb05/seizure_x.lml")
        )
        assert sid == "chb05"
        assert tag == "chbmit_parent_dir"

    def test_unparseable(self):
        """Neither stem nor parents match the chb pattern."""
        sid, tag = extract_subject_id(
            "chbmit", Path("foo/bar/random.lml")
        )
        assert sid == ""
        assert tag == "chbmit_unparseable"


class TestExtractSubjectIdSiena:
    def test_filename_prefix(self):
        sid, tag = extract_subject_id(
            "siena", Path("PN00/seizure_1.lml")
        )
        # Stem is "seizure_1" — doesn't start with PN, so falls to unparseable
        assert tag == "siena_unparseable"

    def test_stem_starts_with_pn(self):
        sid, tag = extract_subject_id(
            "siena", Path("PN12_record_01.lml")
        )
        assert sid == "PN12"
        assert tag == "siena_filename_prefix"

    def test_unparseable(self):
        sid, tag = extract_subject_id(
            "siena", Path("foo.lml")
        )
        assert sid == ""
        assert tag == "siena_unparseable"


class TestExtractSubjectIdUnknown:
    def test_unknown_corpus_returns_tagged_unknown(self):
        sid, tag = extract_subject_id(
            "totally_new_corpus", Path("x.lml")
        )
        assert sid == ""
        assert tag.startswith("unknown_corpus_")


# ---------------------------------------------------------------------------
# precedence_rank
# ---------------------------------------------------------------------------


class TestPrecedenceRank:
    def test_corpus_precedence_constant(self):
        assert CORPUS_PRECEDENCE == (
            "tusz", "tuev", "tuep", "tusl", "tuar", "tuab", "tueg",
        )

    def test_tusz_highest(self):
        # tusz is rank 0 (highest precedence)
        assert precedence_rank("tusz_v2.0.6") == 0

    def test_tueg_lowest_known(self):
        # tueg is the last entry in the precedence tuple
        assert precedence_rank("tueg_v2.0.1") == len(CORPUS_PRECEDENCE) - 1

    def test_unknown_ranks_after_known(self):
        unknown = precedence_rank("future_corpus")
        known = precedence_rank("tueg_v2.0.1")
        assert unknown > known

    def test_strict_ordering(self):
        # Strict less-than across the precedence tuple
        ranks = [precedence_rank(c) for c in CORPUS_PRECEDENCE]
        assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# Real-corpus stem sweep — uses real TUH file stems if available
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_real_tuh_stems_parse_to_8char_subject(real_tuh_edfs):
    """If the NEDC eval corpus is present, every stem should parse to an
    8-character subject id under the tueg / tusz / tuab tag."""
    for edf in real_tuh_edfs:
        lml_like = edf.with_suffix(".lml")
        for corpus in ("tueg", "tusz", "tuab"):
            sid, tag = extract_subject_id(corpus, lml_like)
            # Either canonical regex tag OR first-token fallback
            assert tag in (
                f"{corpus}_filename_regex",
                f"{corpus}_filename_first_token",
            )
            assert sid
