//! LML v1 random-access seek table — additive wire format extension.
//!
//! When `container::FLAG_HAS_FOOTER` (bit 0 of header byte 21) is set,
//! the file carries a fixed 32-byte footer at EOF plus a per-window
//! offset table immediately preceding it. Old readers ignore the
//! flag bit and skip the footer; new readers with the flag set can
//! random-access window N in O(log n) via [`OffsetTable::window_for_sample`].
//!
//! ## Wire layout
//!
//! ```text
//! [file_end - 32 .. file_end]   footer (fixed 32 bytes)
//!   [0:8]    "LMLFOOT1"   (magic)
//!   [8:12]   footer_len    u32 LE  (always 32)
//!   [12:16]  n_windows     u32 LE
//!   [16:20]  CRC-32 of:    u32 LE  (table bytes + footer[0..16])
//!   [20:24]  table_start   u32 LE  (abs file offset where table begins)
//!   [24:32]  reserved       8 × 0
//!
//! [table_start .. footer_start]   offset table (n_windows × 16 bytes)
//!   per entry:
//!     u64 LE  abs_offset       (start of window payload's length prefix)
//!     u32 LE  payload_len      (bytes consumed by length+payload)
//!     u32 LE  first_sample_idx (sample index within signal of first sample)
//! ```
//!
//! Bible alignment:
//! - R4  Version-from-day-one: footer carries `LMLFOOT1` magic; future
//!   changes bump the trailing digit, never overload the existing layout.
//! - R7  CRC-32 over both the offset table AND the leading 16 bytes of
//!   the footer guarantees a corrupted footer is detected, not silently
//!   consumed. Mismatch → caller falls back to the slow path (scanning
//!   the existing window-length index).
//! - R26 Strict-typed `OffsetTable::read_from_buffer` returns
//!   `LmlResult<Option<OffsetTable>>`: `Ok(None)` for "no footer present",
//!   `Ok(Some(_))` for "footer parsed", `Err(_)` for "footer present but
//!   malformed".

use lamquant_lml_mcu::crc32::{crc32_update, CRC32_INIT};
use lamquant_lml_mcu::error::{LmlError, LmlResult};

/// Footer magic — identifies the seek-table trailer.
pub const FOOTER_MAGIC: &[u8; 8] = b"LMLFOOT1";

/// Fixed footer size in bytes. Pinned across the lifetime of `LMLFOOT1`;
/// any future format change bumps the magic suffix.
pub const FOOTER_SIZE: usize = 32;

/// Bytes per offset table entry.
pub const ENTRY_SIZE: usize = 16;

/// Hard upper bound on `n_windows` we'll trust from a footer before
/// allocating the offset table. The current `container.rs` already
/// bounds `n_windows` to `u16::MAX` (per the file header field
/// width), but the footer's `u32` field could theoretically claim
/// more — clamp to a sane ceiling to avoid OOM on adversarial input.
const MAX_FOOTER_WINDOWS: u32 = 1 << 20; // 1 Mi windows = 16 MiB table

/// One seekable window entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetEntry {
    /// Absolute file offset where the window's length-prefix begins.
    pub abs_offset: u64,
    /// `len_prefix(4) + payload` bytes total.
    pub payload_len: u32,
    /// Sample index of the first sample in this window's signal.
    pub first_sample_idx: u32,
}

/// Parsed random-access seek table.
#[derive(Debug, Clone)]
pub struct OffsetTable {
    entries: Vec<OffsetEntry>,
}

impl OffsetTable {
    /// Build from an in-order list of entries. Callers (encoder) push
    /// while writing windows; future Phase 0.7 random-access readers
    /// consume.
    pub fn new(entries: Vec<OffsetEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[OffsetEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the window covering `sample` via binary search on
    /// `first_sample_idx`. Returns the index of the last entry whose
    /// `first_sample_idx <= sample`. `None` only if `sample` lies
    /// *before* the first window (well-formed tables have entry 0
    /// start at sample 0, so this is defensive). Samples past the
    /// last window's end return `Some(entries.len() - 1)` — the
    /// caller must check `payload_len` / `first_sample_idx + len`
    /// to know if the sample is actually inside that window.
    pub fn window_for_sample(&self, sample: u32) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        // Standard binary search: find last entry whose first_sample_idx
        // <= sample. If none, the sample is before window 0 (impossible
        // for well-formed tables — first entry has first_sample_idx 0
        // — but we handle it defensively).
        let mut lo = 0usize;
        let mut hi = self.entries.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.entries[mid].first_sample_idx <= sample {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            None
        } else {
            Some(lo - 1)
        }
    }

    /// Inclusive range of windows covering [start, end) sample range.
    /// `None` if start is past the last sample.
    pub fn windows_for_range(
        &self,
        start: u32,
        end_exclusive: u32,
    ) -> Option<core::ops::RangeInclusive<usize>> {
        if end_exclusive <= start || self.entries.is_empty() {
            return None;
        }
        let first = self.window_for_sample(start)?;
        // Last window whose first_sample_idx < end_exclusive.
        let last = self.window_for_sample(end_exclusive.saturating_sub(1))?;
        Some(first..=last)
    }

    /// Serialize footer (offset table + 32-byte trailer) to a sink.
    ///
    /// `current_offset` is the absolute file offset where this serialise
    /// begins; needed for the footer's `table_start` field.
    ///
    /// WRITE-only (ADR 0069 L5(c)), but NOT legacy-encode-only: this is a
    /// nontrivial CRC'd binary seek-table serializer that the LIVE
    /// `write_legacy_recording` path (`lamquant-lossless::legacy_bcs1_container`) also needs —
    /// duplicating it there would itself be a wire-format-fork risk (see
    /// that module's docs). ADR 0069 L8: moved from `legacy-encode` to
    /// `legacy-decode` (available whenever this module is compiled) so the
    /// clean writer can call it without linking the retiring `encode_into`/
    /// `write_into` v1 writer. `read_from_buffer` (below) is the FROZEN
    /// reader half and has always been under `legacy-decode`.
    pub fn write_into<W: std::io::Write + ?Sized>(
        &self,
        sink: &mut W,
        current_offset: u64,
    ) -> LmlResult<()> {
        if self.entries.len() > MAX_FOOTER_WINDOWS as usize {
            return Err(LmlError::InvalidHeader(format!(
                "OffsetTable: {} entries exceeds MAX_FOOTER_WINDOWS {}",
                self.entries.len(),
                MAX_FOOTER_WINDOWS
            )));
        }
        let table_start = current_offset;
        let n_windows = self.entries.len() as u32;
        let table_bytes = n_windows as u64 * ENTRY_SIZE as u64;
        let _footer_start = table_start + table_bytes;

        // Serialize entries to a buffer so we can CRC them, then write
        // both buffer + footer prefix to sink.
        let mut table_buf = Vec::with_capacity(table_bytes as usize);
        for e in &self.entries {
            table_buf.extend_from_slice(&e.abs_offset.to_le_bytes());
            table_buf.extend_from_slice(&e.payload_len.to_le_bytes());
            table_buf.extend_from_slice(&e.first_sample_idx.to_le_bytes());
        }
        debug_assert_eq!(table_buf.len(), table_bytes as usize);

        // Footer leading 16 bytes (magic + len + n_windows). The CRC
        // covers the table bytes AND these 16 leading footer bytes.
        let mut footer_head = [0u8; 16];
        footer_head[0..8].copy_from_slice(FOOTER_MAGIC);
        footer_head[8..12].copy_from_slice(&(FOOTER_SIZE as u32).to_le_bytes());
        footer_head[12..16].copy_from_slice(&n_windows.to_le_bytes());

        let mut crc = CRC32_INIT;
        crc = crc32_update(crc, &table_buf);
        crc = crc32_update(crc, &footer_head);

        // Bounds: table_start must fit in u32 for the footer field.
        let table_start_u32: u32 = table_start.try_into().map_err(|_| {
            LmlError::InvalidHeader(format!(
                "OffsetTable: table_start {table_start} exceeds u32 — file too large for v1 footer"
            ))
        })?;

        sink.write_all(&table_buf).map_err(LmlError::Io)?;
        sink.write_all(&footer_head).map_err(LmlError::Io)?;
        sink.write_all(&crc.to_le_bytes()).map_err(LmlError::Io)?;
        sink.write_all(&table_start_u32.to_le_bytes())
            .map_err(LmlError::Io)?;
        sink.write_all(&[0u8; 8]).map_err(LmlError::Io)?; // reserved

        Ok(())
    }

    /// Parse the footer from a buffer ending in the LMLFOOT1 trailer.
    ///
    /// `buf` may be the whole file *or* a slice covering only
    /// `[table_start .. file_end]`. The table is located by the
    /// `n_windows` count carried in the footer; the footer's
    /// `table_start` field is informational only (telling fresh-stream
    /// callers where to seek) and is NOT cross-checked against
    /// `buf.len()`, so both callers (LmlReader's slice and tests'
    /// whole-file buffers) share one parser.
    ///
    /// Returns:
    ///   - `Ok(None)` — no footer present (buf shorter than footer
    ///     size, or magic mismatch). Caller falls back to the slow path.
    ///   - `Ok(Some(table))` — footer parsed; CRC over table bytes +
    ///     footer leading 16 bytes verified.
    ///   - `Err(_)` — footer present (magic matched) but body is
    ///     malformed (CRC mismatch, n_windows overflow, footer_len
    ///     mismatch, table doesn't fit in buf). Caller surfaces the
    ///     error — Bible R7 / R30, never silently consume corrupt data.
    pub fn read_from_buffer(buf: &[u8]) -> LmlResult<Option<OffsetTable>> {
        if buf.len() < FOOTER_SIZE {
            return Ok(None);
        }
        let footer = &buf[buf.len() - FOOTER_SIZE..];
        if &footer[0..8] != FOOTER_MAGIC {
            return Ok(None);
        }
        let footer_len =
            u32::from_le_bytes([footer[8], footer[9], footer[10], footer[11]]) as usize;
        if footer_len != FOOTER_SIZE {
            return Err(LmlError::InvalidHeader(format!(
                "OffsetTable footer_len {footer_len} != {FOOTER_SIZE}"
            )));
        }
        let n_windows = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]);
        if n_windows > MAX_FOOTER_WINDOWS {
            return Err(LmlError::InvalidHeader(format!(
                "OffsetTable n_windows {n_windows} exceeds MAX_FOOTER_WINDOWS {MAX_FOOTER_WINDOWS}"
            )));
        }
        let claimed_crc = u32::from_le_bytes([footer[16], footer[17], footer[18], footer[19]]);
        let footer_start = buf.len() - FOOTER_SIZE;
        let table_bytes_expected =
            (n_windows as usize)
                .checked_mul(ENTRY_SIZE)
                .ok_or_else(|| {
                    LmlError::InvalidHeader(format!(
                        "OffsetTable: n_windows {n_windows} * ENTRY_SIZE overflows usize"
                    ))
                })?;
        if table_bytes_expected > footer_start {
            return Err(LmlError::InvalidHeader(format!(
                "OffsetTable: table needs {table_bytes_expected} bytes but only \
                 {footer_start} bytes precede footer"
            )));
        }
        let table_start = footer_start - table_bytes_expected;
        if table_start > buf.len() {
            return Err(LmlError::InvalidHeader(format!(
                "OffsetTable: table_start {table_start} past buffer end {}",
                buf.len()
            )));
        }
        let table_buf = &buf[table_start..footer_start];
        // Recompute CRC over table + footer leading 16 bytes.
        let mut crc = CRC32_INIT;
        crc = crc32_update(crc, table_buf);
        crc = crc32_update(crc, &footer[0..16]);
        if crc != claimed_crc {
            return Err(LmlError::CrcMismatch {
                expected: claimed_crc,
                actual: crc,
            });
        }
        let mut entries = Vec::with_capacity(n_windows as usize);
        for w in 0..n_windows as usize {
            let off = w * ENTRY_SIZE;
            let abs_offset = u64::from_le_bytes([
                table_buf[off],
                table_buf[off + 1],
                table_buf[off + 2],
                table_buf[off + 3],
                table_buf[off + 4],
                table_buf[off + 5],
                table_buf[off + 6],
                table_buf[off + 7],
            ]);
            let payload_len = u32::from_le_bytes([
                table_buf[off + 8],
                table_buf[off + 9],
                table_buf[off + 10],
                table_buf[off + 11],
            ]);
            let first_sample_idx = u32::from_le_bytes([
                table_buf[off + 12],
                table_buf[off + 13],
                table_buf[off + 14],
                table_buf[off + 15],
            ]);
            entries.push(OffsetEntry {
                abs_offset,
                payload_len,
                first_sample_idx,
            });
        }
        Ok(Some(OffsetTable::new(entries)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_table(n: u32) -> OffsetTable {
        let entries = (0..n)
            .map(|i| OffsetEntry {
                abs_offset: 100 + (i as u64) * 64,
                payload_len: 64,
                first_sample_idx: i * 250,
            })
            .collect();
        OffsetTable::new(entries)
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_then_read_roundtrip() {
        let table = synth_table(5);
        let mut buf: Vec<u8> = Vec::new();
        // Pretend we already wrote 1000 bytes of file content before
        // the table start.
        let prefix = vec![0xAAu8; 1000];
        buf.extend_from_slice(&prefix);
        table.write_into(&mut buf, prefix.len() as u64).unwrap();
        let recovered = OffsetTable::read_from_buffer(&buf)
            .unwrap()
            .expect("footer must be detected");
        assert_eq!(recovered.entries(), table.entries());
        // write_into returns () (file-end placement is fully determined
        // by sink position, no caller bookkeeping needed).
    }

    #[test]
    fn read_returns_none_when_no_footer() {
        let buf = vec![0u8; 100];
        let r = OffsetTable::read_from_buffer(&buf).unwrap();
        assert!(r.is_none(), "absent footer must be Ok(None), not Err");
    }

    #[test]
    fn read_returns_none_when_buf_smaller_than_footer() {
        let buf = vec![0u8; 10];
        let r = OffsetTable::read_from_buffer(&buf).unwrap();
        assert!(r.is_none());
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_returns_err_on_crc_mismatch() {
        let table = synth_table(3);
        let mut buf: Vec<u8> = vec![0xAA; 500];
        table.write_into(&mut buf, 500).unwrap();
        // Flip a byte inside the offset table — CRC must catch.
        buf[510] ^= 0x01;
        match OffsetTable::read_from_buffer(&buf) {
            Err(LmlError::CrcMismatch { .. }) => {}
            other => panic!("expected CrcMismatch, got {other:?}"),
        }
    }

    #[test]
    fn read_returns_err_on_n_windows_overflow() {
        let mut buf: Vec<u8> = vec![0u8; 200];
        let footer_off = buf.len() - FOOTER_SIZE;
        buf[footer_off..footer_off + 8].copy_from_slice(FOOTER_MAGIC);
        buf[footer_off + 8..footer_off + 12].copy_from_slice(&(FOOTER_SIZE as u32).to_le_bytes());
        buf[footer_off + 12..footer_off + 16]
            .copy_from_slice(&(MAX_FOOTER_WINDOWS + 1).to_le_bytes());
        match OffsetTable::read_from_buffer(&buf) {
            Err(LmlError::InvalidHeader(msg)) => {
                assert!(msg.contains("MAX_FOOTER_WINDOWS"), "got: {msg}");
            }
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_returns_err_on_too_many_windows() {
        let table = OffsetTable::new(
            (0..(MAX_FOOTER_WINDOWS as usize + 1))
                .map(|i| OffsetEntry {
                    abs_offset: i as u64,
                    payload_len: 0,
                    first_sample_idx: 0,
                })
                .collect(),
        );
        let mut buf: Vec<u8> = Vec::new();
        assert!(table.write_into(&mut buf, 0).is_err());
    }

    #[test]
    fn window_for_sample_binary_search() {
        let table = synth_table(10); // first_sample_idx = 0,250,500,750,...
        assert_eq!(table.window_for_sample(0), Some(0));
        assert_eq!(table.window_for_sample(249), Some(0));
        assert_eq!(table.window_for_sample(250), Some(1));
        assert_eq!(table.window_for_sample(499), Some(1));
        assert_eq!(table.window_for_sample(500), Some(2));
        assert_eq!(table.window_for_sample(2499), Some(9));
        assert_eq!(table.window_for_sample(99_999_999), Some(9));
    }

    #[test]
    fn window_for_sample_empty_table_returns_none() {
        let table = OffsetTable::new(vec![]);
        assert_eq!(table.window_for_sample(0), None);
    }

    #[test]
    fn windows_for_range_inclusive() {
        let table = synth_table(10);
        // Sample range [300, 1300) covers windows 1..=5
        let r = table.windows_for_range(300, 1300).unwrap();
        assert_eq!(*r.start(), 1);
        assert_eq!(*r.end(), 5);
    }

    #[test]
    fn windows_for_range_empty_when_start_equals_end() {
        let table = synth_table(5);
        assert!(table.windows_for_range(500, 500).is_none());
    }

    #[test]
    fn read_returns_err_on_footer_len_mismatch() {
        let mut buf: Vec<u8> = vec![0u8; 200];
        let footer_off = buf.len() - FOOTER_SIZE;
        buf[footer_off..footer_off + 8].copy_from_slice(FOOTER_MAGIC);
        buf[footer_off + 8..footer_off + 12].copy_from_slice(&(64u32).to_le_bytes()); // wrong size
        match OffsetTable::read_from_buffer(&buf) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("footer_len")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }
}
