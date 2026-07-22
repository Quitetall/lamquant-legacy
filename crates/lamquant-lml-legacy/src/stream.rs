//! Streaming LML reader — decode window-by-window without loading
//! the entire file.
//!
//! Phase 0.5 made this generic over `R: Read + Seek` so stdin
//! (when reads have been buffered into an in-memory Cursor),
//! `Cursor<Vec<u8>>`, S3 streams that support range requests, etc.
//! can plug in. `LmlReader::open(path)` remains as a back-compat
//! shortcut returning a `LmlReader<BufReader<File>>`.

use crate::offset_table::{OffsetTable, ENTRY_SIZE, FOOTER_MAGIC, FOOTER_SIZE};
use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::lml::{self, MAGIC};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Container header parsed from disk.
pub struct ContainerHeader {
    pub n_channels: usize,
    pub n_windows: usize,
    pub total_samples: usize,
    pub window_size: usize,
    pub metadata: String,
}

/// Streaming LML reader. Reads one window at a time.
///
/// Generic over `R: Read + Seek` so any seekable byte source plugs
/// in. The `open(path)` constructor is the legacy file-only path;
/// `from_source<R>(reader)` is the new entry point for stdin /
/// `Cursor<Vec<u8>>` / S3 with range support.
///
/// Bible R6 — the `R: Read + Seek` bound is the strict-typed
/// boundary; misuse fails to compile (a source without `Seek` can't
/// skip the window index, so the trait bound enforces what the
/// algorithm needs).
pub struct LmlReader<R: Read + Seek> {
    reader: R,
    header: ContainerHeader,
    windows_read: usize,
    /// `LMLFOOT1` seek table parsed from EOF when present. `None` for
    /// legacy files (no footer) — random-access methods fall back to
    /// returning an explicit "needs seek table" error rather than
    /// silently scanning the slow window-length index.
    offset_table: Option<OffsetTable>,
    /// Absolute file offset of the first window payload's length
    /// prefix. Stashed during `from_source` so `seek_to_window(0)`
    /// can reset to sequential-decode start.
    first_payload_pos: u64,
}

impl LmlReader<BufReader<File>> {
    /// Open an LML file for streaming decode. Back-compat shortcut
    /// returning the concrete `LmlReader<BufReader<File>>` type.
    pub fn open(path: &Path) -> LmlResult<Self> {
        let file = File::open(path).map_err(LmlError::Io)?;
        Self::from_source(BufReader::new(file))
    }
}

impl<R: Read + Seek> LmlReader<R> {
    /// Construct a streaming reader from any `Read + Seek` source.
    ///
    /// Phase 0.5 entry point. Consumes the source — the reader owns
    /// it for the duration of decoding. Identical behaviour to
    /// `open(path)` once the source is open.
    ///
    /// Requirements:
    ///   - the source is positioned at the start of the LML container
    ///     (byte 0 of the 32-byte header); a `Cursor<Vec<u8>>` that
    ///     has been advanced past byte 0 will misparse
    ///   - wrap unbuffered sources (raw `File`, network stream) in
    ///     `BufReader` before calling — `from_source` issues many
    ///     small `read_exact` calls and the per-syscall overhead is
    ///     a real perf cliff on unbuffered I/O
    pub fn from_source(mut reader: R) -> LmlResult<Self> {
        // Read 32-byte header
        let mut hdr = [0u8; 32];
        reader.read_exact(&mut hdr).map_err(LmlError::Io)?;

        if &hdr[0..4] != MAGIC {
            return Err(LmlError::InvalidMagic([hdr[0], hdr[1], hdr[2], hdr[3]]));
        }

        let n_channels = u16::from_le_bytes([hdr[6], hdr[7]]) as usize;
        let n_windows = u16::from_le_bytes([hdr[8], hdr[9]]) as usize;
        let total_samples = u32::from_le_bytes([hdr[10], hdr[11], hdr[12], hdr[13]]) as usize;
        let window_size = u16::from_le_bytes([hdr[14], hdr[15]]) as usize;
        let meta_len = u32::from_le_bytes([hdr[22], hdr[23], hdr[24], hdr[25]]) as usize;

        // Read metadata
        let mut meta_buf = vec![0u8; meta_len];
        reader.read_exact(&mut meta_buf).map_err(LmlError::Io)?;
        let metadata = String::from_utf8_lossy(&meta_buf).to_string();

        // Skip window index
        let skip = n_windows as u64 * 4;
        reader
            .seek(SeekFrom::Current(skip as i64))
            .map_err(LmlError::Io)?;

        let first_payload_pos = reader.stream_position().map_err(LmlError::Io)?;

        // Probe EOF for an LMLFOOT1 seek table. Absent footer is fine
        // (legacy file) — random-access methods then error explicitly
        // rather than silently scanning. Malformed footer (magic
        // matched but CRC failed) IS surfaced as an error — Bible R7,
        // never silently consume corrupt data.
        let offset_table = Self::try_read_footer(&mut reader)?;

        // Seek back to the first payload so sequential reads continue
        // to work from window 0 (Bible R29 — additive behaviour, no
        // surprise side effects on the existing iterator path).
        reader
            .seek(SeekFrom::Start(first_payload_pos))
            .map_err(LmlError::Io)?;

        Ok(Self {
            reader,
            header: ContainerHeader {
                n_channels,
                n_windows,
                total_samples,
                window_size,
                metadata,
            },
            windows_read: 0,
            offset_table,
            first_payload_pos,
        })
    }

    /// Attempt to parse the optional `LMLFOOT1` seek table at EOF.
    /// Returns `Ok(None)` if absent (legacy file), `Ok(Some(table))`
    /// on success, `Err(_)` if magic matched but the body is malformed
    /// (CRC mismatch, n_windows overflow, etc.) — never silently
    /// accepts corrupt random-access metadata.
    ///
    /// **Security**: the `table_start` field at footer bytes 20–23 is
    /// NOT covered by the footer's CRC-32 (which spans table bytes +
    /// footer leading 16 bytes only). A corrupted or maliciously
    /// crafted footer could supply a bogus `table_start` and redirect
    /// the seek to anywhere in the file — leading to OOB reads or
    /// huge allocations. We therefore IGNORE that field and derive
    /// the table location from `end - FOOTER_SIZE - n_windows *
    /// ENTRY_SIZE`. The footer field is kept on disk as informational
    /// (diagnostics, future format versions) but never trusted.
    /// (Lamu V4 Pro security review of 4d347b0.)
    fn try_read_footer(reader: &mut R) -> LmlResult<Option<OffsetTable>> {
        let end = reader.seek(SeekFrom::End(0)).map_err(LmlError::Io)?;
        if end < FOOTER_SIZE as u64 {
            return Ok(None);
        }
        reader
            .seek(SeekFrom::End(-(FOOTER_SIZE as i64)))
            .map_err(LmlError::Io)?;
        let mut footer = [0u8; FOOTER_SIZE];
        reader.read_exact(&mut footer).map_err(LmlError::Io)?;
        if &footer[0..8] != FOOTER_MAGIC {
            return Ok(None);
        }
        let n_windows = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]) as u64;
        let table_bytes = n_windows.checked_mul(ENTRY_SIZE as u64).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "LMLFOOT1: n_windows {n_windows} * ENTRY_SIZE overflows u64"
            ))
        })?;
        let combined_len = table_bytes.checked_add(FOOTER_SIZE as u64).ok_or_else(|| {
            LmlError::InvalidHeader("LMLFOOT1: combined_len overflows u64".into())
        })?;
        if combined_len > end {
            return Err(LmlError::InvalidHeader(format!(
                "LMLFOOT1: n_windows {n_windows} requires {combined_len} bytes \
                 but file is only {end} bytes"
            )));
        }
        // Derive the seek target ourselves — do NOT trust the
        // unprotected `table_start` field at footer[20..24].
        let derived_table_start = end - combined_len;
        let mut combined = vec![0u8; combined_len as usize];
        reader
            .seek(SeekFrom::Start(derived_table_start))
            .map_err(LmlError::Io)?;
        reader.read_exact(&mut combined).map_err(LmlError::Io)?;
        OffsetTable::read_from_buffer(&combined)
    }

    /// Returns the parsed `LMLFOOT1` seek table if present.
    pub fn offset_table(&self) -> Option<&OffsetTable> {
        self.offset_table.as_ref()
    }

    /// Random-access seek to window `idx`. Requires the file carry a
    /// seek table (new files do; legacy files do not — those return
    /// `Err(InvalidHeader)` so callers can choose to fall back to
    /// sequential decode).
    ///
    /// After a successful seek, the next call to [`next_window`] /
    /// `next()` decodes window `idx`.
    pub fn seek_to_window(&mut self, idx: usize) -> LmlResult<()> {
        let table = self.offset_table.as_ref().ok_or_else(|| {
            LmlError::InvalidHeader(
                "seek_to_window: file has no LMLFOOT1 seek table (legacy format); \
                 fall back to sequential decode via next_window"
                    .into(),
            )
        })?;
        let entry = table.entries().get(idx).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "seek_to_window: window {idx} out of range (len {})",
                table.len()
            ))
        })?;
        self.reader
            .seek(SeekFrom::Start(entry.abs_offset))
            .map_err(LmlError::Io)?;
        self.windows_read = idx;
        Ok(())
    }

    /// Decode every window that intersects the sample range
    /// `[start, end_exclusive)`. Returns a `Vec<Vec<Vec<i64>>>` —
    /// outer index = window position in the range, inner = the
    /// `[n_ch][T_window]` decoded signal. Caller stitches.
    ///
    /// Requires the seek table; legacy files return `Err`.
    pub fn windows_for_range(
        &mut self,
        start: u32,
        end_exclusive: u32,
    ) -> LmlResult<Vec<Vec<Vec<i64>>>> {
        let range = self
            .offset_table
            .as_ref()
            .ok_or_else(|| {
                LmlError::InvalidHeader(
                    "windows_for_range: file has no LMLFOOT1 seek table (legacy format)".into(),
                )
            })?
            .windows_for_range(start, end_exclusive)
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "windows_for_range: empty / past-EOF range [{start}, {end_exclusive})"
                ))
            })?;
        let mut out = Vec::with_capacity(range.end() - range.start() + 1);
        for w in range {
            self.seek_to_window(w)?;
            match self.next_window() {
                Some(Ok(decoded)) => out.push(decoded),
                Some(Err(e)) => return Err(e),
                None => {
                    return Err(LmlError::InvalidHeader(format!(
                        "windows_for_range: seek_to_window({w}) succeeded but next_window EOF"
                    )))
                }
            }
        }
        Ok(out)
    }

    /// Rewind to sequential-decode start (window 0). Useful after a
    /// random-access seek when the caller wants to resume linear iteration.
    pub fn rewind(&mut self) -> LmlResult<()> {
        self.reader
            .seek(SeekFrom::Start(self.first_payload_pos))
            .map_err(LmlError::Io)?;
        self.windows_read = 0;
        Ok(())
    }

    /// Get container header info.
    pub fn header(&self) -> &ContainerHeader {
        &self.header
    }

    /// Read and decompress the next window. Returns None at EOF.
    pub fn next_window(&mut self) -> Option<LmlResult<Vec<Vec<i64>>>> {
        if self.windows_read >= self.header.n_windows {
            return None;
        }

        // Read payload length
        let mut len_buf = [0u8; 4];
        if let Err(e) = self.reader.read_exact(&mut len_buf) {
            return Some(Err(LmlError::Io(e)));
        }
        let payload_len = u32::from_le_bytes(len_buf) as usize;

        // Read payload
        let mut payload = vec![0u8; payload_len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            return Some(Err(LmlError::Io(e)));
        }

        self.windows_read += 1;
        Some(lml::decompress(&payload))
    }
}

impl<R: Read + Seek> Iterator for LmlReader<R> {
    type Item = LmlResult<Vec<Vec<i64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "legacy-encode")]
    use crate::container;
    #[cfg(feature = "legacy-encode")]
    use lamquant_lml_mcu::lpc::LpcMode;

    #[cfg(feature = "legacy-encode")]
    fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut sig = vec![Vec::with_capacity(t); n_ch];
        for ch in 0..n_ch {
            for _ in 0..t {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                sig[ch].push(((state >> 33) as i32) as i64 % 8000);
            }
        }
        sig
    }

    #[test]
    fn open_rejects_invalid_magic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"XXXX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        match LmlReader::open(tmp.path()) {
            Err(LmlError::InvalidMagic(_)) => {}
            Err(e) => panic!("expected InvalidMagic, got {:?}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn open_rejects_truncated_header() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"LML1\x01\x00").unwrap();
        match LmlReader::open(tmp.path()) {
            Err(LmlError::Io(_)) => {}
            Err(e) => panic!("expected Io error, got {:?}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn open_rejects_missing_file() {
        let path = std::path::Path::new("/nonexistent/path/should/not/exist.lml");
        match LmlReader::open(path) {
            Err(LmlError::Io(_)) => {}
            Err(e) => panic!("expected Io error, got {:?}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn streams_one_window_at_a_time() {
        let sig = synth_signal(4, 512, 7);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        container::write_file(tmp.path(), &sig, 250.0, 256, 0, "{}").unwrap();

        let mut reader = LmlReader::open(tmp.path()).unwrap();
        assert_eq!(reader.header().n_channels, 4);
        assert_eq!(reader.header().total_samples, 512);
        assert_eq!(reader.header().window_size, 256);

        let n_windows_expected = reader.header().n_windows;
        let mut count = 0;
        for window in &mut reader {
            let w = window.unwrap();
            assert_eq!(w.len(), 4, "window has wrong channel count");
            count += 1;
        }
        assert_eq!(count, n_windows_expected);
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn next_returns_none_after_eof() {
        let sig = synth_signal(2, 128, 11);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        container::write_file(tmp.path(), &sig, 250.0, 64, 0, "{}").unwrap();

        let mut reader = LmlReader::open(tmp.path()).unwrap();
        let total = reader.header().n_windows;
        for _ in 0..total {
            assert!(reader.next_window().is_some());
        }
        assert!(reader.next_window().is_none());
        assert!(reader.next_window().is_none());
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn iterator_yields_full_signal_in_order() {
        let sig = synth_signal(3, 256, 13);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        container::write_file(tmp.path(), &sig, 250.0, 128, 0, "{}").unwrap();

        let mut reconstructed: Vec<Vec<i64>> = vec![Vec::new(); 3];
        let reader = LmlReader::open(tmp.path()).unwrap();
        for window in reader {
            let w = window.unwrap();
            for (ch, samples) in w.iter().enumerate() {
                reconstructed[ch].extend_from_slice(samples);
            }
        }
        for ch in 0..3 {
            assert_eq!(&reconstructed[ch][..256], &sig[ch][..]);
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn from_source_cursor_streams_one_window_at_a_time() {
        // Phase 0.5: stream from an in-memory Cursor (the LmlSource
        // dispatch path will pass exactly this kind of source).
        let sig = synth_signal(2, 200, 17);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 100, 0, "{}", LpcMode::default()).unwrap();
        let cursor = std::io::Cursor::new(sink);
        let mut reader = LmlReader::from_source(cursor).unwrap();
        assert_eq!(reader.header().n_channels, 2);
        let n_windows = reader.header().n_windows;
        let mut count = 0;
        for w in &mut reader {
            let _ = w.unwrap();
            count += 1;
        }
        assert_eq!(count, n_windows);
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn bogus_table_start_field_is_ignored_for_security() {
        // Lamu V4 Pro security finding: footer bytes 20-23 (table_start)
        // are NOT covered by the CRC. A maliciously crafted footer
        // could redirect the seek to read arbitrary file regions or
        // trigger huge allocs. Phase 0.7 derives table_start from
        // n_windows + footer_size and ignores the on-disk field.
        let sig = synth_signal(1, 256, 1);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        // Corrupt the unprotected table_start field. If the reader
        // still trusts it, OffsetTable::read_from_buffer will fail
        // CRC and the open will error. If the reader correctly
        // ignores it (derives the target itself), the open succeeds.
        let footer_off = sink.len() - 32;
        sink[footer_off + 20] = 0xDE;
        sink[footer_off + 21] = 0xAD;
        sink[footer_off + 22] = 0xBE;
        sink[footer_off + 23] = 0xEF;
        let cursor = std::io::Cursor::new(sink);
        let reader = LmlReader::from_source(cursor)
            .expect("open must succeed — table_start is unprotected and must be ignored");
        assert!(reader.offset_table().is_some());
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn offset_table_loaded_for_new_files() {
        // Phase 0.7: new files (with LMLFOOT1 footer) must populate
        // the offset_table at open time.
        let sig = synth_signal(2, 384, 1);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let cursor = std::io::Cursor::new(sink);
        let reader = LmlReader::from_source(cursor).unwrap();
        let table = reader
            .offset_table()
            .expect("Phase 0.6 files must carry seek table");
        assert_eq!(table.len(), 3, "384 samples / 128-sample windows = 3");
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn seek_to_window_decodes_only_target() {
        let sig = synth_signal(3, 384, 7);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let cursor = std::io::Cursor::new(sink);
        let mut reader = LmlReader::from_source(cursor).unwrap();
        // Seek to window 2 (samples 256..384), decode, expect byte-exact.
        reader.seek_to_window(2).unwrap();
        let w = reader.next_window().unwrap().unwrap();
        for ch in 0..3 {
            assert_eq!(w[ch], sig[ch][256..384], "channel {ch} window 2 drift");
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn seek_to_window_out_of_range_errors() {
        let sig = synth_signal(1, 128, 13);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let mut reader = LmlReader::from_source(std::io::Cursor::new(sink)).unwrap();
        match reader.seek_to_window(99) {
            Err(LmlError::InvalidHeader(msg)) => {
                assert!(msg.contains("out of range"), "got: {msg}");
            }
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn windows_for_range_returns_only_intersecting_windows() {
        let sig = synth_signal(2, 512, 23);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let mut reader = LmlReader::from_source(std::io::Cursor::new(sink)).unwrap();
        // 4 windows total (128 each). Range [150, 400) covers windows 1, 2, 3.
        let windows = reader.windows_for_range(150, 400).unwrap();
        assert_eq!(windows.len(), 3, "should pull 3 windows");
        // Per-window samples concatenated == sig[128..512]
        let mut stitched: Vec<Vec<i64>> = vec![Vec::new(); 2];
        for w in &windows {
            for ch in 0..2 {
                stitched[ch].extend_from_slice(&w[ch]);
            }
        }
        for ch in 0..2 {
            assert_eq!(stitched[ch], sig[ch][128..512]);
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn rewind_resets_to_window_zero() {
        let sig = synth_signal(1, 256, 31);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let mut reader = LmlReader::from_source(std::io::Cursor::new(sink)).unwrap();
        reader.seek_to_window(1).unwrap();
        let _ = reader.next_window().unwrap().unwrap();
        reader.rewind().unwrap();
        // First window after rewind should be window 0.
        let w0 = reader.next_window().unwrap().unwrap();
        assert_eq!(w0[0], sig[0][0..128]);
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn from_source_recovers_full_signal_byte_identical() {
        // Verify file path and source path produce identical decode.
        let sig = synth_signal(3, 256, 31);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let cursor = std::io::Cursor::new(sink);
        let mut reconstructed: Vec<Vec<i64>> = vec![Vec::new(); 3];
        let reader = LmlReader::from_source(cursor).unwrap();
        for window in reader {
            let w = window.unwrap();
            for (ch, samples) in w.iter().enumerate() {
                reconstructed[ch].extend_from_slice(samples);
            }
        }
        for ch in 0..3 {
            assert_eq!(&reconstructed[ch][..256], &sig[ch][..]);
        }
    }
}
