//! LML file container — 32-byte header, JSON metadata, window payloads.
//!
//! Spec: specs/lml-format-v1.md Section 2

// ADR 0069 L5(c): `LosslessMode` (write-only stamp), `LpcMode` (write-fn
// signatures) and `OffsetEntry`/`OffsetTable` (footer construction) are
// consumed ONLY by the encode half below — gated with it. `LmlError`/
// `LmlError`/`LmlResult`/`lml`/`Path` are shared by read+write and stay
// ungated (WATCH #1). `MAGIC` is write/test-only — the non-test read path
// validates the magic via literal bytes, so it rides with the encode half
// (+ the header-crafting read tests).
#[cfg(feature = "legacy-encode")]
use crate::offset_table::{OffsetEntry, OffsetTable};
#[cfg(feature = "legacy-encode")]
use lamquant_lml_mcu::deployment::LosslessMode;
use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::lml;
#[cfg(any(feature = "legacy-encode", test))]
use lamquant_lml_mcu::lml::MAGIC;
#[cfg(feature = "legacy-encode")]
use lamquant_lml_mcu::lpc::LpcMode;
use std::path::Path;
// VERSION_MAJOR/MINOR are written by `encode_into` (encode half) and also
// consumed by pure-READ tests that hand-craft header fixtures — so they ride
// with `legacy-encode` OR `test` (the non-test read path validates the version
// via literal bytes, never these symbols).
#[cfg(any(feature = "legacy-encode", test))]
const VERSION_MAJOR: u8 = 1;
#[cfg(any(feature = "legacy-encode", test))]
const VERSION_MINOR: u8 = 0;

/// Header flag bit 0 — when set, the file carries an `LMLFOOT1` seek
/// table at EOF. Old readers ignore the flag and silently skip the
/// footer; new readers consult it for O(log n) random access.
///
/// WRITE-only: only `encode_into` sets the bit. The read side never
/// branches on it (readers unconditionally probe EOF for the footer
/// via `stream::LmlReader::try_read_footer` / `OffsetTable::read_from_buffer`).
#[cfg(feature = "legacy-encode")]
const FLAG_HAS_FOOTER: u8 = 0b0000_0001;

#[cfg(feature = "legacy-encode")]
fn lossless_mode_for_lpc_mode(mode: LpcMode) -> LosslessMode {
    match mode {
        LpcMode::Fixed => LosslessMode::Mcu,
        LpcMode::Adaptive { .. } | LpcMode::Anytime { .. } => LosslessMode::Basestation,
    }
}

#[cfg(feature = "legacy-encode")]
fn metadata_with_codec_mode(
    metadata_json: &str,
    lpc_mode: LpcMode,
    delta: Option<u64>,
    target_bps: Option<f64>,
) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(metadata_json) else {
        return metadata_json.to_owned();
    };
    let Some(object) = value.as_object_mut() else {
        return metadata_json.to_owned();
    };
    if let Some(tb) = target_bps {
        object.insert(
            "codec_mode".to_owned(),
            serde_json::Value::String("target_bps_lossy".to_owned()),
        );
        object.insert("target_bps".to_owned(), serde_json::json!(tb));
    } else if let Some(d) = delta {
        object.insert(
            "codec_mode".to_owned(),
            serde_json::Value::String("bounded_mae_near_lossless".to_owned()),
        );
        object.insert("max_error".to_owned(), serde_json::json!(d));
    } else {
        object.insert(
            "codec_mode".to_owned(),
            serde_json::Value::String("lossless".to_owned()),
        );
        object.insert(
            "lossless_mode".to_owned(),
            serde_json::Value::String(lossless_mode_for_lpc_mode(lpc_mode).as_str().to_owned()),
        );
    }
    object.insert(
        "lpc_mode".to_owned(),
        serde_json::Value::String(format!("{lpc_mode:?}")),
    );
    serde_json::to_string(&value).unwrap_or_else(|_| metadata_json.to_owned())
}

/// Write a complete LML v1 container file (default LPC mode).
///
/// Thin wrapper over [`write_file_with_mode`] using
/// [`LpcMode::default()`] — `Anytime` with no deadline, which behaves
/// like pure adaptive on host. Streaming consumers should use
/// `write_file_with_mode` directly to pass an explicit deadline.
#[cfg(feature = "legacy-encode")]
pub fn write_file(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
) -> LmlResult<ContainerStats> {
    write_file_with_mode(
        path,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        LpcMode::default(),
    )
}

/// Write a complete LML v1 container file with explicit LPC mode.
#[cfg(feature = "legacy-encode")]
pub fn write_file_with_mode(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    let mut f = std::fs::File::create(path).map_err(LmlError::Io)?;
    let stats = write_into(
        &mut f,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
    )?;
    // Durability — flush the kernel page cache to disk before the
    // function returns. Without this, a crash between return and
    // background flush leaves the file partially written. The old
    // path didn't sync either, but the new path makes it explicit
    // (Bible R27 backup strategy: every committed write is durable).
    f.sync_all().map_err(LmlError::Io)?;
    Ok(stats)
}

/// Write a complete LML container file in **bounded-MAE near-lossless** mode
/// (ADR 0051 track 2): every window is encoded with a guaranteed
/// `max|orig-recon| <= delta`. Produces a standard `.lml` container, so all
/// readers (`decode`, `export`, `info`) work unchanged — only the per-window
/// payloads carry the track-2 mode flag.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "legacy-encode")]
pub fn write_file_bounded_mae(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    delta: u64,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    let mut f = std::fs::File::create(path).map_err(LmlError::Io)?;
    let mut cw = CountingWriter {
        inner: &mut f,
        count: 0,
    };
    let n_windows = encode_into(
        &mut cw,
        signal,
        sample_rate,
        window_size,
        0, // noise_bits unused in bounded mode
        metadata_json,
        lpc_mode,
        Some(delta),
        None,
    )?;
    let compressed_size = cw.count as usize;
    f.sync_all().map_err(LmlError::Io)?;
    let n_ch = signal.len();
    let total_samples = signal.first().map(|ch| ch.len()).unwrap_or(0);
    let raw_size = n_ch * total_samples * 2;
    Ok(ContainerStats {
        n_windows,
        n_channels: n_ch,
        total_samples,
        compressed_size,
        raw_size,
        cr: if compressed_size > 0 {
            raw_size as f64 / compressed_size as f64
        } else {
            0.0
        },
        duration_s: if sample_rate > 0.0 {
            total_samples as f64 / sample_rate
        } else {
            0.0
        },
    })
}

/// Write a complete LML container file in **target-BPS rate-controlled lossy**
/// mode (ADR 0051 track 2 P2): each window is rate-controlled to `target_bps`
/// (minimize distortion s.t. the bits-per-sample ceiling). Standard `.lml`
/// container — all readers work unchanged.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "legacy-encode")]
pub fn write_file_target_bps(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    target_bps: f64,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    let mut f = std::fs::File::create(path).map_err(LmlError::Io)?;
    let mut cw = CountingWriter {
        inner: &mut f,
        count: 0,
    };
    let n_windows = encode_into(
        &mut cw,
        signal,
        sample_rate,
        window_size,
        0,
        metadata_json,
        lpc_mode,
        None,
        Some(target_bps),
    )?;
    let compressed_size = cw.count as usize;
    f.sync_all().map_err(LmlError::Io)?;
    let n_ch = signal.len();
    let total_samples = signal.first().map(|ch| ch.len()).unwrap_or(0);
    let raw_size = n_ch * total_samples * 2;
    Ok(ContainerStats {
        n_windows,
        n_channels: n_ch,
        total_samples,
        compressed_size,
        raw_size,
        cr: if compressed_size > 0 {
            raw_size as f64 / compressed_size as f64
        } else {
            0.0
        },
        duration_s: if sample_rate > 0.0 {
            total_samples as f64 / sample_rate
        } else {
            0.0
        },
    })
}

/// Write a complete LML v1 container into any [`std::io::Write`] sink.
///
/// Phase 0.5 entry point for stdout / S3 / HTTP / arbitrary sinks.
/// Internally tracks bytes written via a private CountingWriter so
/// `ContainerStats::compressed_size` is computed deterministically
/// (no `fs::metadata` round-trip; no race with concurrent unlink).
///
/// Bible R31: identical input + sink type → identical bytes on the
/// wire. R33 backpressure: the sink's `Write::write` is the natural
/// attachment point; Phase 0.7 streams windows incrementally so a
/// slow sink can apply backpressure window-by-window instead of all
/// at once at the end.
#[cfg(feature = "legacy-encode")]
pub fn write_into<W: std::io::Write>(
    sink: &mut W,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let mut cw = CountingWriter {
        inner: sink,
        count: 0,
    };
    let n_windows = encode_into(
        &mut cw,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
        None,
        None,
    )?;
    let compressed_size = cw.count as usize;
    let n_ch = signal.len();
    let total_samples = signal.first().map(|ch| ch.len()).unwrap_or(0);
    let raw_size = n_ch * total_samples * 2;
    Ok(ContainerStats {
        n_windows,
        n_channels: n_ch,
        total_samples,
        compressed_size,
        raw_size,
        cr: if compressed_size > 0 {
            raw_size as f64 / compressed_size as f64
        } else {
            0.0
        },
        duration_s: if sample_rate > 0.0 {
            total_samples as f64 / sample_rate
        } else {
            0.0
        },
    })
}

/// Internal: byte-counting writer wrapper used by `write_into` so the
/// `ContainerStats::compressed_size` field stays accurate without a
/// follow-up `fs::metadata` round-trip (Audit-2026-05-11 Fix-#53 was
/// the race window in the old impl).
#[cfg(feature = "legacy-encode")]
struct CountingWriter<'w, W: std::io::Write + ?Sized> {
    inner: &'w mut W,
    count: u64,
}

#[cfg(feature = "legacy-encode")]
impl<'w, W: std::io::Write + ?Sized> std::io::Write for CountingWriter<'w, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Internal: write the entire container into a sink. Used by both
/// `write_file_with_mode` (file path) and `write_into` (generic
/// sink). Refactored out of the former so the sink-generic path
/// reuses the same encode loop with zero copy-paste.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "legacy-encode")]
fn encode_into<W: std::io::Write + ?Sized>(
    sink: &mut W,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
    // ADR 0051 track 2: Some(δ) → per-window bounded-MAE near-lossless
    // (mutually exclusive with noise_bits); None → standard path.
    delta: Option<u64>,
    // ADR 0051 track 2 P2: Some(bps) → per-window target-BPS lossy (takes
    // precedence over delta/noise_bits). None → not target-BPS.
    target_bps: Option<f64>,
) -> LmlResult<usize /* n_windows */> {
    let n_ch = signal.len();
    if n_ch == 0 {
        return Err(LmlError::InvalidHeader("0 channels".into()));
    }
    let total_samples = signal[0].len();
    if total_samples == 0 {
        return Err(LmlError::InvalidHeader("0 samples".into()));
    }

    // Audit-2026-05-11 Fix-#17: reject NaN/Inf sample_rate before
    // letting them cast silently to 0 or usize::MAX. The cast
    // `(f64 * f64) as usize` is saturating-to-0 on NaN and saturating-
    // to-usize::MAX on +Inf; either path corrupts downstream `n_windows`
    // arithmetic and may overflow the u16 header field.
    if !sample_rate.is_finite() {
        return Err(LmlError::InvalidHeader(format!(
            "sample_rate {sample_rate} is not finite"
        )));
    }
    if sample_rate <= 0.0 {
        return Err(LmlError::InvalidHeader(format!(
            "sample_rate {sample_rate} must be positive"
        )));
    }

    let actual_window_f = window_size as f64 * sample_rate / 250.0;
    if !actual_window_f.is_finite() || actual_window_f < 0.0 {
        return Err(LmlError::InvalidHeader(format!(
            "computed actual_window {actual_window_f} is not a valid window size"
        )));
    }
    // Cap at u16::MAX (the wire-format field width) instead of erroring.
    // High-rate research signals (e.g. 128 kHz × 10 s nominal = 5.1M samples)
    // would otherwise refuse to encode. The shorter window costs a bit of
    // per-window CRC + LPC-metadata overhead, but the format stays usable
    // across all realistic EEG/research sample rates.
    let actual_window = (actual_window_f as usize).min(u16::MAX as usize);
    if actual_window == 0 {
        return Err(LmlError::InvalidHeader("computed window size is 0".into()));
    }

    // Audit-2026-05-11 Fix-#18: checked_add on `total_samples +
    // actual_window - 1` so a multi-TB EDF (total_samples near
    // usize::MAX) cannot overflow silently.
    let n_windows_raw = total_samples
        .checked_add(actual_window)
        .and_then(|s| s.checked_sub(1))
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "n_windows calculation overflow: total_samples={} actual_window={}",
                total_samples, actual_window
            ))
        })?
        / actual_window;
    let n_windows = n_windows_raw.max(1);
    let metadata_json = metadata_with_codec_mode(metadata_json, lpc_mode, delta, target_bps);
    let meta_bytes = metadata_json.as_bytes();
    let sample_rate_mhz = (sample_rate * 1000.0) as u32;

    // Validate u16 header fields won't truncate
    if n_windows > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(format!(
            "too many windows ({}) — max {} for LML header",
            n_windows,
            u16::MAX
        )));
    }
    // (actual_window was already clamped to u16::MAX above; this is the
    // invariant the wire format relies on.)
    debug_assert!(actual_window <= u16::MAX as usize);

    // Compress each window
    // Per-window encode. Dispatch through the process-wide
    // `ComputeBackend` selector (set by CLI/TUI at startup) so that
    // on host the rayon-parallel + AVX2 path runs by default. Output
    // is byte-identical regardless of backend; only wall-clock
    // differs. Firmware builds compile out the Desktop arm via cfg.
    let backend = lamquant_lml_desktop::backend::global_backend();
    let mut window_payloads: Vec<Vec<u8>> = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let start = w * actual_window;
        let end = (start + actual_window).min(total_samples);
        let window: Vec<Vec<i64>> = signal.iter().map(|ch| ch[start..end].to_vec()).collect();
        // ADR 0051 track 2: when `delta` is set, every window is a bounded-MAE
        // packet (closed-loop DPCM, MAE<=delta per window => overall). Backend
        // is irrelevant for the bounded path (integer-only, no SIMD divergence).
        let compressed = if let Some(tb) = target_bps {
            lml::compress_target_bps(&window, tb, lpc_mode)?
        } else if let Some(d) = delta {
            lml::compress_bounded_mae(&window, d, lpc_mode)?
        } else {
            match backend {
                lamquant_lml_desktop::backend::ComputeBackend::Firmware => {
                    lml::compress_with_mode(&window, noise_bits, lpc_mode)?
                }
                // ADR 0058 carve-full: the Desktop owner selects its Rayon
                // execution profile through one byte-equal backend interface.
                lamquant_lml_desktop::backend::ComputeBackend::Desktop => {
                    lamquant_lml_desktop::backend::compress_with_backend(
                        &window,
                        noise_bits,
                        lpc_mode,
                        lamquant_lml_desktop::backend::ComputeBackend::Desktop,
                    )?
                }
            }
        };
        window_payloads.push(compressed);
    }

    // 32-byte header (spec Section 2.1)
    sink.write_all(MAGIC).map_err(LmlError::Io)?;
    sink.write_all(&[VERSION_MAJOR]).map_err(LmlError::Io)?; // byte 4
    sink.write_all(&[VERSION_MINOR]).map_err(LmlError::Io)?; // byte 5
    sink.write_all(&(n_ch as u16).to_le_bytes())
        .map_err(LmlError::Io)?; // 6-7
    sink.write_all(&(n_windows as u16).to_le_bytes())
        .map_err(LmlError::Io)?; // 8-9
    sink.write_all(&(total_samples as u32).to_le_bytes())
        .map_err(LmlError::Io)?; // 10-13
    sink.write_all(&(actual_window as u16).to_le_bytes())
        .map_err(LmlError::Io)?; // 14-15
    sink.write_all(&sample_rate_mhz.to_le_bytes())
        .map_err(LmlError::Io)?; // 16-19
    sink.write_all(&[16u8]).map_err(LmlError::Io)?; // 20: bit_depth (default 16)
                                                    // 21: flags. FLAG_HAS_FOOTER (bit 0) is set unconditionally — the
                                                    // footer is additive and old readers ignore the bit. Phase 0.6 wire
                                                    // format extension.
    sink.write_all(&[FLAG_HAS_FOOTER]).map_err(LmlError::Io)?;
    sink.write_all(&(meta_bytes.len() as u32).to_le_bytes())
        .map_err(LmlError::Io)?; // 22-25
    sink.write_all(&[0u8; 2]).map_err(LmlError::Io)?; // 26-27: reserved_0
    sink.write_all(&[0u8; 4]).map_err(LmlError::Io)?; // 28-31: reserved_1

    // Metadata JSON
    sink.write_all(meta_bytes).map_err(LmlError::Io)?;

    // Window index (u32 offsets) — validate total payload fits in u32
    let total_payload: u64 = window_payloads.iter().map(|p| p.len() as u64 + 4).sum();
    if total_payload > u32::MAX as u64 {
        return Err(LmlError::InvalidHeader(format!(
            "Total payload {} bytes exceeds u32 max for window index",
            total_payload
        )));
    }
    let mut offset = 0u32;
    for payload in &window_payloads {
        sink.write_all(&offset.to_le_bytes())
            .map_err(LmlError::Io)?;
        offset += (payload.len() as u32) + 4;
    }

    // The header is fixed 32 bytes; metadata is `meta_bytes.len()`;
    // window index is `n_windows * 4`. The first payload's length
    // prefix begins immediately after.
    let first_payload_abs = 32u64 + meta_bytes.len() as u64 + n_windows as u64 * 4;

    // Window payloads (length-prefixed) — track absolute offsets so
    // the LMLFOOT1 seek table at EOF carries O(log n) random-access
    // entries (Phase 0.6 wire format extension).
    let mut offset_entries: Vec<OffsetEntry> = Vec::with_capacity(n_windows);
    let mut payload_abs = first_payload_abs;
    for (w, payload) in window_payloads.iter().enumerate() {
        let payload_len_with_prefix = 4u32 + payload.len() as u32;
        offset_entries.push(OffsetEntry {
            abs_offset: payload_abs,
            payload_len: payload_len_with_prefix,
            first_sample_idx: (w * actual_window) as u32,
        });
        sink.write_all(&(payload.len() as u32).to_le_bytes())
            .map_err(LmlError::Io)?;
        sink.write_all(payload).map_err(LmlError::Io)?;
        payload_abs += payload_len_with_prefix as u64;
    }

    // Append LMLFOOT1 seek table + 32-byte footer at EOF. Old readers
    // stop after the last window payload and silently skip these;
    // new readers parse them for random access (Phase 0.7).
    let table = OffsetTable::new(offset_entries);
    table.write_into(sink, payload_abs)?;

    sink.flush().map_err(LmlError::Io)?;
    Ok(n_windows)
}

/// Parsed container header — shape + offsets so callers can iterate
/// windows themselves (`cmd_recover` skips CRC-failed windows, so it
/// cannot use `read_file` which fails on the first CRC mismatch).
///
/// Audit-2026-05-11 Fix-C1: extracted from `read_file` so `cmd_recover`
/// reads the same auto-detected header layout as `read_file` instead of
/// hand-parsing an 18-byte layout that `container::write_file` no longer
/// emits.
pub struct ContainerHeader {
    pub n_ch: usize,
    pub n_windows: usize,
    pub total_samples: usize,
    pub window_size: usize,
    pub metadata: String,
    /// First byte after the variable-length window-length-index section.
    pub payload_start: usize,
}

/// Parse a container header, auto-detecting 32/20/18-byte variants.
/// Returns the parsed shape + the byte offset where per-window payloads
/// begin. Caller is responsible for iterating windows + handling per-
/// window CRC failures (e.g. `cmd_recover` skips failed windows, while
/// `read_file` propagates the first failure).
pub fn parse_header(data: &[u8]) -> LmlResult<ContainerHeader> {
    if data.len() < 18 {
        return Err(LmlError::Truncated {
            expected: 18,
            actual: data.len(),
            context: "container header",
        });
    }

    if &data[0..3] != b"LML" {
        return Err(LmlError::InvalidMagic([data[0], data[1], data[2], data[3]]));
    }
    if data[3] != b'1' {
        if data[3].is_ascii_digit() {
            return Err(LmlError::UnsupportedVersion(data[3]));
        }
        return Err(LmlError::InvalidMagic([data[0], data[1], data[2], data[3]]));
    }

    let probe = u16::from_le_bytes([data[4], data[5]]);

    let (n_ch, n_windows, total_samples, window_size, meta_len, hdr_end) = if probe == 1 {
        if data.len() >= 32 && (data[20] == 16 || data[20] == 24 || data[20] == 32) {
            // 32-byte header (current write_file output)
            let n_ch = u16::from_le_bytes([data[6], data[7]]) as usize;
            let n_win = u16::from_le_bytes([data[8], data[9]]) as usize;
            let total = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
            let ws = u16::from_le_bytes([data[14], data[15]]) as usize;
            let ml = u32::from_le_bytes([data[22], data[23], data[24], data[25]]) as usize;
            (n_ch, n_win, total, ws, ml, 32usize)
        } else {
            // 20-byte header. Guard the read: a `probe==1` buffer truncated to
            // 18–19 bytes reaches here (it failed the 32-byte branch) and would
            // index `data[16..20]` out of bounds — return Truncated, never panic.
            if data.len() < 20 {
                return Err(LmlError::Truncated {
                    expected: 20,
                    actual: data.len(),
                    context: "container header (20-byte)",
                });
            }
            let n_ch = u16::from_le_bytes([data[6], data[7]]) as usize;
            let n_win = u16::from_le_bytes([data[8], data[9]]) as usize;
            let total = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
            let ws = u16::from_le_bytes([data[14], data[15]]) as usize;
            let ml = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
            (n_ch, n_win, total, ws, ml, 20usize)
        }
    } else {
        let n_ch = probe as usize;
        let n_win = u16::from_le_bytes([data[6], data[7]]) as usize;
        let total = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let ws = u16::from_le_bytes([data[12], data[13]]) as usize;
        let ml = u32::from_le_bytes([data[14], data[15], data[16], data[17]]) as usize;
        (n_ch, n_win, total, ws, ml, 18usize)
    };

    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("channel count: {}", n_ch)));
    }
    if total_samples == 0 {
        return Err(LmlError::InvalidHeader("zero samples".into()));
    }
    if n_windows == 0 {
        return Err(LmlError::InvalidHeader("zero windows".into()));
    }

    if hdr_end + meta_len > data.len() {
        return Err(LmlError::Truncated {
            expected: hdr_end + meta_len,
            actual: data.len(),
            context: "metadata",
        });
    }
    let metadata = core::str::from_utf8(&data[hdr_end..hdr_end + meta_len])
        .map_err(|e| LmlError::InvalidHeader(format!("metadata is not valid UTF-8: {e}")))?
        .to_string();

    // Skip the window-length index (n_windows × u32 LE offsets).
    let payload_start = hdr_end + meta_len + n_windows * 4;

    Ok(ContainerHeader {
        n_ch,
        n_windows,
        total_samples,
        window_size,
        metadata,
        payload_start,
    })
}

/// Read an LML container from a file, decompress all windows.
///
/// Auto-detects header format: 32-byte (current), 20-byte (v1 with
/// version), 18-byte (no version). Thin wrapper over [`read_bytes`]
/// for callers that have a file path.
pub fn read_file(path: &Path) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let data = std::fs::read(path).map_err(LmlError::Io)?;
    read_bytes(&data)
}

/// Decode every window directly into a pre-allocated `[n_ch, total_samples]`
/// `f32` row-major buffer, applying per-channel digital→physical
/// calibration in place.
///
/// `calib` is the same row-major buffer view, indexed
/// `[ch * 4 + {0:dig_min, 1:dig_max, 2:phys_min, 3:phys_max}]`. Length
/// must equal `n_ch * 4`.
///
/// Peak transient = `n_ch * window_size * 8` (single decoded i64 window)
/// + the output f32 buffer itself. For an 8 hr 27-ch TUEG file this is
///   roughly 5 MB + 4.7 GB instead of the 18 GB peak via
///   `read_bytes -> Vec<Vec<i64>> -> np.asarray -> np.float64`.
///
/// Caller pre-allocates the output (e.g. a `PyArray2<f32>` backing
/// slice). Channels whose `dig_range == 0` are emitted as zero (matches
/// the Python `lml_digital_to_float` `safe` branch).
pub fn read_bytes_into_f32_calibrated(
    data: &[u8],
    out: &mut [f32],
    calib: &[f32],
) -> LmlResult<ContainerHeader> {
    let header = parse_header(data)?;
    let n_ch = header.n_ch;
    let total = header.total_samples;
    if out.len() != n_ch * total {
        return Err(LmlError::InvalidHeader(format!(
            "output buffer size mismatch: expected {} got {}",
            n_ch * total,
            out.len()
        )));
    }
    if calib.len() != n_ch * 4 {
        return Err(LmlError::InvalidHeader(format!(
            "calib length {} != n_ch*4 ({})",
            calib.len(),
            n_ch * 4
        )));
    }

    // Per-channel scale + offset so we do one mul + one add per sample.
    let mut scale = vec![0.0f32; n_ch];
    let mut offset = vec![0.0f32; n_ch];
    for ch in 0..n_ch {
        let dig_min = calib[ch * 4];
        let dig_max = calib[ch * 4 + 1];
        let phys_min = calib[ch * 4 + 2];
        let phys_max = calib[ch * 4 + 3];
        let dig_range = dig_max - dig_min;
        if dig_range == 0.0 {
            scale[ch] = 0.0;
            offset[ch] = 0.0; // degenerate calibration → emit zero row
        } else {
            scale[ch] = (phys_max - phys_min) / dig_range;
            offset[ch] = phys_min - dig_min * scale[ch];
        }
    }

    let window_size = header.window_size;
    // Walk the window-length index sequentially so we don't depend on
    // LMLFOOT1; this path is the workhorse for full-signal training
    // decode, where we need every window anyway.
    let mut pos = header.payload_start;
    for w in 0..header.n_windows {
        if pos + 4 > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + 4,
                actual: data.len(),
                context: "window length",
            });
        }
        let payload_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + payload_len > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + payload_len,
                actual: data.len(),
                context: "window payload",
            });
        }
        let window = lml::decompress(&data[pos..pos + payload_len])?;
        pos += payload_len;

        // V4 Pro 2026-05-18 review #1+#2 (ensemble + critic): refuse to
        // silently truncate. Header says `n_ch` channels; if any window
        // decompresses to a different count the file is corrupt and we
        // must not silently zero-fill the missing rows (would produce
        // garbage training data without a single warning).
        if window.len() != n_ch {
            return Err(LmlError::InvalidHeader(format!(
                "window {w}: decoded channel count {} != header n_ch {}",
                window.len(),
                n_ch
            )));
        }

        let start = w * window_size;
        for ch in 0..n_ch {
            let src = &window[ch];
            // Last window may legitimately be shorter than `window_size`
            // when total_samples isn't an exact multiple; matches the
            // legacy `read_bytes` behaviour at line 581.
            let copy_len = (src.len()).min(total.saturating_sub(start));
            let dst_off = ch * total + start;
            let s = scale[ch];
            let o = offset[ch];
            // Branchless per-sample fma. Rust autovectorises this on
            // x86_64 with -C opt-level=3 (release).
            for i in 0..copy_len {
                out[dst_off + i] = src[i] as f32 * s + o;
            }
        }
    }

    Ok(header)
}

/// Random-access read of a single window by index, in-memory.
///
/// Uses the in-header window-length index (always present, even
/// for legacy containers without `LMLFOOT1`): the 4 bytes at
/// `payload_start - n_windows*4 + idx*4` give the relative offset
/// from `payload_start` to window `idx`'s `[u32 len][payload]`
/// block. Decompresses ONLY that one window.
///
/// Designed for the LMA-direct training dataloader: cuts peak RSS
/// from `n_ch * total_samples * 8` bytes (full-LMA decode, e.g.
/// 9 GB on an 8 hr 27-ch TUEG file) down to
/// `n_ch * window_size * 8` (e.g. 540 KB per window).
///
/// Returns `(window[n_ch][window_size_actual] i64, header)`. Caller
/// keeps the header for `metadata` + `n_windows` queries.
pub fn read_window_from_bytes(
    data: &[u8],
    window_idx: usize,
) -> LmlResult<(Vec<Vec<i64>>, ContainerHeader)> {
    let header = parse_header(data)?;
    if window_idx >= header.n_windows {
        return Err(LmlError::InvalidHeader(format!(
            "window_idx {} out of range (n_windows {})",
            window_idx, header.n_windows
        )));
    }
    // window-length index sits immediately before payload_start.
    let index_base = header
        .payload_start
        .checked_sub(header.n_windows * 4)
        .ok_or_else(|| LmlError::InvalidHeader("payload_start underflow".into()))?;
    let entry_pos = index_base + window_idx * 4;
    if entry_pos + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: entry_pos + 4,
            actual: data.len(),
            context: "window index entry",
        });
    }
    let rel_off = u32::from_le_bytes([
        data[entry_pos],
        data[entry_pos + 1],
        data[entry_pos + 2],
        data[entry_pos + 3],
    ]) as usize;
    let block_pos = header
        .payload_start
        .checked_add(rel_off)
        .ok_or_else(|| LmlError::InvalidHeader("window block position overflow".into()))?;
    if block_pos + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: block_pos + 4,
            actual: data.len(),
            context: "window length",
        });
    }
    let payload_len = u32::from_le_bytes([
        data[block_pos],
        data[block_pos + 1],
        data[block_pos + 2],
        data[block_pos + 3],
    ]) as usize;
    let payload_start = block_pos + 4;
    if payload_start + payload_len > data.len() {
        return Err(LmlError::Truncated {
            expected: payload_start + payload_len,
            actual: data.len(),
            context: "window payload",
        });
    }
    let window = lml::decompress(&data[payload_start..payload_start + payload_len])?;
    // V4 Pro 2026-05-18 critic #3: refuse silent truncation if a window
    // decompresses to a channel count that disagrees with the header.
    // Caller would otherwise silently get a truncated [shorter_n_ch, T]
    // window when allocating its own PyArray2.
    if window.len() != header.n_ch {
        return Err(LmlError::InvalidHeader(format!(
            "window {window_idx}: decoded channel count {} != header n_ch {}",
            window.len(),
            header.n_ch
        )));
    }
    Ok((window, header))
}

/// Read an LML container from any [`std::io::Read`] source.
///
/// Phase 0.5 entry point for stdin / S3 / HTTP / arbitrary streams.
/// Today the impl buffers the whole stream via `read_to_end` then runs
/// the existing buffer decoder; Phase 0.7 will teach the decoder to
/// consume windows incrementally so streaming sources don't need to
/// fit in memory.
///
/// Partial-read sources (e.g. one-byte-at-a-time pipes, network
/// reads) work correctly because `read_to_end` loops internally.
pub fn read_from<R: std::io::Read>(src: &mut R) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let mut data = Vec::new();
    src.read_to_end(&mut data).map_err(LmlError::Io)?;
    read_bytes(&data)
}

/// Read an LML container directly from in-memory bytes.
///
/// Same parsing semantics as [`read_file`]: auto-detects the header
/// format (32-byte / 20-byte / 18-byte), validates magic + dimensions,
/// decompresses every window, returns `(signal, metadata_json)`.
///
/// Used by the LMA-direct training dataloader, which reads the .lml
/// payload bytes out of an LMA archive (via `lma::read_entry`) and
/// decodes them in-memory without writing a tempfile. Also the shared
/// body of [`read_file`] and [`read_from`].
pub fn read_bytes(data: &[u8]) -> LmlResult<(Vec<Vec<i64>>, String)> {
    if data.len() < 18 {
        return Err(LmlError::Truncated {
            expected: 18,
            actual: data.len(),
            context: "container header",
        });
    }

    // Validate magic
    if &data[0..3] != b"LML" {
        return Err(LmlError::InvalidMagic([data[0], data[1], data[2], data[3]]));
    }
    if data[3] != b'1' {
        if data[3].is_ascii_digit() {
            return Err(LmlError::UnsupportedVersion(data[3]));
        }
        return Err(LmlError::InvalidMagic([data[0], data[1], data[2], data[3]]));
    }

    // Auto-detect header format by probing bytes 4-5.
    // 32-byte: [4]=version_major(1), [5]=version_minor(0) → u16 LE = 0x0001
    // 20-byte: [4:6]=version u16 = 1
    // 18-byte: [4:6]=n_channels u16 (always >= 2 for EEG)
    let probe = u16::from_le_bytes([data[4], data[5]]);

    let (n_ch, n_windows, total_samples, window_size, meta_len, hdr_end) = if probe == 1 {
        // Version field present. Check if 32-byte (has sample_rate_mhz at 16-19)
        // or 20-byte (meta_len at 16-19).
        // Heuristic: in 32-byte, bytes 16-19 = sample_rate_mhz (typically 250000).
        // In 20-byte, bytes 16-19 = meta_len (typically < 10000).
        // Safe distinguisher: 32-byte has bit_depth at [20] = 16 or 24.
        if data.len() >= 32 && (data[20] == 16 || data[20] == 24 || data[20] == 32) {
            // 32-byte header
            let n_ch = u16::from_le_bytes([data[6], data[7]]) as usize;
            let n_win = u16::from_le_bytes([data[8], data[9]]) as usize;
            let total = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
            let ws = u16::from_le_bytes([data[14], data[15]]) as usize;
            let ml = u32::from_le_bytes([data[22], data[23], data[24], data[25]]) as usize;
            (n_ch, n_win, total, ws, ml, 32usize)
        } else {
            // 20-byte header (version u16 + 14 bytes). Guard the read: a `probe==1`
            // buffer truncated to 18–19 bytes reaches here (it failed the 32-byte
            // branch) and would index `data[16..20]` out of bounds — Truncated, not panic.
            if data.len() < 20 {
                return Err(LmlError::Truncated {
                    expected: 20,
                    actual: data.len(),
                    context: "container header (20-byte)",
                });
            }
            let n_ch = u16::from_le_bytes([data[6], data[7]]) as usize;
            let n_win = u16::from_le_bytes([data[8], data[9]]) as usize;
            let total = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
            let ws = u16::from_le_bytes([data[14], data[15]]) as usize;
            let ml = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
            (n_ch, n_win, total, ws, ml, 20usize)
        }
    } else {
        // 18-byte header (no version, probe = n_channels)
        let n_ch = probe as usize;
        let n_win = u16::from_le_bytes([data[6], data[7]]) as usize;
        let total = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let ws = u16::from_le_bytes([data[12], data[13]]) as usize;
        let ml = u32::from_le_bytes([data[14], data[15], data[16], data[17]]) as usize;
        (n_ch, n_win, total, ws, ml, 18usize)
    };

    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("channel count: {}", n_ch)));
    }
    if total_samples == 0 {
        return Err(LmlError::InvalidHeader("zero samples".into()));
    }
    if n_windows == 0 {
        return Err(LmlError::InvalidHeader("zero windows".into()));
    }
    // Bound the signal allocation against the data-implied size. The decode
    // fills window-by-window, so total_samples can never exceed
    // n_windows * window_size. Without this an attacker-set total_samples
    // (u32 from the header) drives a multi-TB
    // `vec![vec![0i64; total_samples]; n_ch]` BEFORE any window is read or
    // bounds-checked. (Mirrors the lml.rs MAX_DECODE_BYTES guard; also
    // catches window_size == 0, which forces max_samples == 0 < total.)
    let max_samples = (n_windows as u64)
        .checked_mul(window_size as u64)
        .ok_or_else(|| LmlError::InvalidHeader("n_windows * window_size overflows u64".into()))?;
    if total_samples as u64 > max_samples {
        return Err(LmlError::InvalidHeader(format!(
            "total_samples {total_samples} exceeds n_windows*window_size {max_samples}"
        )));
    }

    let mut pos = hdr_end;
    if pos + meta_len > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + meta_len,
            actual: data.len(),
            context: "metadata",
        });
    }
    // Audit-2026-05-11 Fix-#19: strict UTF-8. Metadata is JSON; invalid
    // UTF-8 means the file is corrupted, not that we should silently
    // substitute U+FFFD and propagate garbage to JSON consumers.
    let metadata = core::str::from_utf8(&data[pos..pos + meta_len])
        .map_err(|e| LmlError::InvalidHeader(format!("metadata is not valid UTF-8: {e}")))?
        .to_string();
    pos += meta_len;

    // Skip window index
    pos += n_windows * 4;

    // Decompress windows
    let mut signal = vec![vec![0i64; total_samples]; n_ch];
    for w in 0..n_windows {
        if pos + 4 > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + 4,
                actual: data.len(),
                context: "window length",
            });
        }
        let payload_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + payload_len > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + payload_len,
                actual: data.len(),
                context: "window payload",
            });
        }
        let window = lml::decompress(&data[pos..pos + payload_len])?;
        pos += payload_len;

        // Refuse to silently truncate/zero-fill on a channel-count
        // mismatch — header says `n_ch`, and a window decoding to a
        // different count is corruption, not a partial read. The two
        // sibling decoders (read_bytes_into_f32_calibrated, read_window)
        // already reject this; `read_bytes` previously used
        // `n_ch.min(window.len())`, which dropped extra rows and left
        // missing rows zero-filled — silent garbage. Match the siblings.
        if window.len() != n_ch {
            return Err(LmlError::InvalidHeader(format!(
                "window {w}: decoded channel count {} != header n_ch {}",
                window.len(),
                n_ch
            )));
        }

        let start = w * window_size;
        for ch in 0..n_ch {
            let end = (start + window[ch].len()).min(total_samples);
            let copy_len = end - start;
            signal[ch][start..start + copy_len].copy_from_slice(&window[ch][..copy_len]);
        }
    }

    Ok((signal, metadata))
}

/// Write-side result summary — only the write functions construct it.
///
/// ADR 0069 L8: moved from `legacy-encode` to `legacy-decode` (always
/// available). The type itself carries no encode logic — it's a plain data
/// summary — but the LIVE `write_legacy_recording` facade shims
/// (`lamquant-lossless::legacy_bcs1_container::{write_into,write_file,...}`) need
/// to construct instances of the SAME type the retiring `legacy-encode`
/// writer returns, so downstream call sites (`ContainerStats::compressed_size`
/// etc.) keep resolving to one type regardless of which writer produced it.
pub struct ContainerStats {
    pub n_windows: usize,
    pub n_channels: usize,
    pub total_samples: usize,
    pub compressed_size: usize,
    pub raw_size: usize,
    pub cr: f64,
    pub duration_s: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "legacy-encode")]
    fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut sig = vec![Vec::with_capacity(t); n_ch];
        for channel in sig.iter_mut().take(n_ch) {
            for _ in 0..t {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                channel.push(((state >> 33) as i32) as i64 % 8000);
            }
        }
        sig
    }

    #[test]
    fn magic_and_version_constants_pinned() {
        assert_eq!(MAGIC, b"LML1");
        assert_eq!(VERSION_MAJOR, 1);
        assert_eq!(VERSION_MINOR, 0);
    }

    /// #28 — synthetic known-answer test for the HISTORICAL 18-byte and 20-byte
    /// header layouts `parse_header` auto-detects. The 32-byte header is exercised
    /// throughout this module; the older 18-byte (pre-version-field, `probe` IS the
    /// channel count) and 20-byte (versioned, no bit-depth byte) layouts were never
    /// directly pinned. Hand-crafted headers → known fields, so a regression in the
    /// auto-detect branch (bit_depth discriminator / offsets) is caught.
    #[test]
    fn parse_header_kat_18_and_20_byte_layouts() {
        // 18-byte: `probe != 1`, so `probe` IS n_channels (3); no version/bit-depth.
        let mut h18 = Vec::new();
        h18.extend_from_slice(MAGIC); // "LML1"
        h18.extend_from_slice(&3u16.to_le_bytes()); // probe = n_channels = 3
        h18.extend_from_slice(&2u16.to_le_bytes()); // n_windows = 2
        h18.extend_from_slice(&100u32.to_le_bytes()); // total_samples = 100
        h18.extend_from_slice(&50u16.to_le_bytes()); // window_size = 50
        h18.extend_from_slice(&2u32.to_le_bytes()); // meta_len = 2  (hdr_end = 18)
        h18.extend_from_slice(b"{}"); // metadata
        h18.extend_from_slice(&[0u8; 8]); // window index (2 × u32 LE)
        let h = parse_header(&h18).expect("18-byte header must parse");
        assert_eq!(
            (h.n_ch, h.n_windows, h.total_samples, h.window_size),
            (3, 2, 100, 50),
            "18-byte header fields"
        );
        assert_eq!(h.metadata, "{}");

        // 20-byte: `probe == 1` (version marker) and byte[20] ∉ {16,24,32} (not a
        // bit_depth), so the 20-byte branch is taken EVEN THOUGH the file is ≥ 32
        // bytes — exercising the bit_depth discriminator, not just a length shortcut.
        let mut h20 = Vec::new();
        h20.extend_from_slice(MAGIC); // "LML1"
        h20.extend_from_slice(&1u16.to_le_bytes()); // probe = 1 (version)
        h20.extend_from_slice(&2u16.to_le_bytes()); // n_channels = 2
        h20.extend_from_slice(&4u16.to_le_bytes()); // n_windows = 4
        h20.extend_from_slice(&100u32.to_le_bytes()); // total_samples = 100
        h20.extend_from_slice(&50u16.to_le_bytes()); // window_size = 50
        h20.extend_from_slice(&2u32.to_le_bytes()); // meta_len = 2  (hdr_end = 20)
        h20.extend_from_slice(b"{}"); // metadata; byte[20] = '{' (123) → 20-byte branch
        h20.extend_from_slice(&[0u8; 16]); // window index (4 × u32) → total ≥ 32
        assert!(
            h20.len() >= 32,
            "must be ≥ 32 to prove the discriminator, not length"
        );
        let h = parse_header(&h20).expect("20-byte header must parse");
        assert_eq!(
            (h.n_ch, h.n_windows, h.total_samples, h.window_size),
            (2, 4, 100, 50),
            "20-byte header fields"
        );
        assert_eq!(h.metadata, "{}");
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_writes_footer_with_flag_bit_set() {
        // Phase 0.6: every new file carries the LMLFOOT1 footer and
        // has FLAG_HAS_FOOTER set in header byte 21.
        let sig = synth_signal(2, 256, 1);
        let mut sink: Vec<u8> = Vec::new();
        write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        // Byte 21 = flags
        assert_eq!(
            sink[21] & FLAG_HAS_FOOTER,
            FLAG_HAS_FOOTER,
            "FLAG_HAS_FOOTER must be set"
        );
        // Last 32 bytes = footer; magic must match.
        let footer_start = sink.len() - 32;
        assert_eq!(&sink[footer_start..footer_start + 8], b"LMLFOOT1");
        // Round-trip through OffsetTable parser.
        let table = OffsetTable::read_from_buffer(&sink)
            .unwrap()
            .expect("footer must be detected");
        assert_eq!(table.len(), 2, "2 windows × 128 samples = 2 offset entries");
        // First entry samples index = 0.
        assert_eq!(table.entries()[0].first_sample_idx, 0);
        // Second entry samples index = 128.
        assert_eq!(table.entries()[1].first_sample_idx, 128);
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn decode_skips_footer_silently() {
        // The new footer is at EOF; decode_buffer must ignore it and
        // still recover the full signal byte-exact. This is the
        // backward-compat path — old readers that don't know the flag
        // bit also ignore the footer.
        let sig = synth_signal(3, 192, 11);
        let mut sink: Vec<u8> = Vec::new();
        write_into(
            &mut sink,
            &sig,
            250.0,
            96,
            0,
            "{\"k\":1}",
            LpcMode::default(),
        )
        .unwrap();
        let (recovered, meta) = read_from(&mut std::io::Cursor::new(&sink)).unwrap();
        // Metadata is augmented with the self-describing codec_mode stamp
        // (deployment tier), but the caller's fields are preserved.
        assert!(
            meta.contains("\"k\":1") && meta.contains("\"codec_mode\""),
            "metadata must preserve caller fields + carry codec_mode: {meta}"
        );
        for ch in 0..3 {
            assert_eq!(recovered[ch], sig[ch], "channel {ch} drift");
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn footer_offsets_point_at_real_window_payloads() {
        // The OffsetTable's abs_offset values must point at each
        // window's u32 length-prefix bytes in the file. Verify by
        // reading the length at each offset and confirming it matches
        // the entry's payload_len - 4.
        let sig = synth_signal(2, 384, 17);
        let mut sink: Vec<u8> = Vec::new();
        write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        let table = OffsetTable::read_from_buffer(&sink).unwrap().unwrap();
        for e in table.entries() {
            let off = e.abs_offset as usize;
            assert!(off + 4 <= sink.len(), "abs_offset past EOF");
            let prefix =
                u32::from_le_bytes([sink[off], sink[off + 1], sink[off + 2], sink[off + 3]]);
            assert_eq!(
                prefix + 4,
                e.payload_len,
                "length-prefix mismatch at abs_offset={off}"
            );
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_bytes_rejects_oversized_total_samples() {
        // A header claiming far more total_samples than n_windows*window_size
        // must be rejected BEFORE the signal matrix is allocated — otherwise
        // an attacker-set u32 drives a multi-TB vec on a tiny file.
        let sig = synth_signal(2, 256, 9);
        let mut sink: Vec<u8> = Vec::new();
        write_into(&mut sink, &sig, 250.0, 128, 0, "{}", LpcMode::default()).unwrap();
        assert!(
            read_bytes(&sink).is_ok(),
            "unmutated container should decode"
        );
        // total_samples is the u32 at header bytes [10..14] (32-byte header).
        sink[10..14].copy_from_slice(&u32::MAX.to_le_bytes());
        let r = read_bytes(&sink);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "{r:?}");
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_into_vec_then_read_from_cursor_roundtrip() {
        // Phase 0.5: write_into → Vec<u8> sink, read_from → Cursor.
        // Verifies the generic-sink path produces byte-identical output
        // to the file path (covered by `write_then_read_roundtrip_byte_exact`).
        let sig = synth_signal(3, 256, 1234);
        let mut sink: Vec<u8> = Vec::new();
        let stats = write_into(
            &mut sink,
            &sig,
            250.0,
            128,
            0,
            "{\"src\":\"test\"}",
            LpcMode::default(),
        )
        .unwrap();
        assert_eq!(stats.n_channels, 3);
        assert_eq!(stats.total_samples, 256);
        assert_eq!(
            stats.compressed_size,
            sink.len(),
            "CountingWriter byte total must match sink length",
        );
        let mut cursor = std::io::Cursor::new(&sink);
        let (recovered, meta) = read_from(&mut cursor).unwrap();
        assert!(
            meta.contains("\"src\":\"test\"") && meta.contains("\"codec_mode\""),
            "metadata must preserve caller fields + carry codec_mode: {meta}"
        );
        for ch in 0..3 {
            assert_eq!(recovered[ch], sig[ch], "channel {ch} drift");
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_into_byte_identical_to_write_file() {
        // Same input → same output via both paths.
        let sig = synth_signal(2, 300, 7);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_file(tmp.path(), &sig, 250.0, 250, 0, "{}").unwrap();
        let file_bytes = std::fs::read(tmp.path()).unwrap();
        let mut sink_bytes: Vec<u8> = Vec::new();
        write_into(
            &mut sink_bytes,
            &sig,
            250.0,
            250,
            0,
            "{}",
            LpcMode::default(),
        )
        .unwrap();
        assert_eq!(
            file_bytes, sink_bytes,
            "write_into and write_file must produce byte-identical output"
        );
    }

    /// A `Read` that yields one byte per call — locally defined (the shared helper
    /// lived in `lamquant-lossless::io`, which stays in that crate).
    #[cfg(feature = "legacy-encode")]
    struct ByteAtATime<'a> {
        src: &'a [u8],
    }
    #[cfg(feature = "legacy-encode")]
    impl<'a> ByteAtATime<'a> {
        fn new(src: &'a [u8]) -> Self {
            Self { src }
        }
    }
    #[cfg(feature = "legacy-encode")]
    impl std::io::Read for ByteAtATime<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.src.is_empty() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.src[0];
            self.src = &self.src[1..];
            Ok(1)
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_from_handles_partial_reads() {
        // Force one-byte-per-read; read_to_end inside read_from must
        // still pull the whole container.
        let sig = synth_signal(1, 128, 99);
        let mut sink: Vec<u8> = Vec::new();
        write_into(&mut sink, &sig, 250.0, 64, 0, "{}", LpcMode::default()).unwrap();
        let mut src = ByteAtATime::new(&sink);
        let (recovered, _) = read_from(&mut src).unwrap();
        assert_eq!(recovered[0], sig[0]);
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_then_read_roundtrip_byte_exact() {
        let sig = synth_signal(4, 512, 42);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let stats = write_file(tmp.path(), &sig, 250.0, 256, 0, "{}").unwrap();
        assert_eq!(stats.n_channels, 4);
        assert_eq!(stats.total_samples, 512);
        assert_eq!(stats.n_windows, 2);

        let (recovered, meta) = read_file(tmp.path()).unwrap();
        // "{}" in → augmented with the self-describing lossless codec_mode stamp.
        assert!(
            meta.contains("\"codec_mode\":\"lossless\""),
            "metadata must carry the codec_mode stamp: {meta}"
        );
        assert_eq!(recovered.len(), 4);
        for ch in 0..4 {
            assert_eq!(recovered[ch], sig[ch], "channel {} drift", ch);
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_rejects_zero_channels() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let empty: Vec<Vec<i64>> = vec![];
        match write_file(tmp.path(), &empty, 250.0, 256, 0, "{}") {
            Err(LmlError::InvalidHeader(_)) => {}
            Err(e) => panic!("expected InvalidHeader, got different error: {}", e),
            Ok(_) => panic!("expected InvalidHeader error, got Ok"),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_rejects_zero_samples() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let empty = vec![vec![], vec![]];
        match write_file(tmp.path(), &empty, 250.0, 256, 0, "{}") {
            Err(LmlError::InvalidHeader(_)) => {}
            Err(e) => panic!("expected InvalidHeader, got different error: {}", e),
            Ok(_) => panic!("expected InvalidHeader error, got Ok"),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn write_rejects_zero_window_size() {
        // sample_rate=250 with window_size=0 → actual_window=0 → reject.
        let sig = synth_signal(2, 64, 1);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        match write_file(tmp.path(), &sig, 250.0, 0, 0, "{}") {
            Err(LmlError::InvalidHeader(msg)) => {
                assert!(msg.contains("window"), "got {}", msg)
            }
            Err(e) => panic!("expected InvalidHeader, got different error: {}", e),
            Ok(_) => panic!("expected InvalidHeader error, got Ok"),
        }
    }

    #[test]
    fn read_rejects_invalid_magic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"XXXX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        match read_file(tmp.path()) {
            Err(LmlError::InvalidMagic(_)) => {}
            Err(e) => panic!("expected InvalidMagic, got: {}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn read_rejects_future_version() {
        // LML9 future-version magic — first 3 bytes "LML", 4th byte ASCII digit != '1'.
        let mut buf = vec![0u8; 32];
        buf[0..4].copy_from_slice(b"LML9");
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();
        match read_file(tmp.path()) {
            Err(LmlError::UnsupportedVersion(b'9')) => {}
            Err(e) => panic!("expected UnsupportedVersion(b'9'), got: {}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn read_rejects_truncated_header() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"LML").unwrap();
        match read_file(tmp.path()) {
            Err(LmlError::Truncated { .. }) => {}
            Err(e) => panic!("expected Truncated, got: {}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn read_rejects_truncated_metadata() {
        // Build a valid 32-byte header that claims meta_len=1000, then write
        // only 32 bytes — read must report truncation, not panic.
        let mut hdr = Vec::with_capacity(32);
        hdr.extend_from_slice(MAGIC);
        hdr.push(VERSION_MAJOR);
        hdr.push(VERSION_MINOR);
        hdr.extend_from_slice(&2u16.to_le_bytes()); // n_channels
        hdr.extend_from_slice(&1u16.to_le_bytes()); // n_windows
        hdr.extend_from_slice(&64u32.to_le_bytes()); // total_samples
        hdr.extend_from_slice(&64u16.to_le_bytes()); // window_size
        hdr.extend_from_slice(&250_000u32.to_le_bytes()); // sample_rate_mhz
        hdr.push(16); // bit_depth
        hdr.push(0); // flags
        hdr.extend_from_slice(&1000u32.to_le_bytes()); // meta_len = 1000 (will trip)
        hdr.extend_from_slice(&[0u8; 6]);
        assert_eq!(hdr.len(), 32);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &hdr).unwrap();
        match read_file(tmp.path()) {
            Err(LmlError::Truncated { context, .. }) => {
                assert_eq!(context, "metadata");
            }
            Err(e) => panic!("expected Truncated(metadata), got: {}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn read_rejects_zero_channel_count() {
        // Hand-craft a 32-byte header with n_channels=0.
        let mut hdr = Vec::with_capacity(32);
        hdr.extend_from_slice(MAGIC);
        hdr.push(VERSION_MAJOR);
        hdr.push(VERSION_MINOR);
        hdr.extend_from_slice(&0u16.to_le_bytes()); // n_channels=0 (forbidden)
        hdr.extend_from_slice(&1u16.to_le_bytes());
        hdr.extend_from_slice(&64u32.to_le_bytes());
        hdr.extend_from_slice(&64u16.to_le_bytes());
        hdr.extend_from_slice(&250_000u32.to_le_bytes());
        hdr.push(16);
        hdr.push(0);
        hdr.extend_from_slice(&0u32.to_le_bytes()); // meta_len=0
        hdr.extend_from_slice(&[0u8; 6]);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &hdr).unwrap();
        match read_file(tmp.path()) {
            Err(LmlError::InvalidHeader(msg)) => {
                assert!(msg.contains("channel"), "got {}", msg)
            }
            Err(e) => panic!("expected InvalidHeader, got different error: {}", e),
            Ok(_) => panic!("expected InvalidHeader error, got Ok"),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn metadata_string_preserved() {
        let sig = synth_signal(2, 256, 5);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let meta = r#"{"patient":"X","sample_rate":250,"channels":["FP1","FP2"]}"#;
        write_file(tmp.path(), &sig, 250.0, 128, 0, meta).unwrap();
        let (_, recovered_meta) = read_file(tmp.path()).unwrap();
        // Caller metadata fields are preserved verbatim; the container also
        // stamps the self-describing codec_mode (deployment tier) alongside.
        assert!(
            recovered_meta.contains(r#""patient":"X""#)
                && recovered_meta.contains(r#""channels":["FP1","FP2"]"#)
                && recovered_meta.contains("\"codec_mode\""),
            "metadata must preserve caller fields + carry codec_mode: {recovered_meta}"
        );
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_window_from_bytes_matches_full_decode() {
        // Write an 8-window container (256 samples / 32-window = 8 windows).
        let sig = synth_signal(4, 256, 7);
        let mut buf: Vec<u8> = Vec::new();
        write_into(&mut buf, &sig, 250.0, 32, 0, "{}", LpcMode::default()).unwrap();

        // Full-LMA decode for ground truth.
        let (full, _meta) = read_bytes(&buf).unwrap();
        assert_eq!(full.len(), 4);
        assert_eq!(full[0].len(), 256);

        // Random-access decode every window, compare slice-by-slice.
        for w in 0..8 {
            let (win, header) = read_window_from_bytes(&buf, w).unwrap();
            assert_eq!(header.n_windows, 8);
            assert_eq!(win.len(), 4);
            let start = w * 32;
            for ch in 0..4 {
                for t in 0..win[ch].len() {
                    assert_eq!(
                        win[ch][t],
                        full[ch][start + t],
                        "window {} ch {} t {} mismatch",
                        w,
                        ch,
                        t
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_bytes_into_f32_calibrated_matches_python_formula() {
        // Synthesize 3-channel 128-sample int signal, write container,
        // then decode via the streaming f32 path. Per-channel calibration
        // is identity-like ((x - dig_min) * scale + phys_min); compute
        // the same in Rust and assert equality.
        let sig = synth_signal(3, 128, 9);
        let mut buf: Vec<u8> = Vec::new();
        write_into(&mut buf, &sig, 250.0, 64, 0, "{}", LpcMode::default()).unwrap();

        // Calib: dig=[-32768, 32767], phys=[-200.0, 200.0] for every ch.
        let calib: Vec<f32> = (0..3)
            .flat_map(|_| [-32768.0_f32, 32767.0, -200.0, 200.0])
            .collect();
        let mut out = vec![0f32; 3 * 128];
        let header = read_bytes_into_f32_calibrated(&buf, &mut out, &calib).unwrap();
        assert_eq!(header.n_ch, 3);
        assert_eq!(header.total_samples, 128);

        let (full, _) = read_bytes(&buf).unwrap();
        for ch in 0..3 {
            for t in 0..128 {
                let dig = full[ch][t] as f32;
                let scale = (200.0_f32 - -200.0_f32) / (32767.0_f32 - -32768.0_f32);
                let expected = (dig - -32768.0_f32) * scale + -200.0_f32;
                let got = out[ch * 128 + t];
                assert!(
                    (expected - got).abs() < 1e-3,
                    "ch {ch} t {t}: expected {expected} got {got}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_bytes_into_f32_calibrated_handles_degenerate_calibration() {
        let sig = synth_signal(2, 64, 10);
        let mut buf: Vec<u8> = Vec::new();
        write_into(&mut buf, &sig, 250.0, 32, 0, "{}", LpcMode::default()).unwrap();
        // Channel 0 has dig_max == dig_min → zero row. Channel 1 normal.
        let calib: Vec<f32> = vec![
            0.0, 0.0, -100.0, 100.0, // ch 0 degenerate
            -32768.0, 32767.0, -200.0, 200.0, // ch 1 normal
        ];
        let mut out = vec![0f32; 2 * 64];
        read_bytes_into_f32_calibrated(&buf, &mut out, &calib).unwrap();
        for (t, value) in out.iter().take(64).enumerate() {
            assert_eq!(*value, 0.0, "ch 0 t {t} should be zero (degenerate)");
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn read_window_from_bytes_rejects_oob_idx() {
        let sig = synth_signal(2, 128, 8);
        let mut buf: Vec<u8> = Vec::new();
        write_into(&mut buf, &sig, 250.0, 64, 0, "{}", LpcMode::default()).unwrap();
        // 128 samples / 64 window = 2 windows. idx 2 OOB.
        match read_window_from_bytes(&buf, 2) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("out of range")),
            other => panic!("expected InvalidHeader, got {:?}", other.map(|_| "ok")),
        }
    }

    #[test]
    #[cfg(feature = "legacy-encode")]
    fn noise_bits_lossy_path_recovers_shifted() {
        let sig = synth_signal(2, 256, 6);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let nb: u8 = 4;
        write_file(tmp.path(), &sig, 250.0, 128, nb, "{}").unwrap();
        let (recovered, _) = read_file(tmp.path()).unwrap();
        for ch in 0..2 {
            for i in 0..256 {
                let expected = (sig[ch][i] >> nb) << nb;
                assert_eq!(recovered[ch][i], expected, "ch {} idx {}", ch, i);
            }
        }
    }
}
