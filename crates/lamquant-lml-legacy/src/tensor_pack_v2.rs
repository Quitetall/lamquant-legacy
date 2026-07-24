// SPDX-License-Identifier: AGPL-3.0-or-later
//! PORTED VERBATIM from the isolated training data-plane work (ADR 0144:
//! "port validated pieces from isolated worktrees/commits"). This wire is
//! RETIRED: nothing writes it any more, and future evolution is
//! `bcs.training.lqtp.v1+` under BCS2 grammar, never a standalone LQTP4.
//! It lives here so old snapshots keep decoding, and so the ADR 0143 adapter
//! has a real reader instead of a magic it cannot open.
//!
//! Only the LQTP1 import path is re-pointed: `crate::tensor_pack` in the codec
//! is `lamquant_lml_archive::tensor_pack` here. The wire itself is untouched.
//! LQTP2: immutable, content-bound, multi-view training snapshots.
//!
//! LQTP1 remains frozen. LQTP2 adds a sorted view directory, per-view shapes,
//! raw and block-floating-point storage, canonical row identity binding, and
//! strict open-time integrity verification. Every view has the same row count;
//! a trainer selects required views when it opens a snapshot and never falls
//! back to archival decode during an epoch.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use lamquant_lml_archive::tensor_pack::{dequantize_window, quantize_window, PackDtype};

/// LQTP2 file magic.
pub const LQTP2_MAGIC: &[u8; 4] = b"LQT2";
/// LQTP2 wire major version.
pub const LQTP2_VERSION_MAJOR: u8 = 2;
/// LQTP2 wire minor version.
pub const LQTP2_VERSION_MINOR: u8 = 0;
/// Fixed LQTP2 header length.
pub const LQTP2_HEADER_LEN: usize = 256;
/// Fixed view-directory entry length.
pub const LQTP2_VIEW_ENTRY_LEN: usize = 192;

const ENDIAN_LITTLE: u8 = 1;
const FLAG_SHA256: u8 = 1;
const VIEW_FLAG_REQUIRED: u8 = 1;
const VIEW_ALIGNMENT: u64 = 64;
const MAX_VIEWS: usize = 256;
const MAX_RANK: usize = 4;
const MAX_VIEW_NAME_BYTES: usize = 64;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Logical scalar type of one LQTP2 view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackV2Dtype {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    Bool,
}

impl PackV2Dtype {
    /// Stable wire tag.
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::I8 => 1,
            Self::U8 => 2,
            Self::I16 => 3,
            Self::U16 => 4,
            Self::I32 => 5,
            Self::U32 => 6,
            Self::I64 => 7,
            Self::U64 => 8,
            Self::F16 => 9,
            Self::Bf16 => 10,
            Self::F32 => 11,
            Self::F64 => 12,
            Self::Bool => 13,
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            1 => Self::I8,
            2 => Self::U8,
            3 => Self::I16,
            4 => Self::U16,
            5 => Self::I32,
            6 => Self::U32,
            7 => Self::I64,
            8 => Self::U64,
            9 => Self::F16,
            10 => Self::Bf16,
            11 => Self::F32,
            12 => Self::F64,
            13 => Self::Bool,
            _ => return None,
        })
    }

    /// Stored bytes per scalar for raw views.
    pub const fn width(self) -> u64 {
        match self {
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Physical row encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackV2Encoding {
    /// Native little-endian scalars.
    Raw,
    /// Per-leading-lane f32 scale plus signed 8-bit mantissas.
    BfpInt8,
    /// Per-leading-lane f32 scale plus signed 16-bit mantissas.
    BfpInt16,
}

impl PackV2Encoding {
    /// Stable wire tag.
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::BfpInt8 => 1,
            Self::BfpInt16 => 2,
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Raw,
            1 => Self::BfpInt8,
            2 => Self::BfpInt16,
            _ => return None,
        })
    }
}

/// Canonical description of one materialized view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewSpec {
    name: String,
    dtype: PackV2Dtype,
    encoding: PackV2Encoding,
    rank: usize,
    row_shape: [u64; MAX_RANK],
    required: bool,
    spec_sha256: [u8; 32],
    row_stride: u64,
}

impl ViewSpec {
    /// Build and validate a fixed-shape row view.
    pub fn new(
        name: impl Into<String>,
        dtype: PackV2Dtype,
        encoding: PackV2Encoding,
        row_shape: &[usize],
        required: bool,
        spec_sha256: [u8; 32],
    ) -> Result<Self, PackV2Error> {
        let name = name.into();
        if name.is_empty() || name.len() > MAX_VIEW_NAME_BYTES || name.as_bytes().contains(&0) {
            return Err(PackV2Error::InvalidViewName(name));
        }
        if row_shape.is_empty() || row_shape.len() > MAX_RANK {
            return Err(PackV2Error::ShapeMismatch(format!(
                "view '{name}' rank {} is outside 1..={MAX_RANK}",
                row_shape.len()
            )));
        }
        let mut dimensions = [0_u64; MAX_RANK];
        for (index, value) in row_shape.iter().copied().enumerate() {
            if value == 0 {
                return Err(PackV2Error::ShapeMismatch(format!(
                    "view '{name}' dimension {index} is zero"
                )));
            }
            dimensions[index] = u64::try_from(value).map_err(|_| {
                PackV2Error::ShapeMismatch(format!(
                    "view '{name}' dimension {index} does not fit u64"
                ))
            })?;
        }
        if encoding != PackV2Encoding::Raw && dtype != PackV2Dtype::F32 {
            return Err(PackV2Error::ShapeMismatch(format!(
                "view '{name}' BFP encoding requires logical f32"
            )));
        }
        let element_count = checked_product(&dimensions[..row_shape.len()], "view elements")?;
        let row_stride = match encoding {
            PackV2Encoding::Raw => element_count
                .checked_mul(dtype.width())
                .ok_or(PackV2Error::IntegerOverflow("raw row stride"))?,
            PackV2Encoding::BfpInt8 => dimensions[0]
                .checked_mul(4)
                .and_then(|scales| scales.checked_add(element_count))
                .ok_or(PackV2Error::IntegerOverflow("BFP8 row stride"))?,
            PackV2Encoding::BfpInt16 => dimensions[0]
                .checked_mul(4)
                .and_then(|scales| {
                    element_count
                        .checked_mul(2)
                        .and_then(|m| scales.checked_add(m))
                })
                .ok_or(PackV2Error::IntegerOverflow("BFP16 row stride"))?,
        };
        Ok(Self {
            name,
            dtype,
            encoding,
            rank: row_shape.len(),
            row_shape: dimensions,
            required,
            spec_sha256,
            row_stride,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn dtype(&self) -> PackV2Dtype {
        self.dtype
    }

    pub const fn encoding(&self) -> PackV2Encoding {
        self.encoding
    }

    pub fn row_shape(&self) -> &[u64] {
        &self.row_shape[..self.rank]
    }

    pub const fn required(&self) -> bool {
        self.required
    }

    pub const fn spec_sha256(&self) -> &[u8; 32] {
        &self.spec_sha256
    }

    pub const fn row_stride(&self) -> u64 {
        self.row_stride
    }

    fn element_count(&self) -> Result<u64, PackV2Error> {
        checked_product(self.row_shape(), "view elements")
    }
}

/// Parsed view directory entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewDescriptor {
    spec: ViewSpec,
    data_offset: u64,
    data_length: u64,
    data_sha256: [u8; 32],
}

impl ViewDescriptor {
    pub fn name(&self) -> &str {
        self.spec.name()
    }

    pub const fn dtype(&self) -> PackV2Dtype {
        self.spec.dtype()
    }

    pub const fn encoding(&self) -> PackV2Encoding {
        self.spec.encoding()
    }

    pub fn row_shape(&self) -> &[u64] {
        self.spec.row_shape()
    }

    pub const fn required(&self) -> bool {
        self.spec.required()
    }

    pub const fn spec_sha256(&self) -> &[u8; 32] {
        self.spec.spec_sha256()
    }

    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    pub const fn data_length(&self) -> u64 {
        self.data_length
    }

    pub const fn row_stride(&self) -> u64 {
        self.spec.row_stride()
    }

    pub const fn data_sha256(&self) -> &[u8; 32] {
        &self.data_sha256
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let name = self.spec.name.as_bytes();
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.push(self.spec.dtype.to_u8());
        out.push(self.spec.encoding.to_u8());
        out.push(self.spec.rank as u8);
        out.push(u8::from(self.spec.required) * VIEW_FLAG_REQUIRED);
        out.extend_from_slice(&0_u16.to_le_bytes());
        for dimension in self.spec.row_shape {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&self.data_offset.to_le_bytes());
        out.extend_from_slice(&self.data_length.to_le_bytes());
        out.extend_from_slice(&self.spec.row_stride.to_le_bytes());
        out.extend_from_slice(&self.data_sha256);
        out.extend_from_slice(&self.spec.spec_sha256);
        out.extend_from_slice(name);
        out.resize(out.len() + MAX_VIEW_NAME_BYTES - name.len(), 0);
    }

    fn parse(bytes: &[u8]) -> Result<Self, PackV2Error> {
        if bytes.len() != LQTP2_VIEW_ENTRY_LEN {
            return Err(PackV2Error::InvalidLayout("view entry length"));
        }
        let name_length = read_u16(bytes, 0)? as usize;
        if name_length == 0 || name_length > MAX_VIEW_NAME_BYTES {
            return Err(PackV2Error::InvalidLayout("view name length"));
        }
        let name_storage = &bytes[128..192];
        if name_storage[name_length..].iter().any(|value| *value != 0) {
            return Err(PackV2Error::InvalidLayout("view name padding"));
        }
        let name = std::str::from_utf8(&name_storage[..name_length])
            .map_err(|_| PackV2Error::InvalidUtf8)?
            .to_owned();
        let dtype = PackV2Dtype::from_u8(bytes[2]).ok_or(PackV2Error::BadDtype(bytes[2]))?;
        let encoding =
            PackV2Encoding::from_u8(bytes[3]).ok_or(PackV2Error::BadEncoding(bytes[3]))?;
        let rank = bytes[4] as usize;
        if rank == 0 || rank > MAX_RANK || read_u16(bytes, 6)? != 0 {
            return Err(PackV2Error::InvalidLayout("view rank or reserved bytes"));
        }
        if bytes[5] & !VIEW_FLAG_REQUIRED != 0 {
            return Err(PackV2Error::InvalidLayout("view flags"));
        }
        let mut dimensions = [0_u64; MAX_RANK];
        for (index, value) in dimensions.iter_mut().enumerate() {
            *value = read_u64(bytes, 8 + index * 8)?;
        }
        if dimensions[..rank].contains(&0) || dimensions[rank..].iter().any(|value| *value != 0) {
            return Err(PackV2Error::InvalidLayout("view dimensions"));
        }
        let shape: Vec<usize> = dimensions[..rank]
            .iter()
            .map(|value| checked_usize(*value, "view dimension"))
            .collect::<Result<_, _>>()?;
        let mut spec_hash = [0_u8; 32];
        spec_hash.copy_from_slice(&bytes[96..128]);
        let spec = ViewSpec::new(
            name,
            dtype,
            encoding,
            &shape,
            bytes[5] & VIEW_FLAG_REQUIRED != 0,
            spec_hash,
        )?;
        if spec.row_stride != read_u64(bytes, 56)? {
            return Err(PackV2Error::InvalidLayout("view row stride"));
        }
        let mut data_hash = [0_u8; 32];
        data_hash.copy_from_slice(&bytes[64..96]);
        Ok(Self {
            spec,
            data_offset: read_u64(bytes, 40)?,
            data_length: read_u64(bytes, 48)?,
            data_sha256: data_hash,
        })
    }
}

#[derive(Clone, Debug)]
struct Header {
    view_count: usize,
    row_count: u64,
    directory_offset: u64,
    directory_length: u64,
    metadata_offset: u64,
    metadata_length: u64,
    data_offset: u64,
    file_length: u64,
    manifest_sha256: [u8; 32],
    view_spec_sha256: [u8; 32],
    metadata_sha256: [u8; 32],
    directory_sha256: [u8; 32],
}

impl Header {
    fn encode(&self) -> [u8; LQTP2_HEADER_LEN] {
        let mut bytes = [0_u8; LQTP2_HEADER_LEN];
        bytes[..4].copy_from_slice(LQTP2_MAGIC);
        bytes[4] = LQTP2_VERSION_MAJOR;
        bytes[5] = LQTP2_VERSION_MINOR;
        bytes[6] = ENDIAN_LITTLE;
        bytes[7] = FLAG_SHA256;
        bytes[8..12].copy_from_slice(&(LQTP2_HEADER_LEN as u32).to_le_bytes());
        bytes[12..16].copy_from_slice(&(LQTP2_VIEW_ENTRY_LEN as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(self.view_count as u32).to_le_bytes());
        bytes[24..32].copy_from_slice(&self.row_count.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.directory_offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.directory_length.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.metadata_offset.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.metadata_length.to_le_bytes());
        bytes[64..72].copy_from_slice(&self.data_offset.to_le_bytes());
        bytes[72..80].copy_from_slice(&self.file_length.to_le_bytes());
        bytes[80..112].copy_from_slice(&self.manifest_sha256);
        bytes[112..144].copy_from_slice(&self.view_spec_sha256);
        bytes[144..176].copy_from_slice(&self.metadata_sha256);
        bytes[208..240].copy_from_slice(&self.directory_sha256);
        let header_hash = sha256(&bytes);
        bytes[176..208].copy_from_slice(&header_hash);
        bytes
    }

    fn parse(bytes: &[u8]) -> Result<Self, PackV2Error> {
        if bytes.len() < LQTP2_HEADER_LEN {
            return Err(PackV2Error::Truncated {
                expected: LQTP2_HEADER_LEN,
                actual: bytes.len(),
            });
        }
        if &bytes[..4] != LQTP2_MAGIC {
            return Err(PackV2Error::BadMagic);
        }
        if bytes[4] != LQTP2_VERSION_MAJOR || bytes[5] != LQTP2_VERSION_MINOR {
            return Err(PackV2Error::BadVersion(bytes[4], bytes[5]));
        }
        if bytes[6] != ENDIAN_LITTLE {
            return Err(PackV2Error::BadEndianness(bytes[6]));
        }
        if bytes[7] != FLAG_SHA256 {
            return Err(PackV2Error::BadFlags(bytes[7]));
        }
        if read_u32(bytes, 8)? as usize != LQTP2_HEADER_LEN
            || read_u32(bytes, 12)? as usize != LQTP2_VIEW_ENTRY_LEN
            || read_u32(bytes, 20)? != 0
            || bytes[240..256].iter().any(|value| *value != 0)
        {
            return Err(PackV2Error::InvalidLayout("header fields"));
        }
        let mut stored_header_hash = [0_u8; 32];
        stored_header_hash.copy_from_slice(&bytes[176..208]);
        let mut canonical = [0_u8; LQTP2_HEADER_LEN];
        canonical.copy_from_slice(&bytes[..LQTP2_HEADER_LEN]);
        canonical[176..208].fill(0);
        if sha256(&canonical) != stored_header_hash {
            return Err(PackV2Error::IntegrityMismatch("header"));
        }
        let view_count = read_u32(bytes, 16)? as usize;
        if view_count == 0 || view_count > MAX_VIEWS {
            return Err(PackV2Error::InvalidLayout("view count"));
        }
        let row_count = read_u64(bytes, 24)?;
        if row_count == 0 {
            return Err(PackV2Error::InvalidLayout("row count"));
        }
        let mut manifest_sha256 = [0_u8; 32];
        manifest_sha256.copy_from_slice(&bytes[80..112]);
        let mut view_spec_sha256 = [0_u8; 32];
        view_spec_sha256.copy_from_slice(&bytes[112..144]);
        let mut metadata_sha256 = [0_u8; 32];
        metadata_sha256.copy_from_slice(&bytes[144..176]);
        let mut directory_sha256 = [0_u8; 32];
        directory_sha256.copy_from_slice(&bytes[208..240]);
        Ok(Self {
            view_count,
            row_count,
            directory_offset: read_u64(bytes, 32)?,
            directory_length: read_u64(bytes, 40)?,
            metadata_offset: read_u64(bytes, 48)?,
            metadata_length: read_u64(bytes, 56)?,
            data_offset: read_u64(bytes, 64)?,
            file_length: read_u64(bytes, 72)?,
            manifest_sha256,
            view_spec_sha256,
            metadata_sha256,
            directory_sha256,
        })
    }
}

struct ViewSink {
    spec: ViewSpec,
    file: Option<BufWriter<File>>,
    temp_path: PathBuf,
    rows_written: u64,
    hasher: Sha256,
}

/// Atomic streaming LQTP2 writer.
pub struct PackV2Writer {
    final_path: PathBuf,
    partial_path: PathBuf,
    row_count: u64,
    manifest_sha256: [u8; 32],
    view_spec_sha256: [u8; 32],
    metadata: Vec<u8>,
    views: Vec<ViewSink>,
    done: bool,
}

impl PackV2Writer {
    /// Create a writer. Views are sorted by name before any bytes are emitted.
    pub fn create(
        path: &Path,
        row_count: usize,
        manifest_sha256: [u8; 32],
        view_spec_sha256: [u8; 32],
        metadata: Vec<u8>,
        mut specs: Vec<ViewSpec>,
    ) -> Result<Self, PackV2Error> {
        if row_count == 0 {
            return Err(PackV2Error::ShapeMismatch("row count is zero".into()));
        }
        if specs.is_empty() || specs.len() > MAX_VIEWS {
            return Err(PackV2Error::ShapeMismatch(format!(
                "view count {} is outside 1..={MAX_VIEWS}",
                specs.len()
            )));
        }
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = specs.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(PackV2Error::DuplicateView(pair[0].name.clone()));
        }
        let row_count = u64::try_from(row_count)
            .map_err(|_| PackV2Error::ShapeMismatch("row count does not fit u64".into()))?;
        let token = format!(
            "{}.{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        );
        let partial_path = sibling_with_suffix(path, &format!(".partial.{token}"));
        let mut views = Vec::with_capacity(specs.len());
        for (index, spec) in specs.into_iter().enumerate() {
            let temp_path = sibling_with_suffix(path, &format!(".view.{token}.{index}"));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => views.push(ViewSink {
                    spec,
                    file: Some(BufWriter::new(file)),
                    temp_path,
                    rows_written: 0,
                    hasher: Sha256::new(),
                }),
                Err(error) => {
                    for view in views {
                        let _ = std::fs::remove_file(view.temp_path);
                    }
                    return Err(PackV2Error::Io(error));
                }
            }
        }
        Ok(Self {
            final_path: path.to_path_buf(),
            partial_path,
            row_count,
            manifest_sha256,
            view_spec_sha256,
            metadata,
            views,
            done: false,
        })
    }

    /// Append one already encoded raw row.
    pub fn write_raw_row(&mut self, view_name: &str, bytes: &[u8]) -> Result<(), PackV2Error> {
        let row_count = self.row_count;
        let sink = self.sink_mut(view_name)?;
        if sink.spec.encoding != PackV2Encoding::Raw {
            return Err(PackV2Error::WrongEncoding(view_name.into()));
        }
        if sink.spec.dtype == PackV2Dtype::Bool && bytes.iter().any(|value| *value > 1) {
            return Err(PackV2Error::ShapeMismatch(format!(
                "view '{view_name}' boolean row contains a value other than 0 or 1"
            )));
        }
        write_sink_row(sink, row_count, bytes)
    }

    /// Append one logical f32 row to raw-f32 or BFP storage.
    pub fn write_f32_row(&mut self, view_name: &str, values: &[f32]) -> Result<(), PackV2Error> {
        let row_count = self.row_count;
        let sink = self.sink_mut(view_name)?;
        let expected = checked_usize(sink.spec.element_count()?, "view elements")?;
        if values.len() != expected {
            return Err(PackV2Error::ShapeMismatch(format!(
                "view '{view_name}' row has {} values, expected {expected}",
                values.len()
            )));
        }
        let encoded = match sink.spec.encoding {
            PackV2Encoding::Raw if sink.spec.dtype == PackV2Dtype::F32 => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            PackV2Encoding::Raw => return Err(PackV2Error::WrongDtype(view_name.into())),
            PackV2Encoding::BfpInt8 | PackV2Encoding::BfpInt16 => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(PackV2Error::ShapeMismatch(format!(
                        "view '{view_name}' BFP row contains a non-finite value"
                    )));
                }
                let lanes = checked_usize(sink.spec.row_shape[0], "BFP lanes")?;
                let lane_length = values.len() / lanes;
                let dtype = if sink.spec.encoding == PackV2Encoding::BfpInt8 {
                    PackDtype::Int8
                } else {
                    PackDtype::Int16
                };
                let (scales, mantissas) = quantize_window(values, lanes, lane_length, dtype);
                let mut bytes =
                    Vec::with_capacity(checked_usize(sink.spec.row_stride, "row stride")?);
                for scale in scales {
                    bytes.extend_from_slice(&scale.to_le_bytes());
                }
                bytes.extend_from_slice(&mantissas);
                bytes
            }
        };
        write_sink_row(sink, row_count, &encoded)
    }

    /// Finalize hashes, assemble one canonical file, fsync, then atomically rename.
    pub fn finish(mut self) -> Result<(), PackV2Error> {
        for sink in &self.views {
            if sink.rows_written != self.row_count {
                return Err(PackV2Error::ShapeMismatch(format!(
                    "view '{}' wrote {} of {} rows",
                    sink.spec.name, sink.rows_written, self.row_count
                )));
            }
        }
        let mut data_hashes = Vec::with_capacity(self.views.len());
        for sink in &mut self.views {
            let mut file = sink
                .file
                .take()
                .ok_or_else(|| PackV2Error::ShapeMismatch("writer already finished".into()))?;
            file.flush()?;
            let file = file
                .into_inner()
                .map_err(|error| PackV2Error::Io(error.into_error()))?;
            file.sync_all()?;
            drop(file);
            data_hashes.push(finalize_hash(sink.hasher.clone().finalize()));
        }

        let directory_length = (self.views.len() as u64)
            .checked_mul(LQTP2_VIEW_ENTRY_LEN as u64)
            .ok_or(PackV2Error::IntegerOverflow("directory length"))?;
        let metadata_offset = (LQTP2_HEADER_LEN as u64)
            .checked_add(directory_length)
            .ok_or(PackV2Error::IntegerOverflow("metadata offset"))?;
        let metadata_length = u64::try_from(self.metadata.len())
            .map_err(|_| PackV2Error::IntegerOverflow("metadata length"))?;
        let data_offset = align_up(
            metadata_offset
                .checked_add(metadata_length)
                .ok_or(PackV2Error::IntegerOverflow("data offset"))?,
            VIEW_ALIGNMENT,
        )?;

        let mut descriptors = Vec::with_capacity(self.views.len());
        let mut next_offset = data_offset;
        for (sink, data_sha256) in self.views.iter().zip(data_hashes) {
            let data_offset = align_up(next_offset, VIEW_ALIGNMENT)?;
            let data_length = self
                .row_count
                .checked_mul(sink.spec.row_stride)
                .ok_or(PackV2Error::IntegerOverflow("view data length"))?;
            next_offset = data_offset
                .checked_add(data_length)
                .ok_or(PackV2Error::IntegerOverflow("view data end"))?;
            descriptors.push(ViewDescriptor {
                spec: sink.spec.clone(),
                data_offset,
                data_length,
                data_sha256,
            });
        }
        let file_length = next_offset;
        let mut directory =
            Vec::with_capacity(checked_usize(directory_length, "directory length")?);
        for descriptor in &descriptors {
            descriptor.encode(&mut directory);
        }
        let header = Header {
            view_count: descriptors.len(),
            row_count: self.row_count,
            directory_offset: LQTP2_HEADER_LEN as u64,
            directory_length,
            metadata_offset,
            metadata_length,
            data_offset,
            file_length,
            manifest_sha256: self.manifest_sha256,
            view_spec_sha256: self.view_spec_sha256,
            metadata_sha256: sha256(&self.metadata),
            directory_sha256: sha256(&directory),
        }
        .encode();

        let mut output = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.partial_path)?,
        );
        output.write_all(&header)?;
        output.write_all(&directory)?;
        output.write_all(&self.metadata)?;
        write_padding(&mut output, data_offset - metadata_offset - metadata_length)?;
        let mut position = data_offset;
        for (sink, descriptor) in self.views.iter().zip(&descriptors) {
            write_padding(&mut output, descriptor.data_offset - position)?;
            let mut source = File::open(&sink.temp_path)?;
            let copied = std::io::copy(&mut source, &mut output)?;
            if copied != descriptor.data_length {
                return Err(PackV2Error::Truncated {
                    expected: checked_usize(descriptor.data_length, "view data length")?,
                    actual: checked_usize(copied, "copied data length")?,
                });
            }
            position = descriptor
                .data_offset
                .checked_add(descriptor.data_length)
                .ok_or(PackV2Error::IntegerOverflow("view data end"))?;
        }
        output.flush()?;
        let output = output
            .into_inner()
            .map_err(|error| PackV2Error::Io(error.into_error()))?;
        output.sync_all()?;
        drop(output);
        std::fs::rename(&self.partial_path, &self.final_path)?;
        for sink in &self.views {
            let _ = std::fs::remove_file(&sink.temp_path);
        }
        self.done = true;
        Ok(())
    }

    fn sink_mut(&mut self, name: &str) -> Result<&mut ViewSink, PackV2Error> {
        self.views
            .binary_search_by(|sink| sink.spec.name.as_str().cmp(name))
            .ok()
            .map(|index| &mut self.views[index])
            .ok_or_else(|| PackV2Error::MissingView(name.into()))
    }
}

impl Drop for PackV2Writer {
    fn drop(&mut self) {
        if !self.done {
            let _ = std::fs::remove_file(&self.partial_path);
            for sink in &self.views {
                let _ = std::fs::remove_file(&sink.temp_path);
            }
        }
    }
}

/// Strict read-only mmap of an LQTP2 snapshot.
pub struct PackV2Reader {
    mmap: memmap2::Mmap,
    header: Header,
    views: Vec<ViewDescriptor>,
    metadata_range: Range<usize>,
}

impl PackV2Reader {
    /// Open and fully verify header, directory, metadata, and every view hash.
    pub fn open(
        path: &Path,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
    ) -> Result<Self, PackV2Error> {
        Self::from_file(
            open_nofollow(path)?,
            expected_manifest_sha256,
            expected_view_spec_sha256,
        )
    }

    /// Fully verify a snapshot through an already-open, owned descriptor.
    ///
    /// The caller transfers ownership of `file`. Path-based callers should use
    /// [`Self::open`]; descriptor bridges can duplicate a borrowed descriptor
    /// and pass the duplicate here without reopening a pathname.
    pub fn from_file(
        file: File,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
    ) -> Result<Self, PackV2Error> {
        if !file.metadata()?.is_file() {
            return Err(PackV2Error::InvalidLayout("non-regular file"));
        }
        // SAFETY: read-only mapping. Callers treat a published pack as immutable;
        // all writer publication is atomic rename.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let header = Header::parse(&mmap)?;
        if header.file_length != mmap.len() as u64 {
            return Err(PackV2Error::InvalidLayout("file length"));
        }
        if expected_manifest_sha256.is_some_and(|expected| expected != header.manifest_sha256) {
            return Err(PackV2Error::ManifestMismatch);
        }
        if expected_view_spec_sha256.is_some_and(|expected| expected != header.view_spec_sha256) {
            return Err(PackV2Error::ViewSpecMismatch);
        }
        let expected_directory_length = (header.view_count as u64)
            .checked_mul(LQTP2_VIEW_ENTRY_LEN as u64)
            .ok_or(PackV2Error::IntegerOverflow("directory length"))?;
        if header.directory_offset != LQTP2_HEADER_LEN as u64
            || header.directory_length != expected_directory_length
            || header.metadata_offset
                != header
                    .directory_offset
                    .checked_add(header.directory_length)
                    .ok_or(PackV2Error::IntegerOverflow("metadata offset"))?
        {
            return Err(PackV2Error::InvalidLayout("directory or metadata offset"));
        }
        let directory = checked_slice(
            &mmap,
            header.directory_offset,
            header.directory_length,
            "directory",
        )?;
        if sha256(directory) != header.directory_sha256 {
            return Err(PackV2Error::IntegrityMismatch("directory"));
        }
        let metadata = checked_slice(
            &mmap,
            header.metadata_offset,
            header.metadata_length,
            "metadata",
        )?;
        if sha256(metadata) != header.metadata_sha256 {
            return Err(PackV2Error::IntegrityMismatch("metadata"));
        }
        let metadata_start = checked_usize(header.metadata_offset, "metadata offset")?;
        let metadata_end_usize = metadata_start
            .checked_add(checked_usize(header.metadata_length, "metadata length")?)
            .ok_or(PackV2Error::IntegerOverflow("metadata end"))?;
        checked_usize(header.row_count, "row count")?;
        let metadata_end = header
            .metadata_offset
            .checked_add(header.metadata_length)
            .ok_or(PackV2Error::IntegerOverflow("metadata end"))?;
        if header.data_offset != align_up(metadata_end, VIEW_ALIGNMENT)? {
            return Err(PackV2Error::InvalidLayout("data offset"));
        }
        require_zero_padding(&mmap, metadata_end, header.data_offset, "metadata padding")?;

        let mut views = Vec::with_capacity(header.view_count);
        let mut previous_name: Option<String> = None;
        let mut expected_data_offset = header.data_offset;
        for index in 0..header.view_count {
            let start = index
                .checked_mul(LQTP2_VIEW_ENTRY_LEN)
                .ok_or(PackV2Error::IntegerOverflow("directory index"))?;
            let descriptor =
                ViewDescriptor::parse(&directory[start..start + LQTP2_VIEW_ENTRY_LEN])?;
            if previous_name
                .as_deref()
                .is_some_and(|previous| previous >= descriptor.name())
            {
                return Err(PackV2Error::InvalidLayout("view sort order"));
            }
            previous_name = Some(descriptor.name().to_owned());
            let aligned_data_offset = align_up(expected_data_offset, VIEW_ALIGNMENT)?;
            require_zero_padding(
                &mmap,
                expected_data_offset,
                aligned_data_offset,
                "view padding",
            )?;
            if descriptor.data_offset != aligned_data_offset {
                return Err(PackV2Error::InvalidLayout("view data offset"));
            }
            let expected_length = header
                .row_count
                .checked_mul(descriptor.row_stride())
                .ok_or(PackV2Error::IntegerOverflow("view data length"))?;
            if descriptor.data_length != expected_length {
                return Err(PackV2Error::InvalidLayout("view data length"));
            }
            let data = checked_slice(
                &mmap,
                descriptor.data_offset,
                descriptor.data_length,
                "view data",
            )?;
            if sha256(data) != descriptor.data_sha256 {
                return Err(PackV2Error::IntegrityMismatch("view data"));
            }
            if descriptor.dtype() == PackV2Dtype::Bool && data.iter().any(|value| *value > 1) {
                return Err(PackV2Error::InvalidLayout("boolean view data"));
            }
            expected_data_offset = descriptor
                .data_offset
                .checked_add(descriptor.data_length)
                .ok_or(PackV2Error::IntegerOverflow("view data end"))?;
            views.push(descriptor);
        }
        if expected_data_offset != header.file_length {
            return Err(PackV2Error::InvalidLayout("trailing or missing view data"));
        }
        Ok(Self {
            mmap,
            header,
            views,
            metadata_range: metadata_start..metadata_end_usize,
        })
    }

    pub const fn row_count(&self) -> u64 {
        self.header.row_count
    }

    pub const fn manifest_sha256(&self) -> &[u8; 32] {
        &self.header.manifest_sha256
    }

    pub const fn view_spec_sha256(&self) -> &[u8; 32] {
        &self.header.view_spec_sha256
    }

    pub fn metadata(&self) -> &[u8] {
        &self.mmap[self.metadata_range.clone()]
    }

    pub fn views(&self) -> &[ViewDescriptor] {
        &self.views
    }

    pub fn view_names(&self) -> Vec<&str> {
        self.views.iter().map(ViewDescriptor::name).collect()
    }

    pub fn view(&self, name: &str) -> Result<&ViewDescriptor, PackV2Error> {
        self.views
            .binary_search_by(|view| view.name().cmp(name))
            .ok()
            .map(|index| &self.views[index])
            .ok_or_else(|| PackV2Error::MissingView(name.into()))
    }

    /// Borrow one encoded row directly from the mmap.
    pub fn row_raw(&self, view_name: &str, row: usize) -> Result<&[u8], PackV2Error> {
        let view = self.view(view_name)?;
        let row = u64::try_from(row).map_err(|_| PackV2Error::RowOutOfBounds {
            row: u64::MAX,
            row_count: self.header.row_count,
        })?;
        if row >= self.header.row_count {
            return Err(PackV2Error::RowOutOfBounds {
                row,
                row_count: self.header.row_count,
            });
        }
        let offset = view
            .data_offset
            .checked_add(
                row.checked_mul(view.row_stride())
                    .ok_or(PackV2Error::IntegerOverflow("row offset"))?,
            )
            .ok_or(PackV2Error::IntegerOverflow("row offset"))?;
        checked_slice(&self.mmap, offset, view.row_stride(), "view row")
    }

    /// Decode one logical f32 row. Raw non-f32 views remain accessible through
    /// [`Self::row_raw`].
    pub fn dequantize_f32(&self, view_name: &str, row: usize) -> Result<Vec<f32>, PackV2Error> {
        let view = self.view(view_name)?;
        if view.dtype() != PackV2Dtype::F32 {
            return Err(PackV2Error::WrongDtype(view_name.into()));
        }
        let bytes = self.row_raw(view_name, row)?;
        let element_count = checked_usize(view.spec.element_count()?, "view elements")?;
        match view.encoding() {
            PackV2Encoding::Raw => Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            PackV2Encoding::BfpInt8 | PackV2Encoding::BfpInt16 => {
                let lanes = checked_usize(view.spec.row_shape[0], "BFP lanes")?;
                let scale_bytes = lanes
                    .checked_mul(4)
                    .ok_or(PackV2Error::IntegerOverflow("BFP scale bytes"))?;
                let scales: Vec<f32> = bytes[..scale_bytes]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let dtype = if view.encoding() == PackV2Encoding::BfpInt8 {
                    PackDtype::Int8
                } else {
                    PackDtype::Int16
                };
                Ok(dequantize_window(
                    &scales,
                    &bytes[scale_bytes..],
                    lanes,
                    element_count / lanes,
                    dtype,
                ))
            }
        }
    }
}

fn open_nofollow(path: &Path) -> Result<File, PackV2Error> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    Ok(options.open(path)?)
}

/// Fail-closed LQTP2 format error.
#[derive(Debug)]
pub enum PackV2Error {
    BadMagic,
    BadVersion(u8, u8),
    BadEndianness(u8),
    BadFlags(u8),
    BadDtype(u8),
    BadEncoding(u8),
    InvalidUtf8,
    InvalidViewName(String),
    InvalidLayout(&'static str),
    Truncated { expected: usize, actual: usize },
    ShapeMismatch(String),
    DuplicateView(String),
    MissingView(String),
    WrongEncoding(String),
    WrongDtype(String),
    RowOutOfBounds { row: u64, row_count: u64 },
    ManifestMismatch,
    ViewSpecMismatch,
    IntegrityMismatch(&'static str),
    IntegerOverflow(&'static str),
    Io(std::io::Error),
}

impl fmt::Display for PackV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not an LQTP2 pack (bad magic)"),
            Self::BadVersion(major, minor) => {
                write!(f, "unsupported LQTP2 version {major}.{minor}")
            }
            Self::BadEndianness(value) => write!(f, "unsupported LQTP2 endianness {value}"),
            Self::BadFlags(value) => write!(f, "unsupported LQTP2 flags {value:#x}"),
            Self::BadDtype(value) => write!(f, "unknown LQTP2 dtype tag {value}"),
            Self::BadEncoding(value) => write!(f, "unknown LQTP2 encoding tag {value}"),
            Self::InvalidUtf8 => f.write_str("invalid UTF-8 in LQTP2"),
            Self::InvalidViewName(name) => write!(f, "invalid LQTP2 view name '{name}'"),
            Self::InvalidLayout(context) => write!(f, "invalid LQTP2 {context}"),
            Self::Truncated { expected, actual } => {
                write!(
                    f,
                    "LQTP2 truncated: expected {expected} bytes, got {actual}"
                )
            }
            Self::ShapeMismatch(message) => write!(f, "LQTP2 shape mismatch: {message}"),
            Self::DuplicateView(name) => write!(f, "duplicate LQTP2 view '{name}'"),
            Self::MissingView(name) => write!(f, "missing LQTP2 view '{name}'"),
            Self::WrongEncoding(name) => write!(f, "wrong row encoding for LQTP2 view '{name}'"),
            Self::WrongDtype(name) => write!(f, "wrong logical dtype for LQTP2 view '{name}'"),
            Self::RowOutOfBounds { row, row_count } => {
                write!(f, "LQTP2 row {row} is outside 0..{row_count}")
            }
            Self::ManifestMismatch => f.write_str("LQTP2 manifest hash mismatch"),
            Self::ViewSpecMismatch => f.write_str("LQTP2 ViewSpec hash mismatch"),
            Self::IntegrityMismatch(context) => {
                write!(f, "LQTP2 integrity mismatch in {context}")
            }
            Self::IntegerOverflow(context) => write!(f, "LQTP2 integer overflow in {context}"),
            Self::Io(error) => write!(f, "LQTP2 I/O error: {error}"),
        }
    }
}

impl std::error::Error for PackV2Error {}

impl From<std::io::Error> for PackV2Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

fn write_sink_row(
    sink: &mut ViewSink,
    declared_rows: u64,
    bytes: &[u8],
) -> Result<(), PackV2Error> {
    if sink.rows_written >= declared_rows {
        return Err(PackV2Error::ShapeMismatch(format!(
            "view '{}' wrote more than {declared_rows} rows",
            sink.spec.name
        )));
    }
    let expected = checked_usize(sink.spec.row_stride, "row stride")?;
    if bytes.len() != expected {
        return Err(PackV2Error::ShapeMismatch(format!(
            "view '{}' encoded row has {} bytes, expected {expected}",
            sink.spec.name,
            bytes.len()
        )));
    }
    let file = sink
        .file
        .as_mut()
        .ok_or_else(|| PackV2Error::ShapeMismatch("writer already finished".into()))?;
    file.write_all(bytes)?;
    sink.hasher.update(bytes);
    sink.rows_written += 1;
    Ok(())
}

fn checked_product(values: &[u64], context: &'static str) -> Result<u64, PackV2Error> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(PackV2Error::IntegerOverflow(context))
    })
}

fn checked_usize(value: u64, context: &'static str) -> Result<usize, PackV2Error> {
    usize::try_from(value).map_err(|_| PackV2Error::IntegerOverflow(context))
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: u64,
    length: u64,
    context: &'static str,
) -> Result<&'a [u8], PackV2Error> {
    let offset = checked_usize(offset, context)?;
    let length = checked_usize(length, context)?;
    let end = offset
        .checked_add(length)
        .ok_or(PackV2Error::IntegerOverflow(context))?;
    bytes.get(offset..end).ok_or(PackV2Error::Truncated {
        expected: end,
        actual: bytes.len(),
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, PackV2Error> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(PackV2Error::IntegerOverflow("alignment"))
    }
}

fn require_zero_padding(
    bytes: &[u8],
    start: u64,
    end: u64,
    context: &'static str,
) -> Result<(), PackV2Error> {
    if end < start {
        return Err(PackV2Error::InvalidLayout(context));
    }
    if checked_slice(bytes, start, end - start, context)?
        .iter()
        .any(|value| *value != 0)
    {
        return Err(PackV2Error::InvalidLayout(context));
    }
    Ok(())
}

fn write_padding(writer: &mut impl Write, length: u64) -> Result<(), PackV2Error> {
    const ZEROES: [u8; 64] = [0; 64];
    let mut remaining = length;
    while remaining != 0 {
        let count = remaining.min(ZEROES.len() as u64) as usize;
        writer.write_all(&ZEROES[..count])?;
        remaining -= count as u64;
    }
    Ok(())
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    value.into()
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    finalize_hash(Sha256::digest(bytes))
}

fn finalize_hash(hash: impl AsRef<[u8]>) -> [u8; 32] {
    let mut out = [0_u8; 32];
    out.copy_from_slice(hash.as_ref());
    out
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, PackV2Error> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(PackV2Error::Truncated {
            expected: offset + 2,
            actual: bytes.len(),
        })?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PackV2Error> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(PackV2Error::Truncated {
            expected: offset + 4,
            actual: bytes.len(),
        })?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, PackV2Error> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(PackV2Error::Truncated {
            expected: offset + 8,
            actual: bytes.len(),
        })?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}
