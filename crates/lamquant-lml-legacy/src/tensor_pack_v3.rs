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
//! LQTP3: deterministic chunked training bundles.
//!
//! LQTP3 is a host-only derived-training format. It keeps LQTP1/LQTP2 frozen,
//! stores fixed-shape views in independently verified row chunks, and supports
//! raw, BFP8, and BFP16 rows with either no compression or deterministic zstd.
//! Metadata bytes are opaque to this layer; callers bind their canonical ABIR
//! or BCS2 source identity through the manifest/ViewSpec hashes and metadata.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};

use lamquant_lml_archive::tensor_pack::{dequantize_window, quantize_window, PackDtype};

pub const LQTP3_MAGIC: &[u8; 4] = b"LQT3";
pub const LQTP3_VERSION_MAJOR: u8 = 3;
pub const LQTP3_VERSION_MINOR: u8 = 0;
pub const LQTP3_HEADER_LEN: usize = 512;
pub const LQTP3_VIEW_ENTRY_LEN: usize = 256;
pub const LQTP3_CHUNK_ENTRY_LEN: usize = 128;
pub const LQTP3_RECEIPT_MAGIC: &[u8; 4] = b"LQVR";
pub const LQTP3_RECEIPT_LEN: usize = 224;

const ENDIAN_LITTLE: u8 = 1;
const FLAG_SHA256: u8 = 1;
const VIEW_FLAG_REQUIRED: u8 = 1;
const ALIGNMENT: u64 = 64;
const MAX_VIEWS: usize = 256;
const MAX_RANK: usize = 4;
const MAX_VIEW_NAME_BYTES: usize = 64;
const MAX_CHUNKS: usize = 1_000_000;
const MAX_METADATA_BYTES: usize = 256 * 1024 * 1024;
const MAX_CHUNK_ENCODED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CHUNK_DECODED_BYTES: u64 = 1024 * 1024 * 1024;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackV3Dtype {
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

impl PackV3Dtype {
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

    pub const fn width(self) -> u64 {
        match self {
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackV3Encoding {
    Raw,
    BfpInt8,
    BfpInt16,
}

impl PackV3Encoding {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackV3Compression {
    None,
    Zstd,
}

impl PackV3Compression {
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Zstd => 1,
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::None,
            1 => Self::Zstd,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewSpecV3 {
    name: String,
    dtype: PackV3Dtype,
    encoding: PackV3Encoding,
    rank: usize,
    row_shape: [u64; MAX_RANK],
    required: bool,
    spec_sha256: [u8; 32],
    chunk_rows: u32,
    compression: PackV3Compression,
    compression_level: i32,
    row_stride: u64,
    decoded_row_length: u64,
}

impl ViewSpecV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        dtype: PackV3Dtype,
        encoding: PackV3Encoding,
        row_shape: &[usize],
        required: bool,
        spec_sha256: [u8; 32],
        chunk_rows: usize,
        compression: PackV3Compression,
        compression_level: i32,
    ) -> Result<Self, PackV3Error> {
        let name = name.into();
        if name.is_empty() || name.len() > MAX_VIEW_NAME_BYTES || name.as_bytes().contains(&0) {
            return Err(PackV3Error::InvalidViewName(name));
        }
        if row_shape.is_empty() || row_shape.len() > MAX_RANK {
            return Err(PackV3Error::ShapeMismatch(format!(
                "view '{name}' rank {} is outside 1..={MAX_RANK}",
                row_shape.len()
            )));
        }
        let chunk_rows = u32::try_from(chunk_rows).map_err(|_| {
            PackV3Error::ShapeMismatch(format!("view '{name}' chunk_rows does not fit u32"))
        })?;
        if chunk_rows == 0 {
            return Err(PackV3Error::ShapeMismatch(format!(
                "view '{name}' chunk_rows is zero"
            )));
        }
        match compression {
            PackV3Compression::None if compression_level != 0 => {
                return Err(PackV3Error::ShapeMismatch(format!(
                    "view '{name}' uncompressed level must be zero"
                )))
            }
            PackV3Compression::Zstd
                if !zstd::compression_level_range().contains(&compression_level) =>
            {
                return Err(PackV3Error::ShapeMismatch(format!(
                    "view '{name}' zstd level {compression_level} is unsupported"
                )))
            }
            _ => {}
        }
        if encoding != PackV3Encoding::Raw && dtype != PackV3Dtype::F32 {
            return Err(PackV3Error::ShapeMismatch(format!(
                "view '{name}' BFP encoding requires logical f32"
            )));
        }
        let mut dimensions = [0_u64; MAX_RANK];
        for (index, value) in row_shape.iter().copied().enumerate() {
            if value == 0 {
                return Err(PackV3Error::ShapeMismatch(format!(
                    "view '{name}' dimension {index} is zero"
                )));
            }
            dimensions[index] =
                u64::try_from(value).map_err(|_| PackV3Error::IntegerOverflow("view dimension"))?;
        }
        let elements = checked_product(&dimensions[..row_shape.len()], "view elements")?;
        let decoded_row_length = elements
            .checked_mul(dtype.width())
            .ok_or(PackV3Error::IntegerOverflow("decoded row length"))?;
        let row_stride = match encoding {
            PackV3Encoding::Raw => decoded_row_length,
            PackV3Encoding::BfpInt8 => dimensions[0]
                .checked_mul(4)
                .and_then(|scales| scales.checked_add(elements))
                .ok_or(PackV3Error::IntegerOverflow("BFP8 row stride"))?,
            PackV3Encoding::BfpInt16 => dimensions[0]
                .checked_mul(4)
                .and_then(|scales| {
                    elements
                        .checked_mul(2)
                        .and_then(|mantissas| scales.checked_add(mantissas))
                })
                .ok_or(PackV3Error::IntegerOverflow("BFP16 row stride"))?,
        };
        validate_chunk_bounds(row_stride, decoded_row_length, chunk_rows as u64)?;
        Ok(Self {
            name,
            dtype,
            encoding,
            rank: row_shape.len(),
            row_shape: dimensions,
            required,
            spec_sha256,
            chunk_rows,
            compression,
            compression_level,
            row_stride,
            decoded_row_length,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub const fn dtype(&self) -> PackV3Dtype {
        self.dtype
    }
    pub const fn encoding(&self) -> PackV3Encoding {
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
    pub const fn chunk_rows(&self) -> u32 {
        self.chunk_rows
    }
    pub const fn compression(&self) -> PackV3Compression {
        self.compression
    }
    pub const fn compression_level(&self) -> i32 {
        self.compression_level
    }
    pub const fn row_stride(&self) -> u64 {
        self.row_stride
    }
    pub const fn decoded_row_length(&self) -> u64 {
        self.decoded_row_length
    }
    fn element_count(&self) -> Result<u64, PackV3Error> {
        checked_product(self.row_shape(), "view elements")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewDescriptorV3 {
    spec: ViewSpecV3,
    first_chunk: u64,
    chunk_count: u64,
    logical_view_sha256: [u8; 32],
}

impl ViewDescriptorV3 {
    pub fn name(&self) -> &str {
        self.spec.name()
    }
    pub const fn dtype(&self) -> PackV3Dtype {
        self.spec.dtype()
    }
    pub const fn encoding(&self) -> PackV3Encoding {
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
    pub const fn chunk_rows(&self) -> u32 {
        self.spec.chunk_rows()
    }
    pub const fn compression(&self) -> PackV3Compression {
        self.spec.compression()
    }
    pub const fn compression_level(&self) -> i32 {
        self.spec.compression_level()
    }
    pub const fn row_stride(&self) -> u64 {
        self.spec.row_stride()
    }
    pub const fn decoded_row_length(&self) -> u64 {
        self.spec.decoded_row_length()
    }
    pub const fn first_chunk(&self) -> u64 {
        self.first_chunk
    }
    pub const fn chunk_count(&self) -> u64 {
        self.chunk_count
    }
    pub const fn logical_view_sha256(&self) -> &[u8; 32] {
        &self.logical_view_sha256
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let start = out.len();
        let name = self.spec.name.as_bytes();
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.push(self.spec.dtype.to_u8());
        out.push(self.spec.encoding.to_u8());
        out.push(self.spec.compression.to_u8());
        out.push(u8::from(self.spec.required) * VIEW_FLAG_REQUIRED);
        out.push(self.spec.rank as u8);
        out.push(0);
        out.extend_from_slice(&self.spec.chunk_rows.to_le_bytes());
        out.extend_from_slice(&self.spec.compression_level.to_le_bytes());
        out.extend_from_slice(&self.first_chunk.to_le_bytes());
        out.extend_from_slice(&self.chunk_count.to_le_bytes());
        out.extend_from_slice(&self.spec.row_stride.to_le_bytes());
        out.extend_from_slice(&self.spec.decoded_row_length.to_le_bytes());
        for dimension in self.spec.row_shape {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&self.spec.spec_sha256);
        out.extend_from_slice(&self.logical_view_sha256);
        out.extend_from_slice(name);
        out.resize(out.len() + MAX_VIEW_NAME_BYTES - name.len(), 0);
        out.resize(start + LQTP3_VIEW_ENTRY_LEN, 0);
    }

    fn parse(bytes: &[u8]) -> Result<Self, PackV3Error> {
        if bytes.len() != LQTP3_VIEW_ENTRY_LEN {
            return Err(PackV3Error::InvalidLayout("view entry length"));
        }
        let name_len = read_u16(bytes, 0)? as usize;
        if name_len == 0 || name_len > MAX_VIEW_NAME_BYTES {
            return Err(PackV3Error::InvalidLayout("view name length"));
        }
        let name_storage = &bytes[144..208];
        if name_storage[name_len..].iter().any(|value| *value != 0)
            || bytes[208..].iter().any(|value| *value != 0)
            || bytes[7] != 0
        {
            return Err(PackV3Error::InvalidLayout("view reserved bytes"));
        }
        let name = std::str::from_utf8(&name_storage[..name_len])
            .map_err(|_| PackV3Error::InvalidUtf8)?
            .to_owned();
        let dtype = PackV3Dtype::from_u8(bytes[2]).ok_or(PackV3Error::BadDtype(bytes[2]))?;
        let encoding =
            PackV3Encoding::from_u8(bytes[3]).ok_or(PackV3Error::BadEncoding(bytes[3]))?;
        let compression =
            PackV3Compression::from_u8(bytes[4]).ok_or(PackV3Error::BadCompression(bytes[4]))?;
        if bytes[5] & !VIEW_FLAG_REQUIRED != 0 {
            return Err(PackV3Error::InvalidLayout("view flags"));
        }
        let rank = bytes[6] as usize;
        if rank == 0 || rank > MAX_RANK {
            return Err(PackV3Error::InvalidLayout("view rank"));
        }
        let mut dimensions = [0_u64; MAX_RANK];
        for (index, dimension) in dimensions.iter_mut().enumerate() {
            *dimension = read_u64(bytes, 48 + index * 8)?;
        }
        if dimensions[..rank].contains(&0) || dimensions[rank..].iter().any(|value| *value != 0) {
            return Err(PackV3Error::InvalidLayout("view dimensions"));
        }
        let shape = dimensions[..rank]
            .iter()
            .map(|value| checked_usize(*value, "view dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut spec_hash = [0_u8; 32];
        spec_hash.copy_from_slice(&bytes[80..112]);
        let spec = ViewSpecV3::new(
            name,
            dtype,
            encoding,
            &shape,
            bytes[5] & VIEW_FLAG_REQUIRED != 0,
            spec_hash,
            read_u32(bytes, 8)? as usize,
            compression,
            read_i32(bytes, 12)?,
        )?;
        if spec.row_stride != read_u64(bytes, 32)?
            || spec.decoded_row_length != read_u64(bytes, 40)?
        {
            return Err(PackV3Error::InvalidLayout("view row lengths"));
        }
        let mut logical_view_sha256 = [0_u8; 32];
        logical_view_sha256.copy_from_slice(&bytes[112..144]);
        Ok(Self {
            spec,
            first_chunk: read_u64(bytes, 16)?,
            chunk_count: read_u64(bytes, 24)?,
            logical_view_sha256,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkDescriptorV3 {
    view_index: u32,
    chunk_index: u32,
    row_start: u64,
    row_count: u32,
    stored_offset: u64,
    stored_length: u64,
    encoded_length: u64,
    decoded_length: u64,
    payload_sha256: [u8; 32],
    logical_sha256: [u8; 32],
}

impl ChunkDescriptorV3 {
    pub const fn view_index(&self) -> u32 {
        self.view_index
    }
    pub const fn chunk_index(&self) -> u32 {
        self.chunk_index
    }
    pub const fn row_start(&self) -> u64 {
        self.row_start
    }
    pub const fn row_count(&self) -> u32 {
        self.row_count
    }
    pub const fn stored_offset(&self) -> u64 {
        self.stored_offset
    }
    pub const fn payload_offset(&self) -> u64 {
        self.stored_offset
    }
    pub const fn stored_length(&self) -> u64 {
        self.stored_length
    }
    pub const fn encoded_length(&self) -> u64 {
        self.encoded_length
    }
    pub const fn decoded_length(&self) -> u64 {
        self.decoded_length
    }
    pub const fn payload_sha256(&self) -> &[u8; 32] {
        &self.payload_sha256
    }
    pub const fn logical_sha256(&self) -> &[u8; 32] {
        &self.logical_sha256
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&self.view_index.to_le_bytes());
        out.extend_from_slice(&self.chunk_index.to_le_bytes());
        out.extend_from_slice(&self.row_start.to_le_bytes());
        out.extend_from_slice(&self.row_count.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&self.stored_offset.to_le_bytes());
        out.extend_from_slice(&self.stored_length.to_le_bytes());
        out.extend_from_slice(&self.encoded_length.to_le_bytes());
        out.extend_from_slice(&self.decoded_length.to_le_bytes());
        out.extend_from_slice(&self.payload_sha256);
        out.extend_from_slice(&self.logical_sha256);
        out.resize(start + LQTP3_CHUNK_ENTRY_LEN, 0);
    }

    fn parse(bytes: &[u8]) -> Result<Self, PackV3Error> {
        if bytes.len() != LQTP3_CHUNK_ENTRY_LEN
            || read_u32(bytes, 20)? != 0
            || bytes[120..].iter().any(|value| *value != 0)
        {
            return Err(PackV3Error::InvalidLayout("chunk entry"));
        }
        let mut payload_sha256 = [0_u8; 32];
        payload_sha256.copy_from_slice(&bytes[56..88]);
        let mut logical_sha256 = [0_u8; 32];
        logical_sha256.copy_from_slice(&bytes[88..120]);
        Ok(Self {
            view_index: read_u32(bytes, 0)?,
            chunk_index: read_u32(bytes, 4)?,
            row_start: read_u64(bytes, 8)?,
            row_count: read_u32(bytes, 16)?,
            stored_offset: read_u64(bytes, 24)?,
            stored_length: read_u64(bytes, 32)?,
            encoded_length: read_u64(bytes, 40)?,
            decoded_length: read_u64(bytes, 48)?,
            payload_sha256,
            logical_sha256,
        })
    }
}

#[derive(Clone, Debug)]
struct HeaderV3 {
    view_count: usize,
    chunk_count: usize,
    row_count: u64,
    directory_offset: u64,
    directory_length: u64,
    metadata_offset: u64,
    metadata_length: u64,
    chunk_table_offset: u64,
    chunk_table_length: u64,
    data_offset: u64,
    file_length: u64,
    manifest_sha256: [u8; 32],
    view_spec_sha256: [u8; 32],
    metadata_sha256: [u8; 32],
    directory_sha256: [u8; 32],
    chunk_table_sha256: [u8; 32],
    logical_value_root_sha256: [u8; 32],
    artifact_root_sha256: [u8; 32],
}

impl HeaderV3 {
    fn encode(&self) -> [u8; LQTP3_HEADER_LEN] {
        let mut bytes = [0_u8; LQTP3_HEADER_LEN];
        bytes[..4].copy_from_slice(LQTP3_MAGIC);
        bytes[4] = LQTP3_VERSION_MAJOR;
        bytes[5] = LQTP3_VERSION_MINOR;
        bytes[6] = ENDIAN_LITTLE;
        bytes[7] = FLAG_SHA256;
        bytes[8..12].copy_from_slice(&(LQTP3_HEADER_LEN as u32).to_le_bytes());
        bytes[12..16].copy_from_slice(&(self.view_count as u32).to_le_bytes());
        bytes[16..24].copy_from_slice(&(self.chunk_count as u64).to_le_bytes());
        bytes[24..32].copy_from_slice(&self.row_count.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.directory_offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.directory_length.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.metadata_offset.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.metadata_length.to_le_bytes());
        bytes[64..72].copy_from_slice(&self.chunk_table_offset.to_le_bytes());
        bytes[72..80].copy_from_slice(&self.chunk_table_length.to_le_bytes());
        bytes[80..88].copy_from_slice(&self.data_offset.to_le_bytes());
        bytes[88..96].copy_from_slice(&self.file_length.to_le_bytes());
        bytes[96..128].copy_from_slice(&self.manifest_sha256);
        bytes[128..160].copy_from_slice(&self.view_spec_sha256);
        bytes[160..192].copy_from_slice(&self.metadata_sha256);
        bytes[192..224].copy_from_slice(&self.directory_sha256);
        bytes[224..256].copy_from_slice(&self.chunk_table_sha256);
        bytes[256..288].copy_from_slice(&self.logical_value_root_sha256);
        bytes[288..320].copy_from_slice(&self.artifact_root_sha256);
        let header_sha = sha256(&bytes);
        bytes[320..352].copy_from_slice(&header_sha);
        bytes
    }

    fn parse(bytes: &[u8]) -> Result<Self, PackV3Error> {
        if bytes.len() < LQTP3_HEADER_LEN {
            return Err(PackV3Error::Truncated {
                expected: LQTP3_HEADER_LEN,
                actual: bytes.len(),
            });
        }
        if &bytes[..4] != LQTP3_MAGIC {
            return Err(PackV3Error::BadMagic);
        }
        if bytes[4] != LQTP3_VERSION_MAJOR || bytes[5] != LQTP3_VERSION_MINOR {
            return Err(PackV3Error::BadVersion(bytes[4], bytes[5]));
        }
        if bytes[6] != ENDIAN_LITTLE {
            return Err(PackV3Error::BadEndianness(bytes[6]));
        }
        if bytes[7] != FLAG_SHA256 {
            return Err(PackV3Error::BadFlags(bytes[7]));
        }
        if read_u32(bytes, 8)? as usize != LQTP3_HEADER_LEN
            || bytes[352..LQTP3_HEADER_LEN].iter().any(|value| *value != 0)
        {
            return Err(PackV3Error::InvalidLayout("header fields"));
        }
        let mut stored_header_sha = [0_u8; 32];
        stored_header_sha.copy_from_slice(&bytes[320..352]);
        let mut canonical = [0_u8; LQTP3_HEADER_LEN];
        canonical.copy_from_slice(&bytes[..LQTP3_HEADER_LEN]);
        canonical[320..352].fill(0);
        if sha256(&canonical) != stored_header_sha {
            return Err(PackV3Error::IntegrityMismatch("header"));
        }
        let view_count = read_u32(bytes, 12)? as usize;
        let chunk_count = checked_usize(read_u64(bytes, 16)?, "chunk count")?;
        if view_count == 0 || view_count > MAX_VIEWS {
            return Err(PackV3Error::InvalidLayout("view count"));
        }
        if chunk_count == 0 || chunk_count > MAX_CHUNKS {
            return Err(PackV3Error::InvalidLayout("chunk count"));
        }
        if read_u64(bytes, 24)? == 0 {
            return Err(PackV3Error::InvalidLayout("row count"));
        }
        Ok(Self {
            view_count,
            chunk_count,
            row_count: read_u64(bytes, 24)?,
            directory_offset: read_u64(bytes, 32)?,
            directory_length: read_u64(bytes, 40)?,
            metadata_offset: read_u64(bytes, 48)?,
            metadata_length: read_u64(bytes, 56)?,
            chunk_table_offset: read_u64(bytes, 64)?,
            chunk_table_length: read_u64(bytes, 72)?,
            data_offset: read_u64(bytes, 80)?,
            file_length: read_u64(bytes, 88)?,
            manifest_sha256: hash_at(bytes, 96)?,
            view_spec_sha256: hash_at(bytes, 128)?,
            metadata_sha256: hash_at(bytes, 160)?,
            directory_sha256: hash_at(bytes, 192)?,
            chunk_table_sha256: hash_at(bytes, 224)?,
            logical_value_root_sha256: hash_at(bytes, 256)?,
            artifact_root_sha256: hash_at(bytes, 288)?,
        })
    }
}

#[derive(Debug)]
pub enum PackV3Error {
    BadMagic,
    BadVersion(u8, u8),
    BadEndianness(u8),
    BadFlags(u8),
    BadDtype(u8),
    BadEncoding(u8),
    BadCompression(u8),
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
    ChunkOutOfBounds { chunk: u64, chunk_count: u64 },
    ManifestMismatch,
    ViewSpecMismatch,
    ReceiptMismatch(&'static str),
    IntegrityMismatch(&'static str),
    IntegerOverflow(&'static str),
    PublicationStateUnknown(String),
    Io(std::io::Error),
}

impl fmt::Display for PackV3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not an LQTP3 pack (bad magic)"),
            Self::BadVersion(a, b) => write!(f, "unsupported LQTP3 version {a}.{b}"),
            Self::BadEndianness(v) => write!(f, "unsupported LQTP3 endianness {v}"),
            Self::BadFlags(v) => write!(f, "unsupported LQTP3 flags {v:#x}"),
            Self::BadDtype(v) => write!(f, "unknown LQTP3 dtype tag {v}"),
            Self::BadEncoding(v) => write!(f, "unknown LQTP3 encoding tag {v}"),
            Self::BadCompression(v) => write!(f, "unknown LQTP3 compression tag {v}"),
            Self::InvalidUtf8 => f.write_str("invalid UTF-8 in LQTP3"),
            Self::InvalidViewName(v) => write!(f, "invalid LQTP3 view name '{v}'"),
            Self::InvalidLayout(v) => write!(f, "invalid LQTP3 {v}"),
            Self::Truncated { expected, actual } => write!(
                f,
                "LQTP3 truncated: expected {expected} bytes, got {actual}"
            ),
            Self::ShapeMismatch(v) => write!(f, "LQTP3 shape mismatch: {v}"),
            Self::DuplicateView(v) => write!(f, "duplicate LQTP3 view '{v}'"),
            Self::MissingView(v) => write!(f, "missing LQTP3 view '{v}'"),
            Self::WrongEncoding(v) => write!(f, "wrong row encoding for LQTP3 view '{v}'"),
            Self::WrongDtype(v) => write!(f, "wrong logical dtype for LQTP3 view '{v}'"),
            Self::RowOutOfBounds { row, row_count } => {
                write!(f, "LQTP3 row {row} is outside 0..{row_count}")
            }
            Self::ChunkOutOfBounds { chunk, chunk_count } => {
                write!(f, "LQTP3 chunk {chunk} is outside 0..{chunk_count}")
            }
            Self::ManifestMismatch => f.write_str("LQTP3 manifest hash mismatch"),
            Self::ViewSpecMismatch => f.write_str("LQTP3 ViewSpec hash mismatch"),
            Self::ReceiptMismatch(v) => write!(f, "LQTP3 verification receipt mismatch in {v}"),
            Self::IntegrityMismatch(v) => write!(f, "LQTP3 integrity mismatch in {v}"),
            Self::IntegerOverflow(v) => write!(f, "LQTP3 integer overflow in {v}"),
            Self::PublicationStateUnknown(v) => {
                write!(f, "LQTP3 publication state is unknown: {v}")
            }
            Self::Io(v) => write!(f, "LQTP3 I/O error: {v}"),
        }
    }
}

impl std::error::Error for PackV3Error {}
impl From<std::io::Error> for PackV3Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone)]
struct TempChunk {
    row_start: u64,
    row_count: u32,
    temp_offset: u64,
    stored_length: u64,
    encoded_length: u64,
    decoded_length: u64,
    payload_sha256: [u8; 32],
    logical_sha256: [u8; 32],
}

struct ViewSinkV3 {
    spec: ViewSpecV3,
    file: Option<BufWriter<File>>,
    source_file: Option<File>,
    temp_path: PathBuf,
    temp_identity: FileIdentity,
    rows_written: u64,
    encoded_chunk: Vec<u8>,
    chunk_row_start: u64,
    chunks: Vec<TempChunk>,
    temp_length: u64,
    logical_hasher: Sha256,
}

pub struct PackV3Writer {
    final_path: PathBuf,
    publication_parent: File,
    publication_parent_identity: FileIdentity,
    partial_path: PathBuf,
    partial_identity: Option<FileIdentity>,
    partial_file: Option<File>,
    row_count: u64,
    manifest_sha256: [u8; 32],
    view_spec_sha256: [u8; 32],
    metadata: Vec<u8>,
    views: Vec<ViewSinkV3>,
    state: PackV3WriterState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackV3WriterState {
    Active,
    PublicationStateUnknown,
    Done,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FinishIoStageV3 {
    ChunkWrite,
    ChunkFlush,
    ChunkSync,
    PartialWrite,
    PartialFlush,
    PartialSync,
    PrePublish,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    volume: Option<u32>,
    #[cfg(windows)]
    index: Option<u64>,
}

impl FileIdentity {
    fn from_file(file: &File) -> std::io::Result<Self> {
        Self::from_metadata(&file.metadata()?)
    }

    fn from_metadata(metadata: &std::fs::Metadata) -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self {
                device: metadata.dev(),
                inode: metadata.ino(),
            })
        }
        #[cfg(windows)]
        {
            let _ = metadata;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "descriptor-safe LQTP3 file identity requires Linux",
            ))
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = metadata;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "file identity is unavailable on this platform",
            ))
        }
    }

    fn matches_path(self, path: &Path) -> std::io::Result<bool> {
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file() {
            return Ok(false);
        }
        Ok(Self::from_metadata(&metadata)? == self)
    }
}

fn retire_owned_file(
    path: &Path,
    identity: FileIdentity,
    file: &File,
    truncate_owned: bool,
) -> std::io::Result<()> {
    retire_owned_file_with_hook(path, identity, file, truncate_owned, |_| {})
}

fn retire_view_sink(sink: &mut ViewSinkV3, truncate_owned: bool) -> std::io::Result<()> {
    // Consume a pending BufWriter without flushing it after retirement. If Drop
    // truncated the descriptor first and then the BufWriter flushed, the
    // quarantine tombstone could be repopulated with buffered chunk data.
    let buffered_file = sink.file.take().map(|writer| writer.into_parts().0);
    let Some(file) = sink.source_file.take().or(buffered_file) else {
        return Ok(());
    };
    retire_owned_file(&sink.temp_path, sink.temp_identity, &file, truncate_owned)
}

fn retire_owned_file_with_hook(
    path: &Path,
    identity: FileIdentity,
    file: &File,
    truncate_owned: bool,
    after_identity_check: impl FnOnce(&Path),
) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let mut after_identity_check = Some(after_identity_check);
        for _ in 0..16 {
            let quarantine = sibling_with_suffix(
                path,
                &format!(
                    ".retired.{}.{}",
                    std::process::id(),
                    NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
                ),
            );
            match rename_noreplace(path, &quarantine) {
                Ok(()) => {
                    if identity.matches_path(&quarantine)? {
                        if let Some(hook) = after_identity_check.take() {
                            hook(&quarantine);
                        }
                        if truncate_owned {
                            file.set_len(0)?;
                        }
                        // Never unlink the quarantine pathname. Even after an
                        // identity check, path deletion would admit a new
                        // substitution race. Owned files remain as zero-byte
                        // tombstones; unowned replacements are never removed.
                        return Ok(());
                    }
                    // The atomically claimed path was a substitute. Restore it
                    // if the original name is still free; otherwise retain it
                    // under quarantine. Neither branch deletes it.
                    let _ = rename_noreplace(&quarantine, path);
                    if truncate_owned {
                        file.set_len(0)?;
                    }
                    return Ok(());
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    if truncate_owned {
                        file.set_len(0)?;
                    }
                    return Ok(());
                }
                Err(error) => {
                    if truncate_owned {
                        file.set_len(0)?;
                    }
                    return Err(error);
                }
            }
        }
        if truncate_owned {
            file.set_len(0)?;
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique LQTP3 retirement name",
        ))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, identity, after_identity_check);
        // No portable atomic conditional-unlink primitive exists. Retain the
        // pathname conservatively and release owned storage through the exact
        // descriptor instead of risking deletion of a substitute.
        if truncate_owned {
            file.set_len(0)?;
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn rename_noreplace(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let from = CString::new(from.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "LQTP3 path contains a NUL byte",
        )
    })?;
    let to = CString::new(to.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "LQTP3 path contains a NUL byte",
        )
    })?;
    // SAFETY: both pointers are live NUL-terminated path strings for the
    // duration of the syscall; RENAME_NOREPLACE never overwrites `to`.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

impl PackV3Writer {
    pub fn create(
        path: &Path,
        row_count: usize,
        manifest_sha256: [u8; 32],
        view_spec_sha256: [u8; 32],
        metadata: Vec<u8>,
        mut specs: Vec<ViewSpecV3>,
    ) -> Result<Self, PackV3Error> {
        if row_count == 0 {
            return Err(PackV3Error::ShapeMismatch("row count is zero".into()));
        }
        if metadata.len() > MAX_METADATA_BYTES {
            return Err(PackV3Error::ShapeMismatch(
                "metadata exceeds bounded limit".into(),
            ));
        }
        if specs.is_empty() || specs.len() > MAX_VIEWS {
            return Err(PackV3Error::ShapeMismatch(format!(
                "view count {} is outside 1..={MAX_VIEWS}",
                specs.len()
            )));
        }
        let parent_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // Bind the destination directory (and, for relative paths, the current
        // working directory) at writer creation. Publication never redirects
        // to a later same-name directory or traverses a symlink component.
        let publication_parent = open_publication_parent(parent_path)?;
        let publication_parent_identity = FileIdentity::from_file(&publication_parent)?;
        if path.symlink_metadata().is_ok() {
            return Err(PackV3Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "LQTP3 destination already exists",
            )));
        }
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = specs.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(PackV3Error::DuplicateView(pair[0].name.clone()));
        }
        let row_count =
            u64::try_from(row_count).map_err(|_| PackV3Error::IntegerOverflow("row count"))?;
        let token = format!(
            "{}.{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        );
        let partial_path = sibling_with_suffix(path, &format!(".partial.{token}"));
        let mut views: Vec<ViewSinkV3> = Vec::with_capacity(specs.len());
        for (index, spec) in specs.into_iter().enumerate() {
            let temp_path = sibling_with_suffix(path, &format!(".chunks.{token}.{index}"));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => match FileIdentity::from_file(&file) {
                    Ok(temp_identity) => views.push(ViewSinkV3 {
                        spec,
                        file: Some(BufWriter::new(file)),
                        source_file: None,
                        temp_path,
                        temp_identity,
                        rows_written: 0,
                        encoded_chunk: Vec::new(),
                        chunk_row_start: 0,
                        chunks: Vec::new(),
                        temp_length: 0,
                        logical_hasher: Sha256::new(),
                    }),
                    Err(error) => {
                        for view in &mut views {
                            let _ = retire_view_sink(view, true);
                        }
                        return Err(PackV3Error::Io(error));
                    }
                },
                Err(error) => {
                    for view in &mut views {
                        let _ = retire_view_sink(view, true);
                    }
                    return Err(PackV3Error::Io(error));
                }
            }
        }
        Ok(Self {
            final_path: path.to_path_buf(),
            publication_parent,
            publication_parent_identity,
            partial_path,
            partial_identity: None,
            partial_file: None,
            row_count,
            manifest_sha256,
            view_spec_sha256,
            metadata,
            views,
            state: PackV3WriterState::Active,
        })
    }

    pub fn write_raw_row(&mut self, view_name: &str, bytes: &[u8]) -> Result<(), PackV3Error> {
        let row_count = self.row_count;
        let sink = self.sink_mut(view_name)?;
        if sink.spec.encoding != PackV3Encoding::Raw {
            return Err(PackV3Error::WrongEncoding(view_name.into()));
        }
        if sink.spec.dtype == PackV3Dtype::Bool && bytes.iter().any(|value| *value > 1) {
            return Err(PackV3Error::ShapeMismatch(format!(
                "view '{view_name}' boolean row contains a value other than 0 or 1"
            )));
        }
        append_row(sink, row_count, bytes)
    }

    pub fn write_f32_row(&mut self, view_name: &str, values: &[f32]) -> Result<(), PackV3Error> {
        let row_count = self.row_count;
        let sink = self.sink_mut(view_name)?;
        let expected = checked_usize(sink.spec.element_count()?, "view elements")?;
        if values.len() != expected {
            return Err(PackV3Error::ShapeMismatch(format!(
                "view '{view_name}' row has {} values, expected {expected}",
                values.len()
            )));
        }
        let encoded = match sink.spec.encoding {
            PackV3Encoding::Raw if sink.spec.dtype == PackV3Dtype::F32 => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            PackV3Encoding::Raw => return Err(PackV3Error::WrongDtype(view_name.into())),
            PackV3Encoding::BfpInt8 | PackV3Encoding::BfpInt16 => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(PackV3Error::ShapeMismatch(format!(
                        "view '{view_name}' BFP row contains a non-finite value"
                    )));
                }
                let lanes = checked_usize(sink.spec.row_shape[0], "BFP lanes")?;
                let dtype = if sink.spec.encoding == PackV3Encoding::BfpInt8 {
                    PackDtype::Int8
                } else {
                    PackDtype::Int16
                };
                let (scales, mantissas) =
                    quantize_window(values, lanes, values.len() / lanes, dtype);
                let mut bytes =
                    Vec::with_capacity(checked_usize(sink.spec.row_stride, "row stride")?);
                for scale in scales {
                    bytes.extend_from_slice(&scale.to_le_bytes());
                }
                bytes.extend_from_slice(&mantissas);
                bytes
            }
        };
        append_row(sink, row_count, &encoded)
    }

    pub fn finish(self) -> Result<(), PackV3Error> {
        self.finish_with_io_hook(|_| Ok(()))
    }

    fn finish_with_io_hook(
        mut self,
        mut before_io: impl FnMut(FinishIoStageV3) -> std::io::Result<()>,
    ) -> Result<(), PackV3Error> {
        for sink in &mut self.views {
            if sink.rows_written != self.row_count {
                return Err(PackV3Error::ShapeMismatch(format!(
                    "view '{}' wrote {} of {} rows",
                    sink.spec.name, sink.rows_written, self.row_count
                )));
            }
            before_io(FinishIoStageV3::ChunkWrite)?;
            flush_chunk(sink)?;
            let file = sink
                .file
                .as_mut()
                .ok_or_else(|| PackV3Error::ShapeMismatch("writer already finished".into()))?;
            before_io(FinishIoStageV3::ChunkFlush)?;
            file.flush()?;
            before_io(FinishIoStageV3::ChunkSync)?;
            file.get_ref().sync_all()?;
            let file = sink
                .file
                .take()
                .ok_or(PackV3Error::InvalidLayout("missing writer chunk file"))?;
            let (file, buffered) = file.into_parts();
            debug_assert!(matches!(buffered, Ok(bytes) if bytes.is_empty()));
            sink.source_file = Some(file);
        }
        let chunk_count = self.views.iter().try_fold(0_usize, |count, view| {
            count
                .checked_add(view.chunks.len())
                .ok_or(PackV3Error::IntegerOverflow("chunk count"))
        })?;
        if chunk_count == 0 || chunk_count > MAX_CHUNKS {
            return Err(PackV3Error::ShapeMismatch(format!(
                "chunk count {chunk_count} outside bounds"
            )));
        }

        let directory_length = (self.views.len() as u64)
            .checked_mul(LQTP3_VIEW_ENTRY_LEN as u64)
            .ok_or(PackV3Error::IntegerOverflow("directory length"))?;
        let metadata_offset = (LQTP3_HEADER_LEN as u64)
            .checked_add(directory_length)
            .ok_or(PackV3Error::IntegerOverflow("metadata offset"))?;
        let metadata_length = u64::try_from(self.metadata.len())
            .map_err(|_| PackV3Error::IntegerOverflow("metadata length"))?;
        let chunk_table_offset = align_up(
            metadata_offset
                .checked_add(metadata_length)
                .ok_or(PackV3Error::IntegerOverflow("chunk table offset"))?,
            ALIGNMENT,
        )?;
        let chunk_table_length = (chunk_count as u64)
            .checked_mul(LQTP3_CHUNK_ENTRY_LEN as u64)
            .ok_or(PackV3Error::IntegerOverflow("chunk table length"))?;
        let data_offset = align_up(
            chunk_table_offset
                .checked_add(chunk_table_length)
                .ok_or(PackV3Error::IntegerOverflow("data offset"))?,
            ALIGNMENT,
        )?;

        let mut descriptors = Vec::with_capacity(self.views.len());
        let mut first_chunk = 0_u64;
        for sink in &self.views {
            let chunk_count = sink.chunks.len() as u64;
            descriptors.push(ViewDescriptorV3 {
                spec: sink.spec.clone(),
                first_chunk,
                chunk_count,
                logical_view_sha256: finalize_hash(sink.logical_hasher.clone().finalize()),
            });
            first_chunk = first_chunk
                .checked_add(chunk_count)
                .ok_or(PackV3Error::IntegerOverflow("first chunk"))?;
        }
        let mut directory =
            Vec::with_capacity(checked_usize(directory_length, "directory length")?);
        for descriptor in &descriptors {
            descriptor.encode(&mut directory);
        }

        let mut chunks = Vec::with_capacity(chunk_count);
        let mut next_data_offset = data_offset;
        for (view_index, sink) in self.views.iter().enumerate() {
            for (chunk_index, chunk) in sink.chunks.iter().enumerate() {
                let stored_offset = align_up(next_data_offset, ALIGNMENT)?;
                next_data_offset = stored_offset
                    .checked_add(chunk.stored_length)
                    .ok_or(PackV3Error::IntegerOverflow("chunk data end"))?;
                chunks.push(ChunkDescriptorV3 {
                    view_index: view_index as u32,
                    chunk_index: chunk_index as u32,
                    row_start: chunk.row_start,
                    row_count: chunk.row_count,
                    stored_offset,
                    stored_length: chunk.stored_length,
                    encoded_length: chunk.encoded_length,
                    decoded_length: chunk.decoded_length,
                    payload_sha256: chunk.payload_sha256,
                    logical_sha256: chunk.logical_sha256,
                });
            }
        }
        let file_length = next_data_offset;
        let mut chunk_table =
            Vec::with_capacity(checked_usize(chunk_table_length, "chunk table length")?);
        for chunk in &chunks {
            chunk.encode(&mut chunk_table);
        }
        let metadata_sha256 = sha256(&self.metadata);
        let directory_sha256 = sha256(&directory);
        let chunk_table_sha256 = sha256(&chunk_table);
        let logical_value_root_sha256 = logical_value_root(self.row_count, &descriptors);
        let artifact_root_sha256 = artifact_root(
            &self.manifest_sha256,
            &self.view_spec_sha256,
            &metadata_sha256,
            &directory_sha256,
            &chunk_table_sha256,
            &chunks,
        );
        let header = HeaderV3 {
            view_count: descriptors.len(),
            chunk_count,
            row_count: self.row_count,
            directory_offset: LQTP3_HEADER_LEN as u64,
            directory_length,
            metadata_offset,
            metadata_length,
            chunk_table_offset,
            chunk_table_length,
            data_offset,
            file_length,
            manifest_sha256: self.manifest_sha256,
            view_spec_sha256: self.view_spec_sha256,
            metadata_sha256,
            directory_sha256,
            chunk_table_sha256,
            logical_value_root_sha256,
            artifact_root_sha256,
        }
        .encode();

        let output_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&self.partial_path)?;
        self.partial_file = Some(output_file);
        let partial_file = self
            .partial_file
            .as_ref()
            .ok_or(PackV3Error::InvalidLayout("missing writer partial file"))?;
        let partial_identity = FileIdentity::from_file(partial_file)?;
        self.partial_identity = Some(partial_identity);
        let mut output = HashingWriter::new(BufWriter::new(partial_file.try_clone()?));
        before_io(FinishIoStageV3::PartialWrite)?;
        output.write_all(&header)?;
        output.write_all(&directory)?;
        output.write_all(&self.metadata)?;
        write_padding(
            &mut output,
            chunk_table_offset - metadata_offset - metadata_length,
        )?;
        output.write_all(&chunk_table)?;
        write_padding(
            &mut output,
            data_offset - chunk_table_offset - chunk_table_length,
        )?;
        let mut position = data_offset;
        let mut flat_index = 0_usize;
        for sink in &mut self.views {
            let source = sink
                .source_file
                .as_mut()
                .ok_or(PackV3Error::InvalidLayout("missing writer chunk file"))?;
            for temp_chunk in &sink.chunks {
                let descriptor = &chunks[flat_index];
                write_padding(&mut output, descriptor.stored_offset - position)?;
                source.seek(SeekFrom::Start(temp_chunk.temp_offset))?;
                let mut limited = (&mut *source).take(temp_chunk.stored_length);
                let copied = std::io::copy(&mut limited, &mut output)?;
                if copied != temp_chunk.stored_length {
                    return Err(PackV3Error::Truncated {
                        expected: checked_usize(temp_chunk.stored_length, "chunk stored length")?,
                        actual: checked_usize(copied, "copied chunk length")?,
                    });
                }
                position = descriptor
                    .stored_offset
                    .checked_add(descriptor.stored_length)
                    .ok_or(PackV3Error::IntegerOverflow("chunk data end"))?;
                flat_index += 1;
            }
        }
        before_io(FinishIoStageV3::PartialFlush)?;
        output.flush()?;
        let (output, expected_bundle_sha256) = output.into_parts();
        let (output, buffered) = output.into_parts();
        debug_assert!(matches!(buffered, Ok(bytes) if bytes.is_empty()));
        drop(output);
        before_io(FinishIoStageV3::PartialSync)?;
        self.partial_file
            .as_ref()
            .ok_or(PackV3Error::InvalidLayout("missing writer partial file"))?
            .sync_all()?;
        let final_name = self.final_path.file_name().ok_or_else(|| {
            PackV3Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "LQTP3 destination has no file name",
            ))
        })?;
        // Keep a descriptor-created audit link before exposing the prepublish
        // seam. If the named staging path is unlinked or substituted and the
        // later directory sync fails, the synced source inode remains named.
        let _audit_name = create_staging_audit_link(
            self.partial_file
                .as_ref()
                .ok_or(PackV3Error::InvalidLayout("missing writer partial file"))?,
            &self.publication_parent,
            final_name,
        )?;
        self.publication_parent.sync_all()?;
        before_io(FinishIoStageV3::PrePublish)?;
        match publish_noreplace(
            &self.partial_path,
            partial_identity,
            self.partial_file
                .as_ref()
                .ok_or(PackV3Error::InvalidLayout("missing writer partial file"))?,
            expected_bundle_sha256,
            PublicationTarget {
                parent: &self.publication_parent,
                parent_identity: self.publication_parent_identity,
                final_path: &self.final_path,
            },
        ) {
            Ok(()) => {}
            Err(error @ PackV3Error::PublicationStateUnknown(_)) => {
                // The final no-replace link was created but its directory sync
                // failed. Retain every owned name for audit; Drop must not race
                // an external substitution by deleting any pathname.
                self.state = PackV3WriterState::PublicationStateUnknown;
                return Err(error);
            }
            Err(error) => return Err(error),
        }
        for sink in &mut self.views {
            let _ = retire_view_sink(sink, true);
        }
        self.state = PackV3WriterState::Done;
        Ok(())
    }

    fn sink_mut(&mut self, name: &str) -> Result<&mut ViewSinkV3, PackV3Error> {
        self.views
            .binary_search_by(|sink| sink.spec.name.as_str().cmp(name))
            .ok()
            .map(|index| &mut self.views[index])
            .ok_or_else(|| PackV3Error::MissingView(name.into()))
    }
}

impl Drop for PackV3Writer {
    fn drop(&mut self) {
        if self.state == PackV3WriterState::Active {
            if let Some(file) = self.partial_file.as_ref() {
                if let Some(identity) = self.partial_identity {
                    let _ = retire_owned_file(&self.partial_path, identity, file, true);
                } else {
                    // Identity acquisition may itself fail after creation. The
                    // exact retained descriptor can still release owned storage;
                    // retain the pathname rather than risk touching a substitute.
                    let _ = file.set_len(0);
                }
            }
            for sink in &mut self.views {
                let _ = retire_view_sink(sink, true);
            }
        }
    }
}

fn append_row(sink: &mut ViewSinkV3, declared_rows: u64, bytes: &[u8]) -> Result<(), PackV3Error> {
    if sink.rows_written >= declared_rows {
        return Err(PackV3Error::ShapeMismatch(format!(
            "view '{}' wrote more than {declared_rows} rows",
            sink.spec.name
        )));
    }
    let expected = checked_usize(sink.spec.row_stride, "row stride")?;
    if bytes.len() != expected {
        return Err(PackV3Error::ShapeMismatch(format!(
            "view '{}' encoded row has {} bytes, expected {expected}",
            sink.spec.name,
            bytes.len()
        )));
    }
    sink.encoded_chunk.extend_from_slice(bytes);
    sink.logical_hasher.update(bytes);
    sink.rows_written += 1;
    if sink.rows_written - sink.chunk_row_start == sink.spec.chunk_rows as u64 {
        flush_chunk(sink)?;
    }
    Ok(())
}

fn flush_chunk(sink: &mut ViewSinkV3) -> Result<(), PackV3Error> {
    let row_count = sink.rows_written - sink.chunk_row_start;
    if row_count == 0 {
        return Ok(());
    }
    if row_count > sink.spec.chunk_rows as u64 {
        return Err(PackV3Error::InvalidLayout("writer chunk row count"));
    }
    let encoded_length = u64::try_from(sink.encoded_chunk.len())
        .map_err(|_| PackV3Error::IntegerOverflow("encoded chunk length"))?;
    let expected_encoded = row_count
        .checked_mul(sink.spec.row_stride)
        .ok_or(PackV3Error::IntegerOverflow("encoded chunk length"))?;
    let decoded_length = row_count
        .checked_mul(sink.spec.decoded_row_length)
        .ok_or(PackV3Error::IntegerOverflow("decoded chunk length"))?;
    if encoded_length != expected_encoded {
        return Err(PackV3Error::InvalidLayout("writer encoded chunk length"));
    }
    validate_chunk_lengths(encoded_length, decoded_length)?;
    let logical_sha256 = sha256(&sink.encoded_chunk);
    let stored = match sink.spec.compression {
        PackV3Compression::None => sink.encoded_chunk.clone(),
        PackV3Compression::Zstd => {
            zstd::bulk::compress(&sink.encoded_chunk, sink.spec.compression_level)?
        }
    };
    let stored_length = u64::try_from(stored.len())
        .map_err(|_| PackV3Error::IntegerOverflow("stored chunk length"))?;
    if stored_length == 0 {
        return Err(PackV3Error::InvalidLayout("empty stored chunk"));
    }
    let file = sink
        .file
        .as_mut()
        .ok_or_else(|| PackV3Error::ShapeMismatch("writer already finished".into()))?;
    file.write_all(&stored)?;
    sink.chunks.push(TempChunk {
        row_start: sink.chunk_row_start,
        row_count: u32::try_from(row_count)
            .map_err(|_| PackV3Error::IntegerOverflow("chunk rows"))?,
        temp_offset: sink.temp_length,
        stored_length,
        encoded_length,
        decoded_length,
        payload_sha256: sha256(&stored),
        logical_sha256,
    });
    sink.temp_length = sink
        .temp_length
        .checked_add(stored_length)
        .ok_or(PackV3Error::IntegerOverflow("temporary chunk length"))?;
    sink.chunk_row_start = sink.rows_written;
    sink.encoded_chunk.clear();
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationReceiptV3 {
    file_size: u64,
    mtime_ns: u64,
    artifact_root_sha256: [u8; 32],
    logical_value_root_sha256: [u8; 32],
    manifest_sha256: [u8; 32],
    view_spec_sha256: [u8; 32],
    bundle_sha256: [u8; 32],
}

impl VerificationReceiptV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        file_size: u64,
        mtime_ns: u64,
        artifact_root_sha256: [u8; 32],
        logical_value_root_sha256: [u8; 32],
        manifest_sha256: [u8; 32],
        view_spec_sha256: [u8; 32],
        bundle_sha256: [u8; 32],
    ) -> Self {
        Self {
            file_size,
            mtime_ns,
            artifact_root_sha256,
            logical_value_root_sha256,
            manifest_sha256,
            view_spec_sha256,
            bundle_sha256,
        }
    }

    pub const fn file_size(&self) -> u64 {
        self.file_size
    }
    pub const fn mtime_ns(&self) -> u64 {
        self.mtime_ns
    }
    pub const fn artifact_root_sha256(&self) -> &[u8; 32] {
        &self.artifact_root_sha256
    }
    pub const fn logical_value_root_sha256(&self) -> &[u8; 32] {
        &self.logical_value_root_sha256
    }
    pub const fn manifest_sha256(&self) -> &[u8; 32] {
        &self.manifest_sha256
    }
    pub const fn view_spec_sha256(&self) -> &[u8; 32] {
        &self.view_spec_sha256
    }
    pub const fn bundle_sha256(&self) -> &[u8; 32] {
        &self.bundle_sha256
    }

    pub fn encode(&self) -> [u8; LQTP3_RECEIPT_LEN] {
        let mut bytes = [0_u8; LQTP3_RECEIPT_LEN];
        bytes[..4].copy_from_slice(LQTP3_RECEIPT_MAGIC);
        bytes[4] = 1;
        bytes[5] = FLAG_SHA256;
        bytes[6..8].copy_from_slice(&(LQTP3_RECEIPT_LEN as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&self.file_size.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.mtime_ns.to_le_bytes());
        bytes[24..56].copy_from_slice(&self.artifact_root_sha256);
        bytes[56..88].copy_from_slice(&self.logical_value_root_sha256);
        bytes[88..120].copy_from_slice(&self.manifest_sha256);
        bytes[120..152].copy_from_slice(&self.view_spec_sha256);
        bytes[152..184].copy_from_slice(&self.bundle_sha256);
        let receipt_sha = sha256(&bytes);
        bytes[192..224].copy_from_slice(&receipt_sha);
        bytes
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, PackV3Error> {
        if bytes.len() != LQTP3_RECEIPT_LEN {
            return Err(PackV3Error::ReceiptMismatch("length"));
        }
        if &bytes[..4] != LQTP3_RECEIPT_MAGIC
            || bytes[4] != 1
            || bytes[5] != FLAG_SHA256
            || read_u16(bytes, 6)? as usize != LQTP3_RECEIPT_LEN
            || bytes[184..192].iter().any(|value| *value != 0)
        {
            return Err(PackV3Error::ReceiptMismatch("header"));
        }
        let stored_sha = hash_at(bytes, 192)?;
        let mut canonical = [0_u8; LQTP3_RECEIPT_LEN];
        canonical.copy_from_slice(bytes);
        canonical[192..224].fill(0);
        if sha256(&canonical) != stored_sha {
            return Err(PackV3Error::ReceiptMismatch("checksum"));
        }
        Ok(Self {
            file_size: read_u64(bytes, 8)?,
            mtime_ns: read_u64(bytes, 16)?,
            artifact_root_sha256: hash_at(bytes, 24)?,
            logical_value_root_sha256: hash_at(bytes, 56)?,
            manifest_sha256: hash_at(bytes, 88)?,
            view_spec_sha256: hash_at(bytes, 120)?,
            bundle_sha256: hash_at(bytes, 152)?,
        })
    }
}

pub struct PackV3Reader {
    mmap: memmap2::Mmap,
    header: HeaderV3,
    views: Vec<ViewDescriptorV3>,
    chunks: Vec<ChunkDescriptorV3>,
    metadata_range: Range<usize>,
    verified_chunks: Mutex<Vec<bool>>,
    decoded_cache: Mutex<DecodedChunkCache>,
    file_size: u64,
    mtime_ns: u64,
}

impl PackV3Reader {
    pub fn open(
        path: &Path,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
    ) -> Result<Self, PackV3Error> {
        Self::open_with_cache_slots(path, expected_manifest_sha256, expected_view_spec_sha256, 0)
    }

    pub fn open_with_cache_slots(
        path: &Path,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        chunk_cache_slots: usize,
    ) -> Result<Self, PackV3Error> {
        Self::from_file_with_cache_slots(
            open_nofollow(path)?,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            chunk_cache_slots,
        )
    }

    /// Fully verify a snapshot through an already-open, owned descriptor.
    pub fn from_file(
        file: File,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
    ) -> Result<Self, PackV3Error> {
        Self::from_file_with_cache_slots(
            file,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            0,
        )
    }

    /// Fully verify a snapshot through an owned descriptor with a decoded
    /// chunk-cache capacity.
    pub fn from_file_with_cache_slots(
        file: File,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        chunk_cache_slots: usize,
    ) -> Result<Self, PackV3Error> {
        Self::open_inner(
            file,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            None,
            chunk_cache_slots,
        )
    }

    pub fn open_verified(
        path: &Path,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        receipt_bytes: &[u8],
    ) -> Result<Self, PackV3Error> {
        Self::open_verified_with_cache_slots(
            path,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            receipt_bytes,
            0,
        )
    }

    pub fn open_verified_with_cache_slots(
        path: &Path,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        receipt_bytes: &[u8],
        chunk_cache_slots: usize,
    ) -> Result<Self, PackV3Error> {
        Self::from_verified_file_with_cache_slots(
            open_nofollow(path)?,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            receipt_bytes,
            chunk_cache_slots,
        )
    }

    /// Open an owned descriptor using a verification receipt and lazy payload
    /// verification.
    pub fn from_verified_file(
        file: File,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        receipt_bytes: &[u8],
    ) -> Result<Self, PackV3Error> {
        Self::from_verified_file_with_cache_slots(
            file,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            receipt_bytes,
            0,
        )
    }

    /// Open an owned descriptor using a verification receipt and decoded
    /// chunk-cache capacity.
    pub fn from_verified_file_with_cache_slots(
        file: File,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        receipt_bytes: &[u8],
        chunk_cache_slots: usize,
    ) -> Result<Self, PackV3Error> {
        let receipt = VerificationReceiptV3::parse(receipt_bytes)?;
        Self::open_inner(
            file,
            expected_manifest_sha256,
            expected_view_spec_sha256,
            Some(receipt),
            chunk_cache_slots,
        )
    }

    fn open_inner(
        file: File,
        expected_manifest_sha256: Option<[u8; 32]>,
        expected_view_spec_sha256: Option<[u8; 32]>,
        receipt: Option<VerificationReceiptV3>,
        chunk_cache_slots: usize,
    ) -> Result<Self, PackV3Error> {
        let file_metadata = file.metadata()?;
        if !file_metadata.is_file() {
            return Err(PackV3Error::InvalidLayout("non-regular file"));
        }
        let file_size = file_metadata.len();
        let mtime_ns = metadata_mtime_ns(&file_metadata)?;
        if let Some(receipt) = &receipt {
            if receipt.file_size != file_size {
                return Err(PackV3Error::ReceiptMismatch("file size"));
            }
            if receipt.mtime_ns != mtime_ns {
                return Err(PackV3Error::ReceiptMismatch("mtime"));
            }
        }
        // SAFETY: read-only mapping of the same no-follow file descriptor whose
        // metadata was validated above. Published packs are immutable.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let header = HeaderV3::parse(&mmap)?;
        if header.file_length != file_size || header.file_length != mmap.len() as u64 {
            return Err(PackV3Error::InvalidLayout("file length"));
        }
        if expected_manifest_sha256.is_some_and(|value| value != header.manifest_sha256) {
            return Err(PackV3Error::ManifestMismatch);
        }
        if expected_view_spec_sha256.is_some_and(|value| value != header.view_spec_sha256) {
            return Err(PackV3Error::ViewSpecMismatch);
        }
        if let Some(receipt) = &receipt {
            if receipt.artifact_root_sha256 != header.artifact_root_sha256 {
                return Err(PackV3Error::ReceiptMismatch("artifact root"));
            }
            if receipt.logical_value_root_sha256 != header.logical_value_root_sha256 {
                return Err(PackV3Error::ReceiptMismatch("logical value root"));
            }
            if receipt.manifest_sha256 != header.manifest_sha256 {
                return Err(PackV3Error::ReceiptMismatch("manifest"));
            }
            if receipt.view_spec_sha256 != header.view_spec_sha256 {
                return Err(PackV3Error::ReceiptMismatch("ViewSpec"));
            }
        }
        let expected_directory_length = (header.view_count as u64)
            .checked_mul(LQTP3_VIEW_ENTRY_LEN as u64)
            .ok_or(PackV3Error::IntegerOverflow("directory length"))?;
        let expected_chunk_table_length = (header.chunk_count as u64)
            .checked_mul(LQTP3_CHUNK_ENTRY_LEN as u64)
            .ok_or(PackV3Error::IntegerOverflow("chunk table length"))?;
        if header.directory_offset != LQTP3_HEADER_LEN as u64
            || header.directory_length != expected_directory_length
            || header.metadata_offset
                != header
                    .directory_offset
                    .checked_add(header.directory_length)
                    .ok_or(PackV3Error::IntegerOverflow("metadata offset"))?
            || header.metadata_length > MAX_METADATA_BYTES as u64
        {
            return Err(PackV3Error::InvalidLayout("directory or metadata offsets"));
        }
        let expected_chunk_table_offset = align_up(
            header
                .metadata_offset
                .checked_add(header.metadata_length)
                .ok_or(PackV3Error::IntegerOverflow("chunk table offset"))?,
            ALIGNMENT,
        )?;
        if header.chunk_table_offset != expected_chunk_table_offset
            || header.chunk_table_length != expected_chunk_table_length
        {
            return Err(PackV3Error::InvalidLayout("chunk table offset or length"));
        }
        let expected_data_offset = align_up(
            header
                .chunk_table_offset
                .checked_add(header.chunk_table_length)
                .ok_or(PackV3Error::IntegerOverflow("data offset"))?,
            ALIGNMENT,
        )?;
        if header.data_offset != expected_data_offset {
            return Err(PackV3Error::InvalidLayout("data offset"));
        }
        let directory = checked_slice(
            &mmap,
            header.directory_offset,
            header.directory_length,
            "directory",
        )?;
        let metadata = checked_slice(
            &mmap,
            header.metadata_offset,
            header.metadata_length,
            "metadata",
        )?;
        let chunk_table = checked_slice(
            &mmap,
            header.chunk_table_offset,
            header.chunk_table_length,
            "chunk table",
        )?;
        if sha256(directory) != header.directory_sha256 {
            return Err(PackV3Error::IntegrityMismatch("directory"));
        }
        if sha256(metadata) != header.metadata_sha256 {
            return Err(PackV3Error::IntegrityMismatch("metadata"));
        }
        if sha256(chunk_table) != header.chunk_table_sha256 {
            return Err(PackV3Error::IntegrityMismatch("chunk table"));
        }
        require_zero_padding(
            &mmap,
            header.metadata_offset + header.metadata_length,
            header.chunk_table_offset,
            "metadata padding",
        )?;
        require_zero_padding(
            &mmap,
            header.chunk_table_offset + header.chunk_table_length,
            header.data_offset,
            "chunk table padding",
        )?;

        let mut views = Vec::with_capacity(header.view_count);
        let mut previous_name: Option<String> = None;
        for index in 0..header.view_count {
            let start = index
                .checked_mul(LQTP3_VIEW_ENTRY_LEN)
                .ok_or(PackV3Error::IntegerOverflow("directory index"))?;
            let descriptor =
                ViewDescriptorV3::parse(&directory[start..start + LQTP3_VIEW_ENTRY_LEN])?;
            if previous_name
                .as_deref()
                .is_some_and(|previous| previous >= descriptor.name())
            {
                return Err(PackV3Error::InvalidLayout("view sort order"));
            }
            previous_name = Some(descriptor.name().to_owned());
            views.push(descriptor);
        }
        let mut chunks = Vec::with_capacity(header.chunk_count);
        for index in 0..header.chunk_count {
            let start = index
                .checked_mul(LQTP3_CHUNK_ENTRY_LEN)
                .ok_or(PackV3Error::IntegerOverflow("chunk table index"))?;
            chunks.push(ChunkDescriptorV3::parse(
                &chunk_table[start..start + LQTP3_CHUNK_ENTRY_LEN],
            )?);
        }
        validate_layout(&mmap, &header, &views, &chunks, receipt.is_none())?;
        if logical_value_root(header.row_count, &views) != header.logical_value_root_sha256 {
            return Err(PackV3Error::IntegrityMismatch("logical value root"));
        }
        if artifact_root(
            &header.manifest_sha256,
            &header.view_spec_sha256,
            &header.metadata_sha256,
            &header.directory_sha256,
            &header.chunk_table_sha256,
            &chunks,
        ) != header.artifact_root_sha256
        {
            return Err(PackV3Error::IntegrityMismatch("artifact root"));
        }
        let metadata_start = checked_usize(header.metadata_offset, "metadata offset")?;
        let metadata_end = metadata_start
            .checked_add(checked_usize(header.metadata_length, "metadata length")?)
            .ok_or(PackV3Error::IntegerOverflow("metadata end"))?;
        let fully_verified = receipt.is_none();
        let verified_chunk_count = header.chunk_count;
        Ok(Self {
            mmap,
            header,
            views,
            chunks,
            metadata_range: metadata_start..metadata_end,
            verified_chunks: Mutex::new(vec![fully_verified; verified_chunk_count]),
            decoded_cache: Mutex::new(DecodedChunkCache::new(chunk_cache_slots)),
            file_size,
            mtime_ns,
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
    pub const fn logical_value_root_sha256(&self) -> &[u8; 32] {
        &self.header.logical_value_root_sha256
    }
    pub const fn materialized_value_root_sha256(&self) -> &[u8; 32] {
        &self.header.logical_value_root_sha256
    }
    pub const fn artifact_root_sha256(&self) -> &[u8; 32] {
        &self.header.artifact_root_sha256
    }
    /// Compatibility alias. Prefer [`Self::logical_value_root_sha256`].
    pub const fn logical_root_sha256(&self) -> &[u8; 32] {
        &self.header.logical_value_root_sha256
    }
    pub fn metadata(&self) -> &[u8] {
        &self.mmap[self.metadata_range.clone()]
    }
    pub fn views(&self) -> &[ViewDescriptorV3] {
        &self.views
    }
    pub fn chunks(&self) -> &[ChunkDescriptorV3] {
        &self.chunks
    }
    pub fn view_names(&self) -> Vec<&str> {
        self.views.iter().map(ViewDescriptorV3::name).collect()
    }
    pub fn view(&self, name: &str) -> Result<&ViewDescriptorV3, PackV3Error> {
        self.views
            .binary_search_by(|view| view.name().cmp(name))
            .ok()
            .map(|index| &self.views[index])
            .ok_or_else(|| PackV3Error::MissingView(name.into()))
    }
    pub fn chunks_for_view(&self, name: &str) -> Result<&[ChunkDescriptorV3], PackV3Error> {
        let view = self.view(name)?;
        checked_descriptor_slice(
            &self.chunks,
            view.first_chunk,
            view.chunk_count,
            "view chunks",
        )
    }

    pub fn chunk_stored_bytes(
        &self,
        view_name: &str,
        chunk_index: usize,
    ) -> Result<&[u8], PackV3Error> {
        let view = self.view(view_name)?;
        if chunk_index >= view.chunk_count as usize {
            return Err(PackV3Error::ChunkOutOfBounds {
                chunk: chunk_index as u64,
                chunk_count: view.chunk_count,
            });
        }
        let flat = checked_usize(view.first_chunk, "first chunk")?
            .checked_add(chunk_index)
            .ok_or(PackV3Error::IntegerOverflow("chunk index"))?;
        self.verify_chunk(flat)?;
        let chunk = &self.chunks[flat];
        checked_slice(
            &self.mmap,
            chunk.stored_offset,
            chunk.stored_length,
            "chunk payload",
        )
    }

    pub fn read_raw_row(&self, view_name: &str, row: usize) -> Result<Vec<u8>, PackV3Error> {
        let view = self.view(view_name)?;
        if view.encoding() != PackV3Encoding::Raw {
            return Err(PackV3Error::WrongEncoding(view_name.into()));
        }
        let (chunk_index, row_in_chunk) = self.locate_row(view, row)?;
        let encoded = self.decode_chunk(chunk_index)?;
        let stride = checked_usize(view.row_stride(), "row stride")?;
        let start = row_in_chunk
            .checked_mul(stride)
            .ok_or(PackV3Error::IntegerOverflow("row offset"))?;
        Ok(encoded[start..start + stride].to_vec())
    }

    pub fn gather_raw(&self, view_name: &str, rows: &[usize]) -> Result<Vec<u8>, PackV3Error> {
        let view = self.view(view_name)?;
        if view.encoding() != PackV3Encoding::Raw {
            return Err(PackV3Error::WrongEncoding(view_name.into()));
        }
        let stride = checked_usize(view.row_stride(), "row stride")?;
        let capacity = rows
            .len()
            .checked_mul(stride)
            .ok_or(PackV3Error::IntegerOverflow("gather bytes"))?;
        let mut grouped = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (output_index, &row) in rows.iter().enumerate() {
            let location = self.locate_row(view, row)?;
            grouped
                .entry(location.0)
                .or_default()
                .push((output_index, location.1));
        }
        let mut out = vec![0_u8; capacity];
        for (chunk_index, destinations) in grouped {
            let chunk = self.decode_chunk(chunk_index)?;
            for (output_index, row_in_chunk) in destinations {
                let source_start = row_in_chunk
                    .checked_mul(stride)
                    .ok_or(PackV3Error::IntegerOverflow("row offset"))?;
                let output_start = output_index
                    .checked_mul(stride)
                    .ok_or(PackV3Error::IntegerOverflow("gather offset"))?;
                out[output_start..output_start + stride]
                    .copy_from_slice(&chunk[source_start..source_start + stride]);
            }
        }
        Ok(out)
    }

    pub fn dequantize_f32(&self, view_name: &str, row: usize) -> Result<Vec<f32>, PackV3Error> {
        let view = self.view(view_name)?;
        if view.dtype() != PackV3Dtype::F32 {
            return Err(PackV3Error::WrongDtype(view_name.into()));
        }
        let (chunk_index, row_in_chunk) = self.locate_row(view, row)?;
        let encoded = self.decode_chunk(chunk_index)?;
        decode_f32_row(view, &encoded, row_in_chunk)
    }

    pub fn gather_f32(
        &self,
        view_name: &str,
        rows: &[usize],
    ) -> Result<Vec<Vec<f32>>, PackV3Error> {
        let view = self.view(view_name)?;
        if view.dtype() != PackV3Dtype::F32 {
            return Err(PackV3Error::WrongDtype(view_name.into()));
        }
        let elements = checked_usize(view.spec.element_count()?, "view elements")?;
        let mut grouped = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (output_index, &row) in rows.iter().enumerate() {
            let location = self.locate_row(view, row)?;
            grouped
                .entry(location.0)
                .or_default()
                .push((output_index, location.1));
        }
        let mut output = vec![vec![0_f32; elements]; rows.len()];
        for (chunk_index, destinations) in grouped {
            let chunk = self.decode_chunk(chunk_index)?;
            for (output_index, row_in_chunk) in destinations {
                output[output_index] = decode_f32_row(view, &chunk, row_in_chunk)?;
            }
        }
        Ok(output)
    }

    /// Gather f32 rows into one contiguous row-major allocation.
    ///
    /// Order and duplicate indices are preserved. This is the transfer-oriented
    /// counterpart to [`Self::gather_f32`]: callers such as PyO3 can move this
    /// allocation directly into their destination runtime without first
    /// materializing `Vec<Vec<f32>>` and then flattening it.
    pub fn gather_f32_flat(
        &self,
        view_name: &str,
        rows: &[usize],
    ) -> Result<Vec<f32>, PackV3Error> {
        let view = self.view(view_name)?;
        if view.dtype() != PackV3Dtype::F32 {
            return Err(PackV3Error::WrongDtype(view_name.into()));
        }
        let elements = checked_usize(view.spec.element_count()?, "view elements")?;
        let output_len = rows
            .len()
            .checked_mul(elements)
            .ok_or(PackV3Error::IntegerOverflow("gather elements"))?;
        let mut grouped = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (output_index, &row) in rows.iter().enumerate() {
            let location = self.locate_row(view, row)?;
            grouped
                .entry(location.0)
                .or_default()
                .push((output_index, location.1));
        }
        let mut output = vec![0_f32; output_len];
        for (chunk_index, destinations) in grouped {
            let chunk = self.decode_chunk(chunk_index)?;
            for (output_index, row_in_chunk) in destinations {
                let row = decode_f32_row(view, &chunk, row_in_chunk)?;
                let start = output_index
                    .checked_mul(elements)
                    .ok_or(PackV3Error::IntegerOverflow("gather offset"))?;
                output[start..start + elements].copy_from_slice(&row);
            }
        }
        Ok(output)
    }

    pub fn verification_receipt(&self, bundle_sha256: [u8; 32]) -> VerificationReceiptV3 {
        VerificationReceiptV3::new(
            self.file_size,
            self.mtime_ns,
            self.header.artifact_root_sha256,
            self.header.logical_value_root_sha256,
            self.header.manifest_sha256,
            self.header.view_spec_sha256,
            bundle_sha256,
        )
    }

    fn locate_row(
        &self,
        view: &ViewDescriptorV3,
        row: usize,
    ) -> Result<(usize, usize), PackV3Error> {
        let row_u64 = u64::try_from(row).map_err(|_| PackV3Error::IntegerOverflow("row"))?;
        if row_u64 >= self.header.row_count {
            return Err(PackV3Error::RowOutOfBounds {
                row: row_u64,
                row_count: self.header.row_count,
            });
        }
        let chunks = self.chunks_for_view(view.name())?;
        let relative = chunks
            .binary_search_by(|chunk| {
                if row_u64 < chunk.row_start {
                    std::cmp::Ordering::Greater
                } else if row_u64 >= chunk.row_start + chunk.row_count as u64 {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .map_err(|_| PackV3Error::InvalidLayout("row chunk coverage"))?;
        let flat = checked_usize(view.first_chunk, "first chunk")?
            .checked_add(relative)
            .ok_or(PackV3Error::IntegerOverflow("chunk index"))?;
        Ok((
            flat,
            checked_usize(row_u64 - chunks[relative].row_start, "row in chunk")?,
        ))
    }

    fn verify_chunk(&self, flat_index: usize) -> Result<(), PackV3Error> {
        let mut verified = self
            .verified_chunks
            .lock()
            .map_err(|_| PackV3Error::InvalidLayout("verification cache poisoned"))?;
        if verified
            .get(flat_index)
            .copied()
            .ok_or(PackV3Error::ChunkOutOfBounds {
                chunk: flat_index as u64,
                chunk_count: self.chunks.len() as u64,
            })?
        {
            return Ok(());
        }
        let chunk = &self.chunks[flat_index];
        let stored = checked_slice(
            &self.mmap,
            chunk.stored_offset,
            chunk.stored_length,
            "chunk payload",
        )?;
        if sha256(stored) != chunk.payload_sha256 {
            return Err(PackV3Error::IntegrityMismatch("chunk payload"));
        }
        verified[flat_index] = true;
        Ok(())
    }

    pub fn cache_stats(&self) -> Result<PackV3CacheStats, PackV3Error> {
        let cache = self
            .decoded_cache
            .lock()
            .map_err(|_| PackV3Error::InvalidLayout("decoded cache poisoned"))?;
        Ok(cache.stats())
    }

    fn decode_chunk(&self, flat_index: usize) -> Result<Arc<[u8]>, PackV3Error> {
        {
            let mut cache = self
                .decoded_cache
                .lock()
                .map_err(|_| PackV3Error::InvalidLayout("decoded cache poisoned"))?;
            if let Some(value) = cache.get(flat_index) {
                return Ok(value);
            }
        }
        self.verify_chunk(flat_index)?;
        let chunk = &self.chunks[flat_index];
        validate_chunk_lengths(chunk.encoded_length, chunk.decoded_length)?;
        let view = self
            .views
            .get(chunk.view_index as usize)
            .ok_or(PackV3Error::InvalidLayout("chunk view index"))?;
        let stored = checked_slice(
            &self.mmap,
            chunk.stored_offset,
            chunk.stored_length,
            "chunk payload",
        )?;
        let encoded = match view.compression() {
            PackV3Compression::None => stored.to_vec(),
            PackV3Compression::Zstd => zstd::bulk::decompress(
                stored,
                checked_usize(chunk.encoded_length, "encoded chunk length")?,
            )?,
        };
        if encoded.len() as u64 != chunk.encoded_length {
            return Err(PackV3Error::IntegrityMismatch("decoded chunk length"));
        }
        if sha256(&encoded) != chunk.logical_sha256 {
            return Err(PackV3Error::IntegrityMismatch("logical chunk"));
        }
        let encoded: Arc<[u8]> = encoded.into();
        let mut cache = self
            .decoded_cache
            .lock()
            .map_err(|_| PackV3Error::InvalidLayout("decoded cache poisoned"))?;
        cache.insert(flat_index, Arc::clone(&encoded));
        Ok(encoded)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackV3CacheStats {
    pub slots: usize,
    pub resident_chunks: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct DecodedChunkCache {
    slots: usize,
    entries: VecDeque<(usize, Arc<[u8]>)>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl DecodedChunkCache {
    fn new(slots: usize) -> Self {
        Self {
            slots,
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn get(&mut self, chunk: usize) -> Option<Arc<[u8]>> {
        let position = self.entries.iter().position(|(index, _)| *index == chunk);
        if let Some(position) = position {
            let entry = self
                .entries
                .remove(position)
                .expect("cache position exists");
            let value = Arc::clone(&entry.1);
            self.entries.push_back(entry);
            self.hits += 1;
            Some(value)
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, chunk: usize, value: Arc<[u8]>) {
        if self.slots == 0 {
            return;
        }
        if let Some(position) = self.entries.iter().position(|(index, _)| *index == chunk) {
            self.entries.remove(position);
        }
        if self.entries.len() == self.slots {
            self.entries.pop_front();
            self.evictions += 1;
        }
        self.entries.push_back((chunk, value));
    }

    fn stats(&self) -> PackV3CacheStats {
        PackV3CacheStats {
            slots: self.slots,
            resident_chunks: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }
}

fn decode_f32_row(
    view: &ViewDescriptorV3,
    encoded_chunk: &[u8],
    row_in_chunk: usize,
) -> Result<Vec<f32>, PackV3Error> {
    let stride = checked_usize(view.row_stride(), "row stride")?;
    let start = row_in_chunk
        .checked_mul(stride)
        .ok_or(PackV3Error::IntegerOverflow("row offset"))?;
    let end = start
        .checked_add(stride)
        .ok_or(PackV3Error::IntegerOverflow("row end"))?;
    let row = encoded_chunk
        .get(start..end)
        .ok_or(PackV3Error::InvalidLayout("row range"))?;
    let elements = checked_usize(view.spec.element_count()?, "view elements")?;
    match view.encoding() {
        PackV3Encoding::Raw => Ok(row
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        PackV3Encoding::BfpInt8 | PackV3Encoding::BfpInt16 => {
            let lanes = checked_usize(view.spec.row_shape[0], "BFP lanes")?;
            let scale_bytes = lanes
                .checked_mul(4)
                .ok_or(PackV3Error::IntegerOverflow("BFP scale bytes"))?;
            let scales = row[..scale_bytes]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            let dtype = if view.encoding() == PackV3Encoding::BfpInt8 {
                PackDtype::Int8
            } else {
                PackDtype::Int16
            };
            Ok(dequantize_window(
                &scales,
                &row[scale_bytes..],
                lanes,
                elements / lanes,
                dtype,
            ))
        }
    }
}

fn validate_layout(
    bytes: &[u8],
    header: &HeaderV3,
    views: &[ViewDescriptorV3],
    chunks: &[ChunkDescriptorV3],
    verify_payloads: bool,
) -> Result<(), PackV3Error> {
    let mut expected_first_chunk = 0_u64;
    let mut expected_stored_offset = header.data_offset;
    for (view_index, view) in views.iter().enumerate() {
        if view.first_chunk != expected_first_chunk || view.chunk_count == 0 {
            return Err(PackV3Error::InvalidLayout("view chunk range"));
        }
        let view_chunks =
            checked_descriptor_slice(chunks, view.first_chunk, view.chunk_count, "view chunks")?;
        let mut expected_row = 0_u64;
        let mut logical_hasher = Sha256::new();
        for (relative_index, chunk) in view_chunks.iter().enumerate() {
            if chunk.view_index as usize != view_index
                || chunk.chunk_index as usize != relative_index
                || chunk.row_start != expected_row
                || chunk.row_count == 0
                || chunk.row_count > view.chunk_rows()
            {
                return Err(PackV3Error::InvalidLayout("chunk identity or row coverage"));
            }
            let expected_encoded = (chunk.row_count as u64)
                .checked_mul(view.row_stride())
                .ok_or(PackV3Error::IntegerOverflow("encoded chunk length"))?;
            let expected_decoded = (chunk.row_count as u64)
                .checked_mul(view.decoded_row_length())
                .ok_or(PackV3Error::IntegerOverflow("decoded chunk length"))?;
            if chunk.encoded_length != expected_encoded
                || chunk.decoded_length != expected_decoded
                || chunk.stored_length == 0
            {
                return Err(PackV3Error::InvalidLayout("chunk lengths"));
            }
            validate_chunk_lengths(chunk.encoded_length, chunk.decoded_length)?;
            let aligned = align_up(expected_stored_offset, ALIGNMENT)?;
            require_zero_padding(bytes, expected_stored_offset, aligned, "chunk padding")?;
            if chunk.stored_offset != aligned {
                return Err(PackV3Error::InvalidLayout("chunk overlap or offset"));
            }
            let stored = checked_slice(
                bytes,
                chunk.stored_offset,
                chunk.stored_length,
                "chunk payload",
            )?;
            if view.compression() == PackV3Compression::None
                && chunk.stored_length != chunk.encoded_length
            {
                return Err(PackV3Error::InvalidLayout("uncompressed chunk length"));
            }
            if verify_payloads && sha256(stored) != chunk.payload_sha256 {
                return Err(PackV3Error::IntegrityMismatch("chunk payload"));
            }
            // Full open validates logical hash too. Deferred receipt mode checks
            // it on first decode, while still binding descriptor bytes at open.
            if verify_payloads {
                let encoded = match view.compression() {
                    PackV3Compression::None => stored.to_vec(),
                    PackV3Compression::Zstd => zstd::bulk::decompress(
                        stored,
                        checked_usize(chunk.encoded_length, "encoded chunk length")?,
                    )?,
                };
                if encoded.len() as u64 != chunk.encoded_length
                    || sha256(&encoded) != chunk.logical_sha256
                {
                    return Err(PackV3Error::IntegrityMismatch("logical chunk"));
                }
                logical_hasher.update(&encoded);
            }
            expected_row = expected_row
                .checked_add(chunk.row_count as u64)
                .ok_or(PackV3Error::IntegerOverflow("row coverage"))?;
            expected_stored_offset = chunk
                .stored_offset
                .checked_add(chunk.stored_length)
                .ok_or(PackV3Error::IntegerOverflow("chunk data end"))?;
        }
        if expected_row != header.row_count {
            return Err(PackV3Error::InvalidLayout("view row coverage"));
        }
        if verify_payloads && finalize_hash(logical_hasher.finalize()) != view.logical_view_sha256 {
            return Err(PackV3Error::IntegrityMismatch("logical view"));
        }
        expected_first_chunk = expected_first_chunk
            .checked_add(view.chunk_count)
            .ok_or(PackV3Error::IntegerOverflow("first chunk"))?;
    }
    if expected_first_chunk != chunks.len() as u64 {
        return Err(PackV3Error::InvalidLayout("unclaimed chunks"));
    }
    if expected_stored_offset != header.file_length {
        return Err(PackV3Error::InvalidLayout("trailing or missing chunk data"));
    }
    Ok(())
}

fn logical_value_root(row_count: u64, views: &[ViewDescriptorV3]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"LQTP3-MATERIALIZED-VALUE-ROOT\0");
    hasher.update(row_count.to_le_bytes());
    hasher.update((views.len() as u64).to_le_bytes());
    for view in views {
        hasher.update((view.name().len() as u16).to_le_bytes());
        hasher.update(view.name().as_bytes());
        hasher.update([
            view.dtype().to_u8(),
            view.encoding().to_u8(),
            view.spec.rank as u8,
        ]);
        for dimension in view.row_shape() {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update(view.logical_view_sha256());
    }
    finalize_hash(hasher.finalize())
}

fn artifact_root(
    manifest_sha256: &[u8; 32],
    view_spec_sha256: &[u8; 32],
    metadata_sha256: &[u8; 32],
    directory_sha256: &[u8; 32],
    chunk_table_sha256: &[u8; 32],
    chunks: &[ChunkDescriptorV3],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"LQTP3-ARTIFACT-ROOT\0");
    hasher.update(manifest_sha256);
    hasher.update(view_spec_sha256);
    hasher.update(metadata_sha256);
    hasher.update(directory_sha256);
    hasher.update(chunk_table_sha256);
    hasher.update((chunks.len() as u64).to_le_bytes());
    for chunk in chunks {
        hasher.update(chunk.payload_sha256());
    }
    finalize_hash(hasher.finalize())
}

fn validate_chunk_bounds(
    row_stride: u64,
    decoded_row_length: u64,
    chunk_rows: u64,
) -> Result<(), PackV3Error> {
    let encoded = row_stride
        .checked_mul(chunk_rows)
        .ok_or(PackV3Error::IntegerOverflow("maximum encoded chunk length"))?;
    let decoded = decoded_row_length
        .checked_mul(chunk_rows)
        .ok_or(PackV3Error::IntegerOverflow("maximum decoded chunk length"))?;
    validate_chunk_lengths(encoded, decoded)
}

fn validate_chunk_lengths(encoded: u64, decoded: u64) -> Result<(), PackV3Error> {
    if encoded == 0 || encoded > MAX_CHUNK_ENCODED_BYTES {
        return Err(PackV3Error::InvalidLayout("encoded chunk allocation bound"));
    }
    if decoded == 0 || decoded > MAX_CHUNK_DECODED_BYTES {
        return Err(PackV3Error::InvalidLayout("decoded chunk allocation bound"));
    }
    checked_usize(encoded, "encoded chunk allocation")?;
    checked_usize(decoded, "decoded chunk allocation")?;
    Ok(())
}

fn open_nofollow(path: &Path) -> Result<File, PackV3Error> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    Ok(options.open(path)?)
}

fn metadata_mtime_ns(metadata: &std::fs::Metadata) -> Result<u64, PackV3Error> {
    let duration = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PackV3Error::ReceiptMismatch("mtime before Unix epoch"))?;
    u64::try_from(duration.as_nanos())
        .map_err(|_| PackV3Error::IntegerOverflow("mtime nanoseconds"))
}

fn checked_product(values: &[u64], context: &'static str) -> Result<u64, PackV3Error> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(PackV3Error::IntegerOverflow(context))
    })
}

fn checked_usize(value: u64, context: &'static str) -> Result<usize, PackV3Error> {
    usize::try_from(value).map_err(|_| PackV3Error::IntegerOverflow(context))
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: u64,
    length: u64,
    context: &'static str,
) -> Result<&'a [u8], PackV3Error> {
    let offset = checked_usize(offset, context)?;
    let length = checked_usize(length, context)?;
    let end = offset
        .checked_add(length)
        .ok_or(PackV3Error::IntegerOverflow(context))?;
    bytes.get(offset..end).ok_or(PackV3Error::Truncated {
        expected: end,
        actual: bytes.len(),
    })
}

fn checked_descriptor_slice<'a, T>(
    values: &'a [T],
    offset: u64,
    length: u64,
    context: &'static str,
) -> Result<&'a [T], PackV3Error> {
    let offset = checked_usize(offset, context)?;
    let length = checked_usize(length, context)?;
    let end = offset
        .checked_add(length)
        .ok_or(PackV3Error::IntegerOverflow(context))?;
    values
        .get(offset..end)
        .ok_or(PackV3Error::InvalidLayout(context))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, PackV3Error> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(PackV3Error::IntegerOverflow("alignment"))
    }
}

fn require_zero_padding(
    bytes: &[u8],
    start: u64,
    end: u64,
    context: &'static str,
) -> Result<(), PackV3Error> {
    if end < start {
        return Err(PackV3Error::InvalidLayout(context));
    }
    if checked_slice(bytes, start, end - start, context)?
        .iter()
        .any(|value| *value != 0)
    {
        return Err(PackV3Error::InvalidLayout(context));
    }
    Ok(())
}

struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
}

impl<W> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn into_parts(self) -> (W, [u8; 32]) {
        (self.inner, self.hasher.finalize().into())
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let count = self.inner.write(bytes)?;
        self.hasher.update(&bytes[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn write_padding(writer: &mut impl Write, length: u64) -> Result<(), PackV3Error> {
    const ZEROES: [u8; 64] = [0; 64];
    let mut remaining = length;
    while remaining > 0 {
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

#[derive(Clone, Copy)]
struct PublicationTarget<'a> {
    parent: &'a File,
    parent_identity: FileIdentity,
    final_path: &'a Path,
}

struct PublicationHooks<S, R> {
    sync_parent: S,
    retire_staging: R,
}

fn publish_noreplace(
    partial_path: &Path,
    partial_identity: FileIdentity,
    partial_file: &File,
    expected_bundle_sha256: [u8; 32],
    target: PublicationTarget<'_>,
) -> Result<(), PackV3Error> {
    #[cfg(target_os = "linux")]
    {
        publish_noreplace_with_sync(
            partial_path,
            partial_identity,
            partial_file,
            expected_bundle_sha256,
            target,
            File::sync_all,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            partial_path,
            partial_identity,
            partial_file,
            expected_bundle_sha256,
            target,
        );
        Err(PackV3Error::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "descriptor-safe no-replace LQTP3 publication is unavailable on this platform",
        )))
    }
}

#[cfg(target_os = "linux")]
fn publish_noreplace_with_sync(
    partial_path: &Path,
    partial_identity: FileIdentity,
    partial_file: &File,
    expected_bundle_sha256: [u8; 32],
    target: PublicationTarget<'_>,
    sync_parent: impl FnOnce(&File) -> std::io::Result<()>,
) -> Result<(), PackV3Error> {
    publish_noreplace_with_hooks(
        partial_path,
        partial_identity,
        partial_file,
        expected_bundle_sha256,
        target,
        PublicationHooks {
            sync_parent,
            retire_staging: retire_owned_file,
        },
    )
}

#[cfg(target_os = "linux")]
fn publish_noreplace_with_hooks(
    partial_path: &Path,
    partial_identity: FileIdentity,
    partial_file: &File,
    expected_bundle_sha256: [u8; 32],
    target: PublicationTarget<'_>,
    hooks: PublicationHooks<
        impl FnOnce(&File) -> std::io::Result<()>,
        impl FnOnce(&Path, FileIdentity, &File, bool) -> std::io::Result<()>,
    >,
) -> Result<(), PackV3Error> {
    let PublicationHooks {
        sync_parent,
        retire_staging,
    } = hooks;
    let parent_path = target
        .final_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let final_name = target.final_path.file_name().ok_or_else(|| {
        PackV3Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "LQTP3 destination has no file name",
        ))
    })?;
    if !directory_path_matches_bound(parent_path, target.parent_identity)? {
        return Err(PackV3Error::Io(std::io::Error::other(
            "LQTP3 destination parent changed after writer creation",
        )));
    }
    // Publish a distinct inode. Linking the named staging inode directly would
    // leave the durable artifact with two links because safe staging cleanup
    // deliberately retains a zero-byte quarantine name. An anonymous file in
    // the already-bound destination directory keeps both no-replace and
    // descriptor authority without making the final artifact unevictable by
    // single-link CAS policy.
    let published_file = create_anonymous_file(target.parent)?;
    copy_exact_file_descriptor(partial_file, &published_file, expected_bundle_sha256)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let source_mode = partial_file.metadata()?.mode() & 0o7777;
        published_file.set_permissions(std::fs::Permissions::from_mode(source_mode))?;
    }
    published_file.sync_all()?;
    link_file_descriptor_noreplace(&published_file, target.parent, final_name)?;
    if let Err(sync_error) = sync_parent(target.parent) {
        // There is no portable atomic "unlink this path only if it still names
        // this inode" operation. A metadata check followed by remove_file has a
        // substitution TOCTOU, so never path-delete after publication. The
        // distinct error is an explicit committed-but-not-durable state: callers
        // must retain both names for audit and must not retry blindly.
        return Err(PackV3Error::PublicationStateUnknown(format!(
            "final no-replace link was created but directory sync failed ({sync_error}); names retained for audit; do not retry"
        )));
    }
    match directory_path_matches_bound(parent_path, target.parent_identity) {
        Ok(true) => {}
        Ok(false) => {
            return Err(PackV3Error::PublicationStateUnknown(
                "final link is durable in the bound parent, but the destination parent path changed; names retained for audit; do not retry"
                    .into(),
            ));
        }
        Err(identity_error) => {
            return Err(PackV3Error::PublicationStateUnknown(format!(
                "final link is durable in the bound parent, but the destination parent path could not be revalidated ({identity_error}); names retained for audit; do not retry"
            )));
        }
    }
    // The final link is durable and owns an independent inode. Retire and
    // truncate the exact staging descriptor, retaining only a zero-byte name;
    // cleanup failure is a distinct committed do-not-retry state, not an
    // ordinary publication failure.
    if let Err(retire_error) = retire_staging(partial_path, partial_identity, partial_file, true) {
        return Err(PackV3Error::PublicationStateUnknown(format!(
            "final link is durable but staging retirement failed ({retire_error}); names retained for audit; do not retry"
        )));
    }
    Ok(())
}

fn open_publication_parent(path: &Path) -> std::io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        open_directory_nofollow(path)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "descriptor-safe no-replace LQTP3 publication requires Linux O_TMPFILE",
        ))
    }
}

#[cfg(target_os = "linux")]
fn directory_path_matches_bound(
    path: &Path,
    expected_identity: FileIdentity,
) -> std::io::Result<bool> {
    let observed = open_directory_nofollow(path)?;
    Ok(FileIdentity::from_file(&observed)? == expected_identity)
}

#[cfg(target_os = "linux")]
fn create_staging_audit_link(
    staging_file: &File,
    publication_parent: &File,
    final_name: &std::ffi::OsStr,
) -> std::io::Result<std::ffi::OsString> {
    for _ in 0..16 {
        let mut audit_name = final_name.to_os_string();
        audit_name.push(format!(
            ".staging-audit.{}.{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        match link_file_descriptor_noreplace(staging_file, publication_parent, &audit_name) {
            Ok(()) => return Ok(audit_name),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique LQTP3 staging audit name",
    ))
}

#[cfg(not(target_os = "linux"))]
fn create_staging_audit_link(
    staging_file: &File,
    publication_parent: &File,
    final_name: &std::ffi::OsStr,
) -> std::io::Result<std::ffi::OsString> {
    let _ = (staging_file, publication_parent, final_name);
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "descriptor-safe LQTP3 staging audit links require Linux",
    ))
}

#[cfg(target_os = "linux")]
fn open_directory_nofollow(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;

    let mut components = path.components().peekable();
    let absolute = matches!(components.peek(), Some(Component::RootDir));
    let anchor = if absolute {
        b"/\0".as_slice()
    } else {
        b".\0".as_slice()
    };
    // SAFETY: `anchor` is NUL-terminated. A successful descriptor is owned by
    // the File constructed immediately below.
    let descriptor = unsafe {
        libc::open(
            anchor.as_ptr().cast(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `descriptor` is a fresh successful open result and ownership is
    // transferred exactly once to File.
    let mut directory = unsafe { File::from_raw_fd(descriptor) };

    for component in components {
        let name = match component {
            Component::RootDir | Component::CurDir => continue,
            Component::Normal(name) => name,
            Component::ParentDir => std::ffi::OsStr::new(".."),
            Component::Prefix(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "unsupported path prefix in Linux LQTP3 destination parent",
                ));
            }
        };
        let name = CString::new(name.as_bytes()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "LQTP3 destination parent contains a NUL byte",
            )
        })?;
        // SAFETY: `name` and the current directory descriptor remain live for
        // the call. O_NOFOLLOW rejects a symlink at every traversed component.
        let next = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if next < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `next` is a fresh successful openat result and ownership is
        // transferred exactly once, replacing and closing the prior component.
        directory = unsafe { File::from_raw_fd(next) };
    }
    Ok(directory)
}

#[cfg(target_os = "linux")]
fn create_anonymous_file(destination_parent: &File) -> std::io::Result<File> {
    use std::os::fd::{AsRawFd, FromRawFd};

    let current_directory = b".\0";
    // SAFETY: `current_directory` is NUL-terminated, the directory descriptor
    // is live, and a successful open returns a new owned descriptor. Omitting
    // O_EXCL deliberately permits linking the O_TMPFILE inode after it is
    // fully copied and synced.
    let descriptor = unsafe {
        libc::openat(
            destination_parent.as_raw_fd(),
            current_directory.as_ptr().cast(),
            libc::O_TMPFILE | libc::O_RDWR | libc::O_CLOEXEC,
            0o600,
        )
    };
    if descriptor < 0 {
        let error = std::io::Error::last_os_error();
        return Err(std::io::Error::new(
            error.kind(),
            format!(
                "Linux O_TMPFILE is required for descriptor-safe single-link LQTP3 publication: {error}"
            ),
        ));
    }
    // SAFETY: `descriptor` is a fresh successful openat result and ownership
    // is transferred exactly once to File.
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

#[cfg(target_os = "linux")]
fn copy_exact_file_descriptor(
    source: &File,
    destination: &File,
    expected_sha256: [u8; 32],
) -> std::io::Result<u64> {
    copy_exact_file_descriptor_with_hook(source, destination, expected_sha256, |_| Ok(()))
}

#[cfg(target_os = "linux")]
fn copy_exact_file_descriptor_with_hook(
    source: &File,
    destination: &File,
    expected_sha256: [u8; 32],
    mut after_chunk: impl FnMut(u64) -> std::io::Result<()>,
) -> std::io::Result<u64> {
    use std::os::unix::fs::FileExt;

    const COPY_BUFFER_BYTES: usize = 1024 * 1024;

    let expected_length = source.metadata()?.len();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut copied_hasher = Sha256::new();
    let mut offset = 0_u64;
    while offset < expected_length {
        let remaining = expected_length - offset;
        let bounded_length = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
            .map_err(|_| std::io::Error::other("LQTP3 publication copy length overflow"))?;
        let count = source.read_at(&mut buffer[..bounded_length], offset)?;
        if count == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "LQTP3 staging descriptor truncated during publication at byte {offset} of {expected_length}"
                ),
            ));
        }
        copied_hasher.update(&buffer[..count]);
        let mut written = 0_usize;
        while written < count {
            let write_offset = offset
                .checked_add(written as u64)
                .ok_or_else(|| std::io::Error::other("LQTP3 publication offset overflow"))?;
            let progress = destination.write_at(&buffer[written..count], write_offset)?;
            if progress == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to copy the complete LQTP3 staging descriptor",
                ));
            }
            written = written
                .checked_add(progress)
                .ok_or_else(|| std::io::Error::other("LQTP3 publication write overflow"))?;
        }
        offset = offset
            .checked_add(count as u64)
            .ok_or_else(|| std::io::Error::other("LQTP3 publication offset overflow"))?;
        after_chunk(offset)?;
    }

    // Reject concurrent growth as well as truncation. The destination must be
    // an exact descriptor-bound copy, never a prefix selected from a changing
    // staging inode.
    let mut trailing = [0_u8; 1];
    if source.metadata()?.len() != expected_length
        || source.read_at(&mut trailing, expected_length)? != 0
    {
        return Err(std::io::Error::other(
            "LQTP3 staging descriptor changed length during publication",
        ));
    }
    if destination.metadata()?.len() != expected_length {
        return Err(std::io::Error::other(
            "LQTP3 anonymous publication copy has the wrong length",
        ));
    }
    let copied_sha256: [u8; 32] = copied_hasher.finalize().into();
    if copied_sha256 != expected_sha256 {
        return Err(std::io::Error::other(
            "LQTP3 staging bytes changed after the writer computed the immutable bundle digest",
        ));
    }
    Ok(expected_length)
}

#[cfg(target_os = "linux")]
fn link_file_descriptor_noreplace(
    source: &File,
    destination_parent: &File,
    destination_name: &std::ffi::OsStr,
) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;

    let destination_name = CString::new(destination_name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "LQTP3 destination name contains a NUL byte",
        )
    })?;
    let empty_source = b"\0";
    // SAFETY: `empty_source` and `destination_name` are live NUL-terminated
    // strings; both descriptors are live for the syscall. AT_EMPTY_PATH makes
    // the retained source descriptor authoritative, and linkat is no-replace.
    let result = unsafe {
        libc::linkat(
            source.as_raw_fd(),
            empty_source.as_ptr().cast(),
            destination_parent.as_raw_fd(),
            destination_name.as_ptr(),
            libc::AT_EMPTY_PATH,
        )
    };
    if result == 0 {
        return Ok(());
    }
    let direct_error = std::io::Error::last_os_error();
    if !matches!(
        direct_error.raw_os_error(),
        Some(libc::ENOENT | libc::EINVAL | libc::ENOSYS)
    ) {
        return Err(direct_error);
    }

    // Some Linux configurations deny AT_EMPTY_PATH to unprivileged callers.
    // `/proc/self/fd/<n>` remains bound to this process's retained descriptor;
    // if procfs is unavailable, fail closed instead of trusting partial_path.
    let proc_source = CString::new(format!("/proc/self/fd/{}", source.as_raw_fd()))
        .expect("numeric file descriptor path cannot contain NUL");
    // SAFETY: both CString pointers and the destination descriptor remain live
    // for the syscall; AT_SYMLINK_FOLLOW resolves only the procfs fd link.
    let result = unsafe {
        libc::linkat(
            libc::AT_FDCWD,
            proc_source.as_ptr(),
            destination_parent.as_raw_fd(),
            destination_name.as_ptr(),
            libc::AT_SYMLINK_FOLLOW,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod publication_tests {
    use super::*;

    fn create_owned(path: &Path, bytes: &[u8]) -> (File, FileIdentity) {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        let identity = FileIdentity::from_file(&file).unwrap();
        (file, identity)
    }

    fn descriptor_sha256(file: &File) -> [u8; 32] {
        use std::os::unix::fs::FileExt;

        let length = usize::try_from(file.metadata().unwrap().len()).unwrap();
        let mut bytes = vec![0_u8; length];
        let mut offset = 0_usize;
        while offset < length {
            let count = file.read_at(&mut bytes[offset..], offset as u64).unwrap();
            assert_ne!(count, 0);
            offset += count;
        }
        sha256(&bytes)
    }

    fn bound_parent(final_path: &Path) -> (File, FileIdentity) {
        let parent_path = final_path.parent().unwrap();
        let parent = open_directory_nofollow(parent_path).unwrap();
        let identity = FileIdentity::from_file(&parent).unwrap();
        (parent, identity)
    }

    fn fail_retirement(_: &Path, _: FileIdentity, _: &File, _: bool) -> std::io::Result<()> {
        Err(std::io::Error::other("injected retirement failure"))
    }

    fn publish_owned_with_sync(
        partial_path: &Path,
        partial_identity: FileIdentity,
        partial_file: &File,
        final_path: &Path,
        sync_parent: impl FnOnce(&File) -> std::io::Result<()>,
    ) -> Result<(), PackV3Error> {
        let (parent, parent_identity) = bound_parent(final_path);
        let final_name = final_path.file_name().unwrap();
        create_staging_audit_link(partial_file, &parent, final_name).unwrap();
        parent.sync_all().unwrap();
        publish_noreplace_with_sync(
            partial_path,
            partial_identity,
            partial_file,
            descriptor_sha256(partial_file),
            PublicationTarget {
                parent: &parent,
                parent_identity,
                final_path,
            },
            sync_parent,
        )
    }

    fn raw_spec(chunk_rows: usize) -> ViewSpecV3 {
        ViewSpecV3::new(
            "signal",
            PackV3Dtype::F32,
            PackV3Encoding::Raw,
            &[1],
            true,
            [0; 32],
            chunk_rows,
            PackV3Compression::None,
            0,
        )
        .unwrap()
    }

    fn single_raw_spec() -> ViewSpecV3 {
        raw_spec(1)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn atomic_retirement_never_deletes_substitution_after_identity_check() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("owned.tmp");
        let (file, identity) = create_owned(&path, b"owned-data");
        let mut retired_path = None;

        retire_owned_file_with_hook(&path, identity, &file, true, |quarantine| {
            std::fs::remove_file(quarantine).unwrap();
            std::fs::write(quarantine, b"post-check-substitution").unwrap();
            retired_path = Some(quarantine.to_path_buf());
        })
        .unwrap();

        let retired_path = retired_path.unwrap();
        assert_eq!(
            std::fs::read(retired_path).unwrap(),
            b"post-check-substitution"
        );
        assert_eq!(file.metadata().unwrap().len(), 0);
    }

    #[test]
    fn sync_failure_retains_both_names_and_reports_do_not_retry_state() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let (partial_file, partial_identity) = create_owned(&partial, b"owned");

        let result = publish_owned_with_sync(
            &partial,
            partial_identity,
            &partial_file,
            &final_path,
            |_| Err(std::io::Error::other("injected sync failure")),
        );
        assert!(matches!(
            result,
            Err(PackV3Error::PublicationStateUnknown(_))
        ));
        assert!(partial.exists());
        assert!(final_path.exists());
        assert_eq!(std::fs::read(&partial).unwrap(), b"owned");
        assert_eq!(std::fs::read(&final_path).unwrap(), b"owned");
        let partial_metadata = std::fs::metadata(&partial).unwrap();
        let final_metadata = std::fs::metadata(&final_path).unwrap();
        assert_ne!(partial_metadata.ino(), final_metadata.ino());
        assert_eq!(partial_metadata.nlink(), 2);
        assert_eq!(final_metadata.nlink(), 1);
        let audit_entries: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(Result::unwrap)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .contains(".staging-audit.")
            })
            .collect();
        assert_eq!(audit_entries.len(), 1);
        assert_eq!(std::fs::read(audit_entries[0].path()).unwrap(), b"owned");
    }

    #[test]
    fn successful_publication_copies_to_a_single_link_final_inode() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let expected = b"owned-pack-bytes";
        let (partial_file, partial_identity) = create_owned(&partial, expected);

        publish_owned_with_sync(
            &partial,
            partial_identity,
            &partial_file,
            &final_path,
            File::sync_all,
        )
        .unwrap();

        let final_metadata = std::fs::metadata(&final_path).unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), expected);
        assert_eq!(final_metadata.nlink(), 1);
        assert_ne!(
            FileIdentity::from_file(&partial_file).unwrap().inode,
            final_metadata.ino()
        );
        assert_eq!(partial_file.metadata().unwrap().len(), 0);

        let partial_name = partial.file_name().unwrap().to_string_lossy();
        let tombstones: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(Result::unwrap)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with(partial_name.as_ref()) && name.contains(".retired.")
            })
            .collect();
        assert_eq!(tombstones.len(), 1);
        assert_eq!(tombstones[0].metadata().unwrap().len(), 0);
        for entry in std::fs::read_dir(directory.path()).unwrap() {
            let entry = entry.unwrap();
            if entry.path() != final_path {
                assert_eq!(entry.metadata().unwrap().len(), 0);
            }
        }
    }

    #[test]
    fn same_length_source_mutation_cannot_publish_a_torn_copy() {
        use std::os::unix::fs::FileExt;

        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source.partial");
        let mut expected = vec![0x5a_u8; 2 * 1024 * 1024];
        let final_index = expected.len() - 1;
        expected[final_index] = 0x7b;
        let (source, _) = create_owned(&source_path, &expected);
        let parent = open_directory_nofollow(directory.path()).unwrap();
        let destination = create_anonymous_file(&parent).unwrap();
        let expected_sha256 = sha256(&expected);
        let mutated = std::cell::Cell::new(false);

        let result = copy_exact_file_descriptor_with_hook(
            &source,
            &destination,
            expected_sha256,
            |copied| {
                if !mutated.get() && copied >= 1024 * 1024 {
                    source.write_at(&[0x33], (expected.len() - 1) as u64)?;
                    source.sync_all()?;
                    mutated.set(true);
                }
                Ok(())
            },
        );

        assert!(mutated.get());
        assert!(result.is_err());
        assert_eq!(source.metadata().unwrap().len(), expected.len() as u64);
    }

    #[test]
    fn retirement_failure_is_an_observable_do_not_retry_state() {
        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let expected = b"owned";
        let (partial_file, partial_identity) = create_owned(&partial, expected);
        let (parent, parent_identity) = bound_parent(&final_path);

        let result = publish_noreplace_with_hooks(
            &partial,
            partial_identity,
            &partial_file,
            sha256(expected),
            PublicationTarget {
                parent: &parent,
                parent_identity,
                final_path: &final_path,
            },
            PublicationHooks {
                sync_parent: File::sync_all,
                retire_staging: fail_retirement,
            },
        );

        assert!(matches!(
            result,
            Err(PackV3Error::PublicationStateUnknown(_))
        ));
        assert_eq!(std::fs::read(&final_path).unwrap(), expected);
        assert_eq!(std::fs::read(&partial).unwrap(), expected);
    }

    #[test]
    fn parent_change_after_final_sync_is_a_do_not_retry_state() {
        let directory = tempfile::tempdir().unwrap();
        let parent_path = directory.path().join("parent");
        let displaced_parent = directory.path().join("displaced-parent");
        let replacement_parent = directory.path().join("replacement-parent");
        std::fs::create_dir(&parent_path).unwrap();
        std::fs::create_dir(&replacement_parent).unwrap();
        let partial = parent_path.join("pack.partial");
        let final_path = parent_path.join("pack.lqtp3");
        let expected = b"owned";
        let (partial_file, partial_identity) = create_owned(&partial, expected);

        let result = publish_owned_with_sync(
            &partial,
            partial_identity,
            &partial_file,
            &final_path,
            |bound_parent| {
                bound_parent.sync_all()?;
                std::fs::rename(&parent_path, &displaced_parent)?;
                std::fs::rename(&replacement_parent, &parent_path)?;
                Ok(())
            },
        );

        assert!(matches!(
            result,
            Err(PackV3Error::PublicationStateUnknown(_))
        ));
        assert_eq!(
            std::fs::read(displaced_parent.join("pack.lqtp3")).unwrap(),
            expected
        );
        assert!(!parent_path.join("pack.lqtp3").exists());
    }

    #[test]
    fn unlink_substitution_and_sync_failure_retain_a_staging_audit_name() {
        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let expected = b"owned";
        let (partial_file, partial_identity) = create_owned(&partial, expected);
        let (parent, parent_identity) = bound_parent(&final_path);
        let audit_name =
            create_staging_audit_link(&partial_file, &parent, final_path.file_name().unwrap())
                .unwrap();
        parent.sync_all().unwrap();
        std::fs::remove_file(&partial).unwrap();
        std::fs::write(&partial, b"substitution").unwrap();

        let result = publish_noreplace_with_sync(
            &partial,
            partial_identity,
            &partial_file,
            sha256(expected),
            PublicationTarget {
                parent: &parent,
                parent_identity,
                final_path: &final_path,
            },
            |_| Err(std::io::Error::other("injected sync failure")),
        );

        assert!(matches!(
            result,
            Err(PackV3Error::PublicationStateUnknown(_))
        ));
        assert_eq!(std::fs::read(&partial).unwrap(), b"substitution");
        assert_eq!(
            std::fs::read(directory.path().join(audit_name)).unwrap(),
            expected
        );
        assert_eq!(std::fs::read(&final_path).unwrap(), expected);
    }

    #[test]
    fn sync_failure_never_deletes_a_substituted_destination() {
        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let (partial_file, partial_identity) = create_owned(&partial, b"owned");

        let result = publish_owned_with_sync(
            &partial,
            partial_identity,
            &partial_file,
            &final_path,
            |_| {
                std::fs::remove_file(&final_path).unwrap();
                std::fs::write(&final_path, b"substitution").unwrap();
                Err(std::io::Error::other("injected sync failure"))
            },
        );
        assert!(matches!(
            result,
            Err(PackV3Error::PublicationStateUnknown(_))
        ));
        assert_eq!(std::fs::read(&final_path).unwrap(), b"substitution");
        assert_eq!(std::fs::read(&partial).unwrap(), b"owned");
    }

    #[test]
    fn descriptor_publication_ignores_partial_substitution_before_link() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("pack.lqtp3");
        let displaced_partial = directory.path().join("displaced.partial");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        let partial = writer.partial_path.clone();
        let substituted = std::cell::Cell::new(false);

        writer
            .finish_with_io_hook(|stage| {
                if stage == FinishIoStageV3::PrePublish {
                    std::fs::rename(&partial, &displaced_partial)?;
                    std::fs::write(&partial, b"partial-substitution")?;
                    substituted.set(true);
                }
                Ok(())
            })
            .unwrap();

        assert!(substituted.get());
        assert_eq!(std::fs::read(&partial).unwrap(), b"partial-substitution");
        let reader = PackV3Reader::open(&final_path, None, None).unwrap();
        assert_eq!(reader.dequantize_f32("signal", 0).unwrap(), vec![7.0]);
    }

    #[test]
    fn descriptor_publication_ignores_unlink_substitution() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        let partial = writer.partial_path.clone();
        let observed_owned_file = std::cell::RefCell::new(None);

        writer
            .finish_with_io_hook(|stage| {
                if stage == FinishIoStageV3::PrePublish {
                    let owned = OpenOptions::new().read(true).write(true).open(&partial)?;
                    std::fs::remove_file(&partial)?;
                    std::fs::write(&partial, b"partial-substitution")?;
                    observed_owned_file.replace(Some(owned));
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(std::fs::read(&partial).unwrap(), b"partial-substitution");
        let reader = PackV3Reader::open(&final_path, None, None).unwrap();
        assert_eq!(reader.dequantize_f32("signal", 0).unwrap(), vec![7.0]);
        assert_eq!(
            observed_owned_file
                .into_inner()
                .unwrap()
                .metadata()
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn publication_rejects_a_symlinked_parent_component() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let real_parent = directory.path().join("real");
        let symlink_parent = directory.path().join("link");
        std::fs::create_dir(&real_parent).unwrap();
        symlink(&real_parent, &symlink_parent).unwrap();
        let final_path = symlink_parent.join("pack.lqtp3");
        let result = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        );

        assert!(matches!(result, Err(PackV3Error::Io(_))));
        assert!(!real_parent.join("pack.lqtp3").exists());
    }

    #[test]
    fn publication_rejects_a_parent_swapped_to_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let parent = directory.path().join("parent");
        let displaced_parent = directory.path().join("displaced-parent");
        let attacker_parent = directory.path().join("attacker-parent");
        std::fs::create_dir(&parent).unwrap();
        std::fs::create_dir(&attacker_parent).unwrap();
        let final_path = parent.join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        let swapped = std::cell::Cell::new(false);

        let result = writer.finish_with_io_hook(|stage| {
            if stage == FinishIoStageV3::PrePublish {
                std::fs::rename(&parent, &displaced_parent)?;
                symlink(&attacker_parent, &parent)?;
                swapped.set(true);
            }
            Ok(())
        });

        assert!(swapped.get());
        assert!(matches!(result, Err(PackV3Error::Io(_))));
        assert!(!attacker_parent.join("pack.lqtp3").exists());
        assert!(!displaced_parent.join("pack.lqtp3").exists());
    }

    #[test]
    fn publication_rejects_a_parent_swapped_to_another_directory() {
        let directory = tempfile::tempdir().unwrap();
        let parent = directory.path().join("parent");
        let displaced_parent = directory.path().join("displaced-parent");
        let replacement_parent = directory.path().join("replacement-parent");
        std::fs::create_dir(&parent).unwrap();
        std::fs::create_dir(&replacement_parent).unwrap();
        let final_path = parent.join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        let swapped = std::cell::Cell::new(false);

        let result = writer.finish_with_io_hook(|stage| {
            if stage == FinishIoStageV3::PrePublish {
                std::fs::rename(&parent, &displaced_parent)?;
                std::fs::rename(&replacement_parent, &parent)?;
                swapped.set(true);
            }
            Ok(())
        });

        assert!(swapped.get());
        assert!(matches!(result, Err(PackV3Error::Io(_))));
        assert!(!parent.join("pack.lqtp3").exists());
        assert!(!displaced_parent.join("pack.lqtp3").exists());
    }

    #[test]
    fn publication_walks_a_relative_parent_from_a_bound_cwd() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::Builder::new()
            .prefix("lqtp3-relative-parent-")
            .tempdir_in(".")
            .unwrap();
        let current_directory = std::env::current_dir().unwrap();
        let relative_parent = directory.path().strip_prefix(current_directory).unwrap();
        assert!(!relative_parent.is_absolute());
        let final_path = relative_parent.join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        writer.finish().unwrap();

        assert_eq!(std::fs::metadata(&final_path).unwrap().nlink(), 1);
        let reader = PackV3Reader::open(&final_path, None, None).unwrap();
        assert_eq!(reader.dequantize_f32("signal", 0).unwrap(), vec![7.0]);
    }

    #[test]
    fn publication_rejects_relative_cwd_change() {
        const CHILD_ENV: &str = "LAMQUANT_LQTP3_CWD_SWAP_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("tensor_pack_v3::publication_tests::publication_rejects_relative_cwd_change")
                .env(CHILD_ENV, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let original_cwd = std::env::current_dir().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let first_cwd = directory.path().join("first");
        let second_cwd = directory.path().join("second");
        std::fs::create_dir(&first_cwd).unwrap();
        std::fs::create_dir(&second_cwd).unwrap();
        std::fs::create_dir(first_cwd.join("output")).unwrap();
        std::fs::create_dir(second_cwd.join("output")).unwrap();
        std::env::set_current_dir(&first_cwd).unwrap();
        let final_path = Path::new("output").join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        std::env::set_current_dir(&second_cwd).unwrap();

        let result = writer.finish();
        std::env::set_current_dir(&original_cwd).unwrap();

        assert!(matches!(result, Err(PackV3Error::Io(_))));
        assert!(!first_cwd.join("output/pack.lqtp3").exists());
        assert!(!second_cwd.join("output/pack.lqtp3").exists());
    }

    #[test]
    fn early_io_failures_retain_descriptors_and_preserve_substitutes() {
        let stages = [
            FinishIoStageV3::ChunkWrite,
            FinishIoStageV3::ChunkFlush,
            FinishIoStageV3::ChunkSync,
            FinishIoStageV3::PartialWrite,
            FinishIoStageV3::PartialFlush,
            FinishIoStageV3::PartialSync,
            FinishIoStageV3::PrePublish,
        ];

        for stage in stages {
            let directory = tempfile::tempdir().unwrap();
            let final_path = directory.path().join("pack.lqtp3");
            let mut writer = PackV3Writer::create(
                &final_path,
                1,
                [0; 32],
                [0; 32],
                Vec::new(),
                vec![raw_spec(2)],
            )
            .unwrap();
            writer.write_f32_row("signal", &[7.0]).unwrap();
            let chunk = writer.views[0].temp_path.clone();
            let partial = writer.partial_path.clone();
            let target = if matches!(
                stage,
                FinishIoStageV3::ChunkWrite
                    | FinishIoStageV3::ChunkFlush
                    | FinishIoStageV3::ChunkSync
            ) {
                chunk
            } else {
                partial
            };
            let substitute = format!("{stage:?}-substitution").into_bytes();
            let observed_owned_file = std::cell::RefCell::new(None);
            let injected = std::cell::Cell::new(false);

            let result = writer.finish_with_io_hook(|observed| {
                if observed != stage {
                    return Ok(());
                }
                let owned = OpenOptions::new().read(true).write(true).open(&target)?;
                std::fs::remove_file(&target)?;
                std::fs::write(&target, &substitute)?;
                observed_owned_file.replace(Some(owned));
                injected.set(true);
                Err(std::io::Error::other("injected finish I/O failure"))
            });

            assert!(injected.get(), "failpoint {stage:?} was not reached");
            assert!(matches!(result, Err(PackV3Error::Io(_))));
            assert!(!final_path.exists());
            assert_eq!(std::fs::read(&target).unwrap(), substitute);
            assert_eq!(
                observed_owned_file
                    .into_inner()
                    .unwrap()
                    .metadata()
                    .unwrap()
                    .len(),
                0,
                "owned storage was not truncated after {stage:?} failure"
            );
        }
    }

    #[test]
    fn unknown_publication_state_disables_drop_path_cleanup() {
        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let (partial_file, partial_identity) = create_owned(&partial, b"owned-partial");
        let (publication_parent, publication_parent_identity) = bound_parent(&final_path);
        std::fs::write(&final_path, b"audit-final").unwrap();

        let writer = PackV3Writer {
            final_path: final_path.clone(),
            publication_parent,
            publication_parent_identity,
            partial_path: partial.clone(),
            partial_identity: Some(partial_identity),
            partial_file: Some(partial_file),
            row_count: 1,
            manifest_sha256: [0; 32],
            view_spec_sha256: [0; 32],
            metadata: Vec::new(),
            views: Vec::new(),
            state: PackV3WriterState::PublicationStateUnknown,
        };
        drop(writer);

        assert_eq!(std::fs::read(&partial).unwrap(), b"owned-partial");
        assert_eq!(std::fs::read(&final_path).unwrap(), b"audit-final");
    }

    #[test]
    fn successful_sync_cleanup_never_unlinks_a_substituted_partial() {
        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("pack.partial");
        let final_path = directory.path().join("pack.lqtp3");
        let (partial_file, partial_identity) = create_owned(&partial, b"owned");

        publish_owned_with_sync(
            &partial,
            partial_identity,
            &partial_file,
            &final_path,
            |_| {
                std::fs::remove_file(&partial).unwrap();
                std::fs::write(&partial, b"partial-substitution").unwrap();
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(std::fs::read(&final_path).unwrap(), b"owned");
        assert_eq!(std::fs::read(&partial).unwrap(), b"partial-substitution");
    }

    #[test]
    fn active_drop_never_unlinks_substituted_partial_or_chunk_paths() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        let partial = writer.partial_path.clone();
        let chunk = writer.views[0].temp_path.clone();
        let (partial_file, partial_identity) = create_owned(&partial, b"owned-partial");
        writer.partial_identity = Some(partial_identity);
        writer.partial_file = Some(partial_file);

        std::fs::remove_file(&partial).unwrap();
        std::fs::write(&partial, b"partial-substitution").unwrap();
        std::fs::remove_file(&chunk).unwrap();
        std::fs::write(&chunk, b"chunk-substitution").unwrap();
        drop(writer);

        assert_eq!(std::fs::read(&partial).unwrap(), b"partial-substitution");
        assert_eq!(std::fs::read(&chunk).unwrap(), b"chunk-substitution");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn active_drop_discards_buffer_before_truncating_chunk_tombstone() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        let chunk = writer.views[0].temp_path.clone();

        drop(writer);

        assert!(!chunk.exists());
        let chunk_name = chunk.file_name().unwrap().to_string_lossy();
        let tombstones: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(Result::unwrap)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with(chunk_name.as_ref()) && name.contains(".retired.")
            })
            .collect();
        assert_eq!(tombstones.len(), 1);
        assert_eq!(tombstones[0].metadata().unwrap().len(), 0);
    }

    #[test]
    fn successful_finish_never_unlinks_a_substituted_chunk_path() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("pack.lqtp3");
        let mut writer = PackV3Writer::create(
            &final_path,
            1,
            [0; 32],
            [0; 32],
            Vec::new(),
            vec![single_raw_spec()],
        )
        .unwrap();
        writer.write_f32_row("signal", &[7.0]).unwrap();
        let chunk = writer.views[0].temp_path.clone();
        std::fs::remove_file(&chunk).unwrap();
        std::fs::write(&chunk, b"chunk-substitution").unwrap();

        writer.finish().unwrap();

        assert_eq!(std::fs::read(&chunk).unwrap(), b"chunk-substitution");
        let reader = PackV3Reader::open(&final_path, None, None).unwrap();
        assert_eq!(reader.dequantize_f32("signal", 0).unwrap(), vec![7.0]);
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    finalize_hash(Sha256::digest(bytes))
}
fn finalize_hash(hash: impl AsRef<[u8]>) -> [u8; 32] {
    let mut output = [0_u8; 32];
    output.copy_from_slice(hash.as_ref());
    output
}
fn hash_at(bytes: &[u8], offset: usize) -> Result<[u8; 32], PackV3Error> {
    let value = bytes
        .get(offset..offset + 32)
        .ok_or(PackV3Error::Truncated {
            expected: offset + 32,
            actual: bytes.len(),
        })?;
    let mut output = [0_u8; 32];
    output.copy_from_slice(value);
    Ok(output)
}
fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, PackV3Error> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(PackV3Error::Truncated {
            expected: offset + 2,
            actual: bytes.len(),
        })?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PackV3Error> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(PackV3Error::Truncated {
            expected: offset + 4,
            actual: bytes.len(),
        })?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}
fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, PackV3Error> {
    Ok(read_u32(bytes, offset)? as i32)
}
fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, PackV3Error> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(PackV3Error::Truncated {
            expected: offset + 8,
            actual: bytes.len(),
        })?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("eight-byte slice"),
    ))
}
