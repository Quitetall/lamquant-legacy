//! Experimental ADR 0140 bridge between LamQuant's BCS1-coupled ABIR and the
//! independent semantic ABIR core.
//!
//! The bridge is intentionally opt-in. Forward conversion borrows legacy
//! column buffers through [`LegacyPayloadAccess`] and therefore does not copy
//! samples. Reverse conversion is fail-closed and accepts only the uniform,
//! single-recording subset that the legacy representation can express.
#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;

use legacy_ir as legacy;
use semantic::{PayloadAccess, PayloadLease};
use semantic_abir as semantic;

const KEY_LABEL: &str = "legacy.channel-label";
const KEY_PHYS_MIN: &str = "legacy.phys-min";
const KEY_PHYS_MAX: &str = "legacy.phys-max";
const KEY_STORAGE: &str = "legacy.storage-width";

/// How one legacy channel was represented in semantic ABIR.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelMapping {
    pub index: usize,
    pub label: String,
    pub storage_width: &'static str,
    pub atom_id: semantic::ObjectId<semantic::AtomTag>,
    pub content_id: semantic::ContentId,
    pub zero_copy: bool,
}

/// One exact event interval to add while constructing the semantic root.
///
/// Times are expressed in seconds relative to the recording origin. Callers
/// must preserve standard-specific fields that do not fit this bounded shape
/// in source capsules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimedEventMapping {
    pub kind: semantic::ConceptId,
    pub start: semantic::Rational,
    pub end: semantic::Rational,
    pub uncertainty: semantic::Rational,
}

/// Auditable identity assigned to one event overlay entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventMapping {
    pub index: usize,
    pub kind: semantic::ConceptId,
    pub event_id: semantic::ObjectId<semantic::EventTag>,
}

/// Auditable description of a forward bridge operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingReport {
    pub source_modality: &'static str,
    pub recording_count: usize,
    pub stream_count: usize,
    pub channels: Vec<ChannelMapping>,
    pub events: Vec<EventMapping>,
}

/// Fidelity claims that the bridge can honestly make.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FidelityReport {
    pub sample_values_exact: bool,
    pub sample_rate_exact: bool,
    pub channel_order_exact: bool,
    pub labels_exact: bool,
    pub physical_ranges_preserved_as_source_keys: bool,
    pub calibration_promoted: bool,
}

/// Result of mapping a legacy recording without taking ownership of its payloads.
#[derive(Debug)]
pub struct LegacyMapped<'a, M: legacy::Modality> {
    pub dataset: semantic::AbirDataset,
    pub access: LegacyPayloadAccess<'a, M>,
    pub mapping: MappingReport,
    pub fidelity: FidelityReport,
}

/// Exact foreign source material that must remain bound to the semantic root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceCapsuleMapping {
    pub namespace: String,
    pub value: String,
    pub content_id: semantic::ContentId,
    pub media_type: Option<String>,
}

/// Failures are structured and conversion never silently drops semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BridgeError {
    InvalidLegacy(String),
    InvalidSemantic(String),
    UnsupportedHostByteOrder,
    InvalidSampleRate,
    InvalidPhysicalRange {
        channel: usize,
    },
    Identifier(String),
    Payload(semantic::PayloadAccessError),
    Unrepresentable(&'static str),
    PayloadDecode {
        channel: usize,
        reason: &'static str,
    },
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLegacy(reason) => write!(f, "invalid legacy ABIR: {reason}"),
            Self::InvalidSemantic(reason) => write!(f, "invalid semantic ABIR: {reason}"),
            Self::UnsupportedHostByteOrder => {
                f.write_str("zero-copy bridge requires a little-endian host")
            }
            Self::InvalidSampleRate => f.write_str("sample rate is not a positive exact rational"),
            Self::InvalidPhysicalRange { channel } => {
                write!(f, "channel {channel} has a non-finite physical range")
            }
            Self::Identifier(reason) => write!(f, "invalid semantic identifier: {reason}"),
            Self::Payload(error) => error.fmt(f),
            Self::Unrepresentable(reason) => write!(f, "legacy ABIR cannot represent: {reason}"),
            Self::PayloadDecode { channel, reason } => {
                write!(f, "channel {channel} payload cannot be decoded: {reason}")
            }
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<semantic::PayloadAccessError> for BridgeError {
    fn from(value: semantic::PayloadAccessError) -> Self {
        Self::Payload(value)
    }
}

/// Payload resolver that lends the original legacy column storage.
#[derive(Clone, Copy, Debug)]
pub struct LegacyPayloadAccess<'a, M: legacy::Modality> {
    source: &'a legacy::LegacyRecording<M>,
}

impl<'a, M: legacy::Modality> LegacyPayloadAccess<'a, M> {
    pub const fn new(source: &'a legacy::LegacyRecording<M>) -> Self {
        Self { source }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LegacyLease<'a>(&'a [u8]);

impl PayloadLease for LegacyLease<'_> {
    fn bytes(&self) -> &[u8] {
        self.0
    }
}

impl<M: legacy::Modality> PayloadAccess for LegacyPayloadAccess<'_, M> {
    type Lease<'a>
        = LegacyLease<'a>
    where
        Self: 'a;

    fn lease<'a>(
        &'a self,
        descriptor: &semantic::PayloadDescriptor,
    ) -> Result<Self::Lease<'a>, semantic::PayloadAccessError> {
        for channel in &self.source.channels {
            let element = semantic_element(&channel.data);
            let bytes = column_bytes(&channel.data);
            if content_id(element, bytes) == descriptor.content_id() {
                if bytes.len() != descriptor.logical_bytes() as usize {
                    return Err(semantic::PayloadAccessError::LengthMismatch {
                        expected: descriptor.logical_bytes(),
                        actual: bytes.len(),
                    });
                }
                return Ok(LegacyLease(bytes));
            }
        }
        Err(semantic::PayloadAccessError::NotFound(
            descriptor.content_id(),
        ))
    }
}

/// Convert legacy ABIR semantics while borrowing every sample buffer.
pub fn from_legacy<M: legacy::Modality>(
    source: &legacy::LegacyRecording<M>,
) -> Result<LegacyMapped<'_, M>, BridgeError> {
    from_legacy_with_source_capsules(source, &[])
}

/// Convert legacy semantics while binding exact foreign source capsules into
/// the same immutable dataset identity.
pub fn from_legacy_with_source_capsules<'a, M: legacy::Modality>(
    source: &'a legacy::LegacyRecording<M>,
    source_capsules: &[SourceCapsuleMapping],
) -> Result<LegacyMapped<'a, M>, BridgeError> {
    from_legacy_with_source_capsules_and_limits(
        source,
        source_capsules,
        semantic::ValidationLimits::default(),
    )
}

/// Convert with caller-supplied structural limits for untrusted imports.
pub fn from_legacy_with_source_capsules_and_limits<'a, M: legacy::Modality>(
    source: &'a legacy::LegacyRecording<M>,
    source_capsules: &[SourceCapsuleMapping],
    limits: semantic::ValidationLimits,
) -> Result<LegacyMapped<'a, M>, BridgeError> {
    from_legacy_with_source_capsules_events_and_limits(source, source_capsules, &[], limits)
}

/// Convert with exact source capsules and a bounded event overlay.
///
/// The overlay creates one recording-relative seconds clock shared by the
/// signal stream and all events. Structural validation rejects reversed event
/// intervals, negative uncertainty, and caller-limit violations before a root
/// is exposed.
pub fn from_legacy_with_source_capsules_events_and_limits<'a, M: legacy::Modality>(
    source: &'a legacy::LegacyRecording<M>,
    source_capsules: &[SourceCapsuleMapping],
    events: &[TimedEventMapping],
    limits: semantic::ValidationLimits,
) -> Result<LegacyMapped<'a, M>, BridgeError> {
    require_little_endian()?;
    if events.len() > limits.max_catalog_records {
        return Err(BridgeError::InvalidSemantic(
            "event overlay exceeds max_catalog_records".to_owned(),
        ));
    }
    source
        .verify()
        .map_err(|error| BridgeError::InvalidLegacy(error.to_string()))?;
    let rate = exact_positive_f64(source.sample_rate)?;
    let samples = u64::try_from(source.n_samples)
        .map_err(|_| BridgeError::Unrepresentable("sample count exceeds u64"))?;
    if samples == 0 {
        return Err(BridgeError::Unrepresentable("empty recordings"));
    }

    let seed = dataset_seed_with_events(source, events);
    let dataset_id = object_id(&seed, b"dataset", 0);
    let recording_id = object_id(&seed, b"recording", 0);
    let stream_id = object_id(&seed, b"stream", 0);
    let basis_id = object_id(&seed, b"channel-basis", 0);
    let mut atom_ids = Vec::with_capacity(source.channels.len());
    let mut atoms = Vec::with_capacity(source.channels.len());
    let mut specs = Vec::with_capacity(source.channels.len());
    let mut mappings = Vec::with_capacity(source.channels.len());

    for (index, channel) in source.channels.iter().enumerate() {
        if !channel.phys_min.is_finite() || !channel.phys_max.is_finite() {
            return Err(BridgeError::InvalidPhysicalRange { channel: index });
        }
        if let legacy::Column::I24(values) = &channel.data {
            if values
                .iter()
                .any(|value| !(-8_388_608..=8_388_607).contains(value))
            {
                return Err(BridgeError::PayloadDecode {
                    channel: index,
                    reason: "i24 value is outside the signed 24-bit range",
                });
            }
        }
        let atom_id = object_id(&seed, b"atom", index as u64);
        let element = semantic_element(&channel.data);
        let bytes = column_bytes(&channel.data);
        let content_id = content_id(element, bytes);
        let descriptor = semantic::PayloadDescriptor::new(
            content_id,
            bytes.len() as u64,
            element,
            semantic::ByteOrder::Little,
            vec![1, samples],
            semantic::Layout::DenseRowMajor,
            None,
            None,
        );
        let segment = semantic::TimeSegment::new(
            semantic::Rational::new(0, 1).expect("zero is canonical"),
            rate,
            samples,
        )
        .map_err(|error| BridgeError::InvalidSemantic(error.to_string()))?;
        atoms.push(semantic::Atom::SignalBlock(semantic::SignalBlock::new(
            atom_id,
            semantic::Presence::Present,
            Some(descriptor),
            semantic::TimeAxis::Regular(segment),
            None,
        )));
        atom_ids.push(atom_id);

        let concept = semantic::ConceptId::new(format!("legacy:channel/{index}"))
            .map_err(|error| BridgeError::Identifier(error.to_string()))?;
        let spec = semantic::ChannelSpec::new(concept)
            .with_source_key(source_key(KEY_LABEL, &channel.label)?)
            .with_source_key(source_key(KEY_PHYS_MIN, &channel.phys_min.to_string())?)
            .with_source_key(source_key(KEY_PHYS_MAX, &channel.phys_max.to_string())?)
            .with_source_key(source_key(KEY_STORAGE, storage_width(&channel.data))?);
        specs.push(spec);
        mappings.push(ChannelMapping {
            index,
            label: channel.label.to_string(),
            storage_width: storage_width(&channel.data),
            atom_id,
            content_id,
            zero_copy: true,
        });
    }

    let modality = semantic::ConceptId::new(format!("abir:modality/{}", M::NAME))
        .map_err(|error| BridgeError::Identifier(error.to_string()))?;
    let mut draft = semantic::DatasetDraft::new(dataset_id);
    draft.add_recording(semantic::Recording::new(recording_id, vec![stream_id]));
    let event_clock_id = (!events.is_empty()).then(|| object_id(&seed, b"event-clock", 0));
    draft.add_stream(semantic::Stream::new(
        stream_id,
        recording_id,
        modality,
        atom_ids,
        event_clock_id,
        Some(basis_id),
        None,
    ));
    draft.add_channel_basis(semantic::ChannelBasis::new(
        basis_id,
        specs,
        semantic::ReferenceKind::Unknown,
    ));
    for atom in atoms {
        draft.add_atom(atom);
    }
    for capsule in source_capsules {
        let source_key = semantic::SourceKey::new(&capsule.namespace, &capsule.value)
            .map_err(|error| BridgeError::Identifier(error.to_string()))?;
        draft.add_source_capsule(semantic::SourceCapsule::new(
            source_key,
            capsule.content_id,
            capsule.media_type.as_deref(),
        ));
    }
    let mut event_mappings = Vec::with_capacity(events.len());
    if let Some(clock_id) = event_clock_id {
        draft.add_clock(semantic::Clock::new(
            clock_id,
            semantic::ConceptId::new("abir:clock/recording-relative-seconds")
                .expect("static concept is canonical"),
            None,
            semantic::Rational::new(0, 1).expect("zero is canonical"),
            semantic::Rational::new(1, 1).expect("one is canonical"),
            semantic::Rational::new(0, 1).expect("zero is canonical"),
        ));
        for (index, event) in events.iter().enumerate() {
            let event_id = object_id(&seed, b"event", index as u64);
            draft.add_event(semantic::Event::new(
                event_id,
                event.kind.clone(),
                clock_id,
                event.start,
                event.end,
                event.uncertainty,
            ));
            event_mappings.push(EventMapping {
                index,
                kind: event.kind.clone(),
                event_id,
            });
        }
    }
    let dataset = draft
        .validate(limits)
        .map_err(|report| BridgeError::InvalidSemantic(format!("{report:?}")))?;

    Ok(LegacyMapped {
        dataset,
        access: LegacyPayloadAccess::new(source),
        mapping: MappingReport {
            source_modality: M::NAME,
            recording_count: 1,
            stream_count: 1,
            channels: mappings,
            events: event_mappings,
        },
        fidelity: FidelityReport {
            sample_values_exact: true,
            sample_rate_exact: true,
            channel_order_exact: true,
            labels_exact: true,
            physical_ranges_preserved_as_source_keys: true,
            calibration_promoted: false,
        },
    })
}

/// Convert the representable semantic subset back to legacy ABIR.
pub fn to_legacy<M: legacy::Modality, A: PayloadAccess>(
    dataset: &semantic::AbirDataset,
    access: &A,
) -> Result<legacy::LegacyRecording<M>, BridgeError> {
    require_little_endian()?;
    ensure_legacy_subset(dataset)?;
    let recording = &dataset.recordings()[0];
    let stream = &dataset.streams()[0];
    if recording.streams() != [stream.id()] || stream.recording_id() != recording.id() {
        return Err(BridgeError::Unrepresentable(
            "noncanonical recording membership",
        ));
    }
    let expected_modality = format!("abir:modality/{}", M::NAME);
    if stream.modality().as_str() != expected_modality {
        return Err(BridgeError::Unrepresentable("stream modality mismatch"));
    }
    let basis_id = stream
        .channel_basis_id()
        .ok_or(BridgeError::Unrepresentable(
            "stream without a channel basis",
        ))?;
    let basis = dataset
        .channel_bases()
        .iter()
        .find(|candidate| candidate.id() == basis_id)
        .ok_or(BridgeError::Unrepresentable("unresolved channel basis"))?;
    if basis.reference() != semantic::ReferenceKind::Unknown
        || basis.channels().len() != stream.atoms().len()
    {
        return Err(BridgeError::Unrepresentable(
            "reference semantics or channel cardinality",
        ));
    }

    let mut channels = Vec::with_capacity(stream.atoms().len());
    let mut common_rate = None;
    let mut common_samples = None;
    for (index, (atom_id, spec)) in stream.atoms().iter().zip(basis.channels()).enumerate() {
        let expected_concept = format!("legacy:channel/{index}");
        if spec.coordinate_frame_id().is_some()
            || spec.concept().as_str() != expected_concept
            || spec.source_keys().len() != 4
        {
            return Err(BridgeError::Unrepresentable(
                "channel concepts, coordinate frames, or extra source keys",
            ));
        }
        let atom = dataset
            .atoms()
            .iter()
            .find(|candidate| candidate.id() == *atom_id)
            .ok_or(BridgeError::Unrepresentable("unresolved stream atom"))?;
        let block = match atom {
            semantic::Atom::SignalBlock(block) => block,
            _ => return Err(BridgeError::Unrepresentable("non-signal atoms")),
        };
        if atom.presence() != semantic::Presence::Present || block.calibration().is_some() {
            return Err(BridgeError::Unrepresentable(
                "presence states or semantic calibration",
            ));
        }
        let segment = match block.time_axis() {
            semantic::TimeAxis::Regular(segment) => *segment,
            _ => return Err(BridgeError::Unrepresentable("nonuniform time axes")),
        };
        if segment.start() != semantic::Rational::new(0, 1).expect("zero is canonical") {
            return Err(BridgeError::Unrepresentable("nonzero time origins"));
        }
        match common_rate {
            Some(rate) if rate != segment.rate() => {
                return Err(BridgeError::Unrepresentable("mixed sample rates"));
            }
            None => common_rate = Some(segment.rate()),
            _ => {}
        }
        match common_samples {
            Some(samples) if samples != segment.samples() => {
                return Err(BridgeError::Unrepresentable("mixed channel lengths"));
            }
            None => common_samples = Some(segment.samples()),
            _ => {}
        }
        let descriptor = atom
            .payload()
            .ok_or(BridgeError::Unrepresentable("missing present payload"))?;
        if descriptor.shape() != [1, segment.samples()]
            || descriptor.layout() != &semantic::Layout::DenseRowMajor
            || descriptor.byte_order() != semantic::ByteOrder::Little
            || descriptor.encoding().is_some()
            || descriptor.media_type().is_some()
        {
            return Err(BridgeError::Unrepresentable("payload layout or encoding"));
        }
        let label = required_key(spec, KEY_LABEL)?;
        let phys_min = parse_finite(required_key(spec, KEY_PHYS_MIN)?, index)?;
        let phys_max = parse_finite(required_key(spec, KEY_PHYS_MAX)?, index)?;
        let storage = required_key(spec, KEY_STORAGE)?;
        let lease = access.lease(descriptor)?;
        let data = decode_column(index, storage, descriptor.element(), lease.bytes())?;
        channels.push(legacy::Channel {
            label: Arc::from(label),
            data,
            phys_min,
            phys_max,
        });
    }

    let rate = common_rate.ok_or(BridgeError::Unrepresentable("empty streams"))?;
    let samples = common_samples.ok_or(BridgeError::Unrepresentable("empty streams"))?;
    let sample_rate = rational_to_f64_exact(rate)?;
    let n_samples = usize::try_from(samples)
        .map_err(|_| BridgeError::Unrepresentable("sample count exceeds usize"))?;
    let typed = legacy::LegacyRecording::from_parts(channels, sample_rate, n_samples)
        .into_modality::<M>(legacy::ModalitySource::Manual);
    typed
        .verify()
        .map_err(|error| BridgeError::InvalidLegacy(error.to_string()))?;
    Ok(typed)
}

fn ensure_legacy_subset(dataset: &semantic::AbirDataset) -> Result<(), BridgeError> {
    if dataset.recordings().len() != 1 || dataset.streams().len() != 1 {
        return Err(BridgeError::Unrepresentable(
            "anything other than one recording and one stream",
        ));
    }
    if dataset.channel_bases().len() != 1
        || !dataset.clocks().is_empty()
        || !dataset.coordinate_frames().is_empty()
        || !dataset.policies().is_empty()
        || !dataset.proofs().is_empty()
        || !dataset.derivations().is_empty()
        || !dataset.fidelity().is_empty()
        || !dataset.source_capsules().is_empty()
        || !dataset.observed_execution().is_empty()
        || !dataset.recordings()[0].source_keys().is_empty()
        || dataset.streams()[0].clock_id().is_some()
        || dataset.streams()[0].policy_id().is_some()
    {
        return Err(BridgeError::Unrepresentable("additional semantic metadata"));
    }
    let mut stream_atom_ids = dataset.streams()[0].atoms().to_vec();
    stream_atom_ids.sort_unstable();
    stream_atom_ids.dedup();
    if stream_atom_ids.len() != dataset.atoms().len()
        || dataset
            .atoms()
            .iter()
            .any(|atom| stream_atom_ids.binary_search(&atom.id()).is_err())
    {
        return Err(BridgeError::Unrepresentable(
            "unreferenced or repeated atoms",
        ));
    }
    Ok(())
}

fn required_key<'a>(
    spec: &'a semantic::ChannelSpec,
    namespace: &'static str,
) -> Result<&'a str, BridgeError> {
    let mut values = spec
        .source_keys()
        .iter()
        .filter(|key| key.namespace() == namespace)
        .map(semantic::SourceKey::value);
    let value = values.next().ok_or(BridgeError::Unrepresentable(
        "missing legacy channel source key",
    ))?;
    if values.next().is_some() {
        return Err(BridgeError::Unrepresentable(
            "duplicate legacy channel source key",
        ));
    }
    Ok(value)
}

fn decode_column(
    channel: usize,
    storage: &str,
    element: semantic::ElementType,
    bytes: &[u8],
) -> Result<legacy::Column, BridgeError> {
    let mismatch = || BridgeError::PayloadDecode {
        channel,
        reason: "storage width, element type, or byte length mismatch",
    };
    let column = match (storage, element) {
        ("i16", semantic::ElementType::I16) => {
            let values = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    Ok(i16::from_le_bytes(
                        chunk.try_into().map_err(|_| mismatch())?,
                    ))
                })
                .collect::<Result<Vec<_>, BridgeError>>()?;
            if bytes.len() % 2 != 0 {
                return Err(mismatch());
            }
            legacy::Column::I16(Arc::from(values))
        }
        ("i24", semantic::ElementType::I32) | ("i32", semantic::ElementType::I32) => {
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    Ok(i32::from_le_bytes(
                        chunk.try_into().map_err(|_| mismatch())?,
                    ))
                })
                .collect::<Result<Vec<_>, BridgeError>>()?;
            if bytes.len() % 4 != 0 {
                return Err(mismatch());
            }
            if storage == "i24"
                && values
                    .iter()
                    .any(|value| !(-8_388_608..=8_388_607).contains(value))
            {
                return Err(BridgeError::PayloadDecode {
                    channel,
                    reason: "i24 value is outside the signed 24-bit range",
                });
            }
            if storage == "i24" {
                legacy::Column::I24(Arc::from(values))
            } else {
                legacy::Column::I32(Arc::from(values))
            }
        }
        ("i64", semantic::ElementType::I64) => {
            let values = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    Ok(i64::from_le_bytes(
                        chunk.try_into().map_err(|_| mismatch())?,
                    ))
                })
                .collect::<Result<Vec<_>, BridgeError>>()?;
            if bytes.len() % 8 != 0 {
                return Err(mismatch());
            }
            legacy::Column::I64(Arc::from(values))
        }
        ("f32", semantic::ElementType::F32) => {
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    Ok(f32::from_le_bytes(
                        chunk.try_into().map_err(|_| mismatch())?,
                    ))
                })
                .collect::<Result<Vec<_>, BridgeError>>()?;
            if bytes.len() % 4 != 0 {
                return Err(mismatch());
            }
            legacy::Column::F32(Arc::from(values))
        }
        _ => return Err(mismatch()),
    };
    Ok(column)
}

fn parse_finite(value: &str, channel: usize) -> Result<f64, BridgeError> {
    let value = value
        .parse::<f64>()
        .map_err(|_| BridgeError::InvalidPhysicalRange { channel })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BridgeError::InvalidPhysicalRange { channel })
    }
}

fn source_key(namespace: &str, value: &str) -> Result<semantic::SourceKey, BridgeError> {
    semantic::SourceKey::new(namespace, value)
        .map_err(|error| BridgeError::Identifier(error.to_string()))
}

fn semantic_element(column: &legacy::Column) -> semantic::ElementType {
    match column {
        legacy::Column::I16(_) => semantic::ElementType::I16,
        legacy::Column::I24(_) | legacy::Column::I32(_) => semantic::ElementType::I32,
        legacy::Column::I64(_) => semantic::ElementType::I64,
        legacy::Column::F32(_) => semantic::ElementType::F32,
    }
}

fn storage_width(column: &legacy::Column) -> &'static str {
    match column {
        legacy::Column::I16(_) => "i16",
        legacy::Column::I24(_) => "i24",
        legacy::Column::I32(_) => "i32",
        legacy::Column::I64(_) => "i64",
        legacy::Column::F32(_) => "f32",
    }
}

fn column_bytes(column: &legacy::Column) -> &[u8] {
    match column {
        legacy::Column::I16(values) => bytemuck::cast_slice(values),
        legacy::Column::I24(values) | legacy::Column::I32(values) => bytemuck::cast_slice(values),
        legacy::Column::I64(values) => bytemuck::cast_slice(values),
        legacy::Column::F32(values) => bytemuck::cast_slice(values),
    }
}

fn content_id(element: semantic::ElementType, bytes: &[u8]) -> semantic::ContentId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"abir.semantic-v1.payload\0");
    hasher.update(element_tag(element));
    hasher.update(&[0]);
    hasher.update(bytes);
    semantic::ContentId::from_bytes(*hasher.finalize().as_bytes())
}

fn element_tag(element: semantic::ElementType) -> &'static [u8] {
    match element {
        semantic::ElementType::I8 => b"i8",
        semantic::ElementType::I16 => b"i16",
        semantic::ElementType::I24 => b"i24",
        semantic::ElementType::I32 => b"i32",
        semantic::ElementType::I64 => b"i64",
        semantic::ElementType::U8 => b"u8",
        semantic::ElementType::U16 => b"u16",
        semantic::ElementType::U32 => b"u32",
        semantic::ElementType::U64 => b"u64",
        semantic::ElementType::F16 => b"f16",
        semantic::ElementType::F32 => b"f32",
        semantic::ElementType::F64 => b"f64",
        semantic::ElementType::Bool => b"bool",
        semantic::ElementType::Utf8 => b"utf8",
        semantic::ElementType::Bytes => b"bytes",
    }
}

fn dataset_seed<M: legacy::Modality>(source: &legacy::LegacyRecording<M>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lamquant.legacy-abir-bridge.dataset-v1\0");
    hasher.update(M::NAME.as_bytes());
    hasher.update(&source.sample_rate.to_bits().to_le_bytes());
    hasher.update(&(source.n_samples as u128).to_le_bytes());
    for channel in &source.channels {
        hasher.update(&(channel.label.len() as u64).to_le_bytes());
        hasher.update(channel.label.as_bytes());
        hasher.update(storage_width(&channel.data).as_bytes());
        hasher.update(
            content_id(semantic_element(&channel.data), column_bytes(&channel.data)).as_bytes(),
        );
        hasher.update(&channel.phys_min.to_bits().to_le_bytes());
        hasher.update(&channel.phys_max.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn dataset_seed_with_events<M: legacy::Modality>(
    source: &legacy::LegacyRecording<M>,
    events: &[TimedEventMapping],
) -> [u8; 32] {
    let base = dataset_seed(source);
    if events.is_empty() {
        return base;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lamquant.legacy-abir-bridge.events-v1\0");
    hasher.update(&base);
    hasher.update(&(events.len() as u64).to_le_bytes());
    for event in events {
        hasher.update(&(event.kind.as_str().len() as u64).to_le_bytes());
        hasher.update(event.kind.as_str().as_bytes());
        for value in [event.start, event.end, event.uncertainty] {
            let (numerator, denominator) = value.parts();
            hasher.update(&numerator.to_le_bytes());
            hasher.update(&denominator.to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

fn object_id<T>(seed: &[u8; 32], domain: &[u8], index: u64) -> semantic::ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"abir.semantic-v1.object-id\0");
    hasher.update(seed);
    hasher.update(domain);
    hasher.update(&index.to_le_bytes());
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    semantic::ObjectId::from_bytes(bytes)
}

fn require_little_endian() -> Result<(), BridgeError> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err(BridgeError::UnsupportedHostByteOrder)
    }
}

fn exact_positive_f64(value: f64) -> Result<semantic::Rational, BridgeError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(BridgeError::InvalidSampleRate);
    }
    // Decompose the IEEE-754 value into its integer significand and base-two
    // exponent, cancel powers of two, then construct the same value as a
    // reduced rational without a decimal or lossy floating-point round trip.
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mut numerator, mut power) = if exponent == 0 {
        (u128::from(fraction), -1074_i32)
    } else {
        (u128::from(fraction | (1_u64 << 52)), exponent - 1023 - 52)
    };
    if numerator == 0 {
        return Err(BridgeError::InvalidSampleRate);
    }
    if power < 0 {
        let reduce = numerator.trailing_zeros().min((-power) as u32);
        numerator >>= reduce;
        power += reduce as i32;
    }
    let (numerator, denominator) = if power >= 0 {
        let value = numerator
            .checked_shl(power as u32)
            .and_then(|value| i128::try_from(value).ok())
            .ok_or(BridgeError::InvalidSampleRate)?;
        (value, 1_i128)
    } else {
        let shift = (-power) as u32;
        if shift > 126 {
            return Err(BridgeError::InvalidSampleRate);
        }
        let numerator = i128::try_from(numerator).map_err(|_| BridgeError::InvalidSampleRate)?;
        (numerator, 1_i128 << shift)
    };
    semantic::Rational::new(numerator, denominator).map_err(|_| BridgeError::InvalidSampleRate)
}

fn rational_to_f64_exact(value: semantic::Rational) -> Result<f64, BridgeError> {
    let (numerator, denominator) = value.parts();
    let converted = numerator as f64 / denominator as f64;
    if exact_positive_f64(converted).ok() == Some(value) {
        Ok(converted)
    } else {
        Err(BridgeError::Unrepresentable(
            "sample rate is not exactly representable as f64",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use legacy::{Column, Eeg};

    fn legacy_fixture() -> legacy::LegacyRecording<Eeg> {
        legacy::LegacyRecording::from_parts(
            vec![
                legacy::Channel {
                    label: Arc::from("Fp1"),
                    data: Column::I16(Arc::from(vec![1_i16, -2, 3, -4])),
                    phys_min: -100.5,
                    phys_max: 100.5,
                },
                legacy::Channel {
                    label: Arc::from("Cz"),
                    data: Column::I64(Arc::from(vec![10_i64, 20, 30, 40])),
                    phys_min: -250.0,
                    phys_max: 250.0,
                },
                legacy::Channel {
                    label: Arc::from("BDF"),
                    data: Column::I24(Arc::from(vec![-8_388_608_i32, -1, 0, 8_388_607])),
                    phys_min: -1.0,
                    phys_max: 1.0,
                },
                legacy::Channel {
                    label: Arc::from("ADC32"),
                    data: Column::I32(Arc::from(vec![i32::MIN, -1, 0, i32::MAX])),
                    phys_min: -2.0,
                    phys_max: 2.0,
                },
                legacy::Channel {
                    label: Arc::from("Derived"),
                    data: Column::F32(Arc::from(vec![-1.5_f32, -0.0, 0.25, 3.5])),
                    phys_min: -3.5,
                    phys_max: 3.5,
                },
            ],
            250.0,
            4,
        )
        .into_modality::<Eeg>(legacy::ModalitySource::Manual)
    }

    #[test]
    fn forward_mapping_is_valid_and_zero_copy() {
        let source = legacy_fixture();
        let mapped = from_legacy(&source).expect("forward bridge");
        assert_eq!(mapped.mapping.channels.len(), 5);
        assert!(mapped.fidelity.sample_values_exact);
        assert!(!mapped.fidelity.calibration_promoted);
        let opened = semantic::OpenedDataset::new(mapped.dataset, mapped.access);
        for (index, mapping) in mapped.mapping.channels.iter().enumerate() {
            let view = opened.block_view(mapping.atom_id).expect("borrowed view");
            assert_eq!(
                view.bytes().as_ptr(),
                column_bytes(&source.channels[index].data).as_ptr()
            );
        }
    }

    #[test]
    fn timed_event_overlay_is_bound_to_a_recording_clock() {
        let source = legacy_fixture();
        let event = TimedEventMapping {
            kind: semantic::ConceptId::new("bids:event/stimulus").unwrap(),
            start: semantic::Rational::new(1, 2).unwrap(),
            end: semantic::Rational::new(3, 5).unwrap(),
            uncertainty: semantic::Rational::new(0, 1).unwrap(),
        };
        let mapped = from_legacy_with_source_capsules_events_and_limits(
            &source,
            &[],
            &[event],
            semantic::ValidationLimits::default(),
        )
        .unwrap();
        assert_eq!(mapped.dataset.clocks().len(), 1);
        assert_eq!(mapped.dataset.events().len(), 1);
        assert_eq!(mapped.mapping.events.len(), 1);
        assert_eq!(
            mapped.dataset.streams()[0].clock_id(),
            Some(mapped.dataset.events()[0].clock_id())
        );
        assert!(matches!(
            to_legacy::<Eeg, _>(&mapped.dataset, &mapped.access),
            Err(BridgeError::Unrepresentable(_))
        ));
    }

    #[test]
    fn timed_event_overlay_rejects_reversed_intervals() {
        let source = legacy_fixture();
        let event = TimedEventMapping {
            kind: semantic::ConceptId::new("bids:event/stimulus").unwrap(),
            start: semantic::Rational::new(2, 1).unwrap(),
            end: semantic::Rational::new(1, 1).unwrap(),
            uncertainty: semantic::Rational::new(0, 1).unwrap(),
        };
        assert!(from_legacy_with_source_capsules_events_and_limits(
            &source,
            &[],
            &[event],
            semantic::ValidationLimits::default(),
        )
        .is_err());
    }

    #[test]
    fn timed_event_overlay_honors_catalog_limits() {
        let source = legacy_fixture();
        let event = TimedEventMapping {
            kind: semantic::ConceptId::new("bids:event/stimulus").unwrap(),
            start: semantic::Rational::new(0, 1).unwrap(),
            end: semantic::Rational::new(1, 1).unwrap(),
            uncertainty: semantic::Rational::new(0, 1).unwrap(),
        };
        let limits = semantic::ValidationLimits {
            max_catalog_records: 0,
            ..semantic::ValidationLimits::default()
        };
        assert!(
            from_legacy_with_source_capsules_events_and_limits(&source, &[], &[event], limits,)
                .is_err()
        );
    }

    #[test]
    fn representable_data_round_trips_exactly() {
        let source = legacy_fixture();
        let mapped = from_legacy(&source).expect("forward bridge");
        let restored =
            to_legacy::<Eeg, _>(&mapped.dataset, &mapped.access).expect("reverse bridge");
        assert_eq!(restored.sample_rate.to_bits(), source.sample_rate.to_bits());
        assert_eq!(restored.n_samples, source.n_samples);
        assert_eq!(restored.channels.len(), source.channels.len());
        for (actual, expected) in restored.channels.iter().zip(&source.channels) {
            assert_eq!(actual.label, expected.label);
            assert_eq!(actual.phys_min.to_bits(), expected.phys_min.to_bits());
            assert_eq!(actual.phys_max.to_bits(), expected.phys_max.to_bits());
            assert_eq!(column_bytes(&actual.data), column_bytes(&expected.data));
            assert_eq!(storage_width(&actual.data), storage_width(&expected.data));
        }
    }

    #[test]
    fn reverse_rejects_nonuniform_or_extra_semantics() {
        let source = legacy_fixture();
        let mapped = from_legacy(&source).expect("forward bridge");
        let mut draft = semantic::DatasetDraft::new(mapped.dataset.id());
        let extra_recording =
            semantic::Recording::new(semantic::ObjectId::from_bytes([0x55; 16]), Vec::new());
        draft.add_recording(extra_recording);
        let invalid = draft.validate(semantic::ValidationLimits::default());
        assert!(
            invalid.is_ok(),
            "an isolated second recording is structurally valid"
        );
        let error = to_legacy::<Eeg, _>(&invalid.unwrap(), &mapped.access).unwrap_err();
        assert!(matches!(error, BridgeError::Unrepresentable(_)));
    }

    #[test]
    fn rejects_nonfinite_legacy_metadata() {
        let mut source = legacy_fixture();
        source.channels[0].phys_min = f64::NAN;
        assert!(matches!(
            from_legacy(&source),
            Err(BridgeError::InvalidPhysicalRange { channel: 0 })
        ));
    }

    #[test]
    fn forward_mapping_can_bind_exact_source_capsules() {
        let source = legacy_fixture();
        let content_id = semantic::ContentId::from_bytes([0x42; 32]);
        let capsule = SourceCapsuleMapping {
            namespace: "adapter.edfplus.1".to_owned(),
            value: "recording.edf".to_owned(),
            content_id,
            media_type: Some("application/edf".to_owned()),
        };
        let mapped = from_legacy_with_source_capsules(&source, &[capsule]).unwrap();
        assert_eq!(mapped.dataset.source_capsules().len(), 1);
        let actual = &mapped.dataset.source_capsules()[0];
        assert_eq!(actual.content_id(), content_id);
        assert_eq!(actual.source().namespace(), "adapter.edfplus.1");
        assert_eq!(actual.source().value(), "recording.edf");
        assert_eq!(actual.media_type(), Some("application/edf"));
    }

    #[test]
    fn caller_limits_fail_closed_before_dataset_exposure() {
        let source = legacy_fixture();
        let limits = semantic::ValidationLimits {
            max_atoms: 0,
            ..semantic::ValidationLimits::default()
        };
        assert!(from_legacy_with_source_capsules_and_limits(&source, &[], limits).is_err());
    }
}
