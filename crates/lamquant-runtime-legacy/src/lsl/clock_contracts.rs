use serde::{Deserialize, Deserializer, Serialize};

use super::validate_identity;
use crate::error::{Result, RuntimeError};

/// Stable identity for one clock domain. It is metadata, not a claim that two
/// hosts share a clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct LslClockId(pub(super) String);

impl LslClockId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() || value.len() > 512 {
            return Err(RuntimeError::Source {
                name: "lsl-clock".into(),
                msg: "clock identity must contain 1..=512 bytes".into(),
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for LslClockId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Measured or explicitly unmeasured relation from publisher timestamps to the
/// receiving host clock. Offset and uncertainty are all-or-none so an unknown
/// relation cannot masquerade as a zero-offset observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LslClockRelation {
    pub publisher_clock: LslClockId,
    pub receiver_clock: LslClockId,
    pub offset_micros: Option<i64>,
    pub uncertainty_micros: Option<u64>,
    pub observed_at_receiver_micros: Option<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLslClockRelation {
    publisher_clock: LslClockId,
    receiver_clock: LslClockId,
    offset_micros: Option<i64>,
    uncertainty_micros: Option<u64>,
    observed_at_receiver_micros: Option<i64>,
}

impl<'de> Deserialize<'de> for LslClockRelation {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawLslClockRelation::deserialize(deserializer)?;
        let relation = Self {
            publisher_clock: raw.publisher_clock,
            receiver_clock: raw.receiver_clock,
            offset_micros: raw.offset_micros,
            uncertainty_micros: raw.uncertainty_micros,
            observed_at_receiver_micros: raw.observed_at_receiver_micros,
        };
        relation.validate().map_err(serde::de::Error::custom)?;
        Ok(relation)
    }
}

impl LslClockRelation {
    pub fn unobserved(stream_name: &str) -> Result<Self> {
        Ok(Self {
            publisher_clock: LslClockId::new(format!(
                "lsl.publisher-name-unverified:{stream_name}"
            ))?,
            receiver_clock: LslClockId::new("host.monotonic:unbound")?,
            offset_micros: None,
            uncertainty_micros: None,
            observed_at_receiver_micros: None,
        })
    }

    pub fn validate(&self) -> Result<()> {
        validate_identity("publisher_clock", self.publisher_clock.as_str()).map_err(|msg| {
            RuntimeError::Source {
                name: "lsl-clock".into(),
                msg,
            }
        })?;
        validate_identity("receiver_clock", self.receiver_clock.as_str()).map_err(|msg| {
            RuntimeError::Source {
                name: "lsl-clock".into(),
                msg,
            }
        })?;
        let unobserved = self.offset_micros.is_none()
            && self.uncertainty_micros.is_none()
            && self.observed_at_receiver_micros.is_none();
        if unobserved {
            Ok(())
        } else {
            Err(RuntimeError::Source {
                name: "lsl-clock".into(),
                msg: "measured LSL clock relations are unsupported until the inlet binds liblsl time-correction evidence; use an explicit unobserved relation".into(),
            })
        }
    }
}
