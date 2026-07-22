use lamquant_legacy_adapter::{handle, LegacyError, ProcessRequest, ProcessResponse};
use std::io::{self, Read};

const MAX_PROCESS_REQUEST_BYTES: usize = 1024 * 1024;

fn run() -> Result<ProcessResponse, LegacyError> {
    let mut input = Vec::new();
    io::stdin()
        .take((MAX_PROCESS_REQUEST_BYTES + 1) as u64)
        .read_to_end(&mut input)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    if input.len() > MAX_PROCESS_REQUEST_BYTES {
        return Err(LegacyError::InvalidProtocol(
            "request exceeds 1 MiB protocol limit".to_owned(),
        ));
    }
    let request = serde_json::from_slice::<ProcessRequest>(&input)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    Ok(handle(request))
}

fn main() {
    let response = run().unwrap_or_else(|error| ProcessResponse::Error {
        code: error.code().to_owned(),
        message: error.to_string(),
    });
    println!(
        "{}",
        serde_json::to_string(&response).expect("protocol response is serializable")
    );
}
