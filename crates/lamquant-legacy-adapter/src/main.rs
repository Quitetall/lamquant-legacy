use lamquant_legacy_adapter::{handle, LegacyError, ProcessRequest, ProcessResponse};
use std::io::{self, Read};

fn run() -> Result<ProcessResponse, LegacyError> {
    let mut input = Vec::new();
    io::stdin()
        .take(1024 * 1024)
        .read_to_end(&mut input)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
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
