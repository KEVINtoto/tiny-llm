//! Synchronous message transport to the standalone Python MLX service.

use std::time::Duration;

use reqwest::{Url, blocking::Client};
use serde_json::{Value, json};

use crate::generation::{Generate, Message};
use crate::protocol::AgentError;

pub struct HttpGenerator {
    client: Client,
    endpoint: Url,
    max_tokens: i64,
    enable_thinking: bool,
}

impl HttpGenerator {
    pub fn new(
        base_url: &str,
        max_tokens: i64,
        enable_thinking: bool,
        timeout: Duration,
    ) -> Result<Self, AgentError> {
        if max_tokens <= 0 || timeout.is_zero() {
            return Err(AgentError(
                "max_tokens and request timeout must be positive".into(),
            ));
        }
        let mut endpoint = Url::parse(base_url)
            .map_err(|e| AgentError(format!("invalid model service URL: {e}")))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(AgentError(
                "model service URL must be an HTTP(S) base URL without query or fragment".into(),
            ));
        }
        endpoint.set_path(&format!(
            "{}/generate",
            endpoint.path().trim_end_matches('/')
        ));
        let client = Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|e| AgentError(format!("could not create model service client: {e}")))?;
        Ok(Self {
            client,
            endpoint,
            max_tokens,
            enable_thinking,
        })
    }
}

impl Generate for HttpGenerator {
    fn generate(&mut self, messages: &[Message]) -> Result<String, AgentError> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .json(&json!({
                "messages": messages,
                "max_tokens": self.max_tokens,
                "enable_thinking": self.enable_thinking,
            }))
            .send()
            .map_err(|e| AgentError(format!("model service request failed: {e}")))?;
        let status = response.status();
        let body = response
            .text()
            .map_err(|e| AgentError(format!("could not read model service response: {e}")))?;
        if !status.is_success() {
            let payload = serde_json::from_str::<Value>(&body).ok();
            let detail = payload
                .as_ref()
                .and_then(|v| v.get("error"))
                .and_then(Value::as_str)
                .unwrap_or(&body);
            return Err(AgentError(format!(
                "model service returned HTTP {status}: {detail}"
            )));
        }
        let payload: Value = serde_json::from_str(&body)
            .map_err(|e| AgentError(format!("invalid model service JSON response: {e}")))?;
        payload
            .get("response")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                AgentError("model service response must contain a string 'response'".into())
            })
    }
}
