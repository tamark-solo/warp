//! Client-side inference against a user-configured custom endpoint.
//!
//! Agent Mode requests reach the user's endpoint through Warp's multi-agent API, which forwards
//! the endpoint config it is sent with the request. Surfaces that Warp serves itself — the Agent
//! Predict endpoints behind next-command autofill among them — carry no model parameter, so they
//! can only ever run on Warp-hosted models. This module lets such a surface call the user's own
//! endpoint directly, gated by [`FeatureFlag::LocalByoInference`].

use std::sync::LazyLock;
use std::time::Duration;

use ai::api_keys::{
    ApiKeyManager, CustomEndpoint, CustomEndpointSchema, validate_custom_endpoint_url,
};
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use settings::Setting as _;
use warp_core::features::FeatureFlag;
use warpui::{AppContext, SingletonEntity};

use crate::settings::AISettings;
use crate::workspaces::user_workspaces::{TeamScope, UserWorkspaces};

const ERROR_BODY_CHARS: usize = 240;

/// A custom endpoint together with the model requests against it are addressed to.
#[derive(Clone, Debug)]
pub(crate) struct ByoEndpoint {
    endpoint: CustomEndpoint,
    model: String,
}

impl ByoEndpoint {
    pub(crate) fn endpoint_name(&self) -> &str {
        &self.endpoint.name
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }
}

/// Resolves the endpoint that client-side inference should use, or `None` when the feature is
/// off, the scope's policy forbids member endpoints, or no usable endpoint is configured.
pub(crate) fn resolve<S: TeamScope + ?Sized>(app: &AppContext, scope: &S) -> Option<ByoEndpoint> {
    if !FeatureFlag::LocalByoInference.is_enabled() {
        return None;
    }

    let workspaces = UserWorkspaces::as_ref(app);
    if !workspaces.is_byo_endpoint_enabled(app)
        || !workspaces.are_member_byo_endpoints_allowed(scope)
    {
        return None;
    }

    let endpoints = ApiKeyManager::as_ref(app).custom_endpoints();
    let configured_model = AISettings::as_ref(app).byo_autofill_model.value().clone();

    // A model the user configured wins wherever it lives; otherwise the first usable endpoint's
    // first named model answers.
    let usable_endpoints = || endpoints.iter().filter(|endpoint| is_usable(endpoint));
    let endpoint = configured_model
        .as_deref()
        .and_then(|config_key| {
            usable_endpoints().find(|endpoint| {
                endpoint
                    .models
                    .iter()
                    .any(|model| model.config_key == config_key)
            })
        })
        .or_else(|| usable_endpoints().next())?;
    let named_models = || {
        endpoint
            .models
            .iter()
            .filter(|model| !model.name.trim().is_empty())
    };
    let model = configured_model
        .as_deref()
        .and_then(|config_key| named_models().find(|model| model.config_key == config_key))
        .or_else(|| named_models().next())?;

    Some(ByoEndpoint {
        endpoint: endpoint.clone(),
        model: model.name.trim().to_owned(),
    })
}

/// Resolves the endpoint for a caller that only carries the request's team uid rather than a
/// view-scoped [`TeamScope`].
#[cfg(not(target_family = "wasm"))]
pub(crate) fn resolve_for_team_uid(
    app: &AppContext,
    team_uid: Option<crate::server::ids::ServerId>,
) -> Option<ByoEndpoint> {
    let scope = match team_uid {
        Some(team_uid) => crate::workspaces::user_workspaces::TeamScopeForCli::Team(team_uid),
        None => crate::workspaces::user_workspaces::TeamScopeForCli::Personal,
    };
    resolve(app, &scope)
}

/// A model autofill can be pointed at, as offered by [`selectable_models`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ByoAutofillChoice {
    pub config_key: String,
    pub endpoint_name: String,
    pub model_label: String,
}

/// Every model autofill can be pointed at, in endpoint order.
pub(crate) fn selectable_models<S: TeamScope + ?Sized>(
    app: &AppContext,
    scope: &S,
) -> Vec<ByoAutofillChoice> {
    if !UserWorkspaces::as_ref(app).are_member_byo_endpoints_allowed(scope) {
        return Vec::new();
    }

    ApiKeyManager::as_ref(app)
        .custom_endpoints()
        .iter()
        .filter(|endpoint| is_usable(endpoint))
        .flat_map(|endpoint| {
            endpoint
                .models
                .iter()
                .filter(|model| !model.name.trim().is_empty())
                .map(|model| ByoAutofillChoice {
                    config_key: model.config_key.clone(),
                    endpoint_name: endpoint.name.clone(),
                    model_label: model.display_label().to_owned(),
                })
        })
        .collect()
}

fn is_usable(endpoint: &CustomEndpoint) -> bool {
    !endpoint.api_key.trim().is_empty()
        && validate_custom_endpoint_url(&endpoint.url).is_ok()
        && endpoint
            .models
            .iter()
            .any(|model| !model.name.trim().is_empty())
}

/// Sends a single non-streaming completion against `endpoint` and returns its text, giving up
/// after `timeout`.
pub(crate) async fn complete(
    endpoint: &ByoEndpoint,
    system: &str,
    user: &str,
    max_tokens: u32,
    timeout: Duration,
) -> Result<String> {
    let url = request_url(&endpoint.endpoint.url, endpoint.endpoint.schema);
    let key = endpoint.endpoint.api_key.trim();
    let model = endpoint.model.as_str();

    let response = match endpoint.endpoint.schema {
        CustomEndpointSchema::OpenaiChatCompletions => {
            client()
                .post(&url)
                .bearer_auth(key)
                .json(&json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": false,
                }))
                .timeout(timeout)
                .send()
                .await
        }
        CustomEndpointSchema::OpenaiResponses => {
            client()
                .post(&url)
                .bearer_auth(key)
                .json(&json!({
                    "model": model,
                    "instructions": system,
                    "input": user,
                    "max_output_tokens": max_tokens,
                    "stream": false,
                }))
                .timeout(timeout)
                .send()
                .await
        }
        CustomEndpointSchema::AnthropicMessages => {
            client()
                .post(&url)
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }))
                .timeout(timeout)
                .send()
                .await
        }
    }
    .context("custom endpoint request failed")?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read custom endpoint response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "custom endpoint returned {status}: {}",
            truncate(body.trim(), ERROR_BODY_CHARS)
        ));
    }

    extract_text(endpoint.endpoint.schema, &body).ok_or_else(|| {
        anyhow!(
            "custom endpoint response contained no text for schema {}",
            endpoint.endpoint.schema.display_name()
        )
    })
}

/// Finds the first JSON object in `text`, tolerating markdown fences and surrounding prose.
pub(crate) fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end > start).then(|| &text[start..=end])
}

fn client() -> &'static reqwest::Client {
    static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);
    &CLIENT
}

/// Resolves the request URL for `schema` from a user-configured base URL, which may already
/// include the endpoint path.
fn request_url(base_url: &str, schema: CustomEndpointSchema) -> String {
    let base = base_url.trim_end_matches('/');
    let path = match schema {
        CustomEndpointSchema::OpenaiChatCompletions => "/chat/completions",
        CustomEndpointSchema::OpenaiResponses => "/responses",
        CustomEndpointSchema::AnthropicMessages => "/messages",
    };
    if base.ends_with(path) {
        return base.to_owned();
    }
    // The Messages API is versioned; a base URL that stops at the host needs the version segment.
    if schema == CustomEndpointSchema::AnthropicMessages && !base.ends_with("/v1") {
        return format!("{base}/v1{path}");
    }
    format!("{base}{path}")
}

fn extract_text(schema: CustomEndpointSchema, body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let text = match schema {
        CustomEndpointSchema::OpenaiChatCompletions => value["choices"][0]["message"]["content"]
            .as_str()?
            .to_owned(),
        CustomEndpointSchema::OpenaiResponses => match value["output_text"].as_str() {
            Some(text) => text.to_owned(),
            None => value["output"]
                .as_array()?
                .iter()
                .filter_map(|item| item["content"].as_array())
                .flatten()
                .filter_map(|part| part["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
        },
        CustomEndpointSchema::AnthropicMessages => value["content"]
            .as_array()?
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
    };
    (!text.trim().is_empty()).then_some(text)
}

fn truncate(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
#[path = "byo_inference_tests.rs"]
mod tests;
