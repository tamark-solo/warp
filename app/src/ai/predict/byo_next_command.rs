//! Next-command autofill answered by the user's own custom endpoint.
//!
//! Warp's `/ai/generate_input_suggestions` endpoint builds its own prompt server-side, so the
//! client cannot route it to a custom endpoint. This module builds an equivalent prompt from the
//! same request and asks the endpoint for the same JSON shape.

use std::fmt::Write as _;
use std::time::Duration;

use anyhow::Context as _;
use instant::Instant;
use itertools::Itertools;
use serde::Deserialize;
use warp_core::safe_info;

use super::generate_ai_input_suggestions::{
    GenerateAIInputSuggestionsRequest, GenerateAIInputSuggestionsResponseV2,
};
use crate::ai::byo_inference::{self, ByoEndpoint};
use crate::server::server_api::AIApiError;
use crate::settings::ByoAutofillWait;

/// Ceiling on the terminal context sent per request, so suggestions stay inside the autofill
/// latency budget.
const MAX_CONTEXT_CHARS: usize = 2_500;
const MAX_COMMANDS: usize = 3;
const MAX_TOKENS: u32 = 200;
const RECENT_BLOCK_LIMIT: usize = 2;
const BLOCK_OUTPUT_CHARS: usize = 400;

/// How long a single autofill call may wait on the endpoint, or `None` when the endpoint is left
/// out of autofill entirely.
pub(crate) fn request_timeout(wait: ByoAutofillWait) -> Option<Duration> {
    match wait {
        ByoAutofillWait::Off => None,
        ByoAutofillWait::TenSeconds => Some(Duration::from_secs(10)),
        ByoAutofillWait::TwentySeconds => Some(Duration::from_secs(20)),
    }
}

pub(crate) async fn generate_suggestions(
    endpoint: &ByoEndpoint,
    request: &GenerateAIInputSuggestionsRequest,
    timeout: Duration,
) -> Result<GenerateAIInputSuggestionsResponseV2, AIApiError> {
    let start = Instant::now();
    let raw = byo_inference::complete(
        endpoint,
        SYSTEM_PROMPT,
        &user_prompt(request),
        MAX_TOKENS,
        timeout,
    )
    .await;
    let latency_ms = start.elapsed().as_millis();

    match raw {
        Ok(raw) => {
            let parsed = parse_response(
                &raw,
                request.prefix.as_deref(),
                &request.rejected_suggestions,
            );
            safe_info!(
                safe: (
                    "event=byo_next_command outcome=response model={} latency_ms={latency_ms} parsed={}",
                    endpoint.model(),
                    parsed.is_ok()
                ),
                full: (
                    "event=byo_next_command outcome=response endpoint={} model={} latency_ms={latency_ms} parsed={} response={raw:?}",
                    endpoint.endpoint_name(),
                    endpoint.model(),
                    parsed.is_ok()
                )
            );
            parsed.map_err(AIApiError::Other)
        }
        Err(err) => {
            safe_info!(
                safe: (
                    "event=byo_next_command outcome=error model={} latency_ms={latency_ms}",
                    endpoint.model()
                ),
                full: (
                    "event=byo_next_command outcome=error endpoint={} model={} latency_ms={latency_ms} error={err:#}",
                    endpoint.endpoint_name(),
                    endpoint.model()
                )
            );
            Err(AIApiError::Other(err))
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawSuggestions {
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    most_likely_action: Option<String>,
}

fn parse_response(
    raw: &str,
    prefix: Option<&str>,
    rejected: &[String],
) -> anyhow::Result<GenerateAIInputSuggestionsResponseV2> {
    let json = byo_inference::extract_json_object(raw)
        .context("custom endpoint response contained no JSON object")?;
    let parsed: RawSuggestions = serde_json::from_str(json)
        .context("custom endpoint response was not the expected shape")?;

    let usable = |candidate: &str| {
        let candidate = candidate.trim();
        !candidate.is_empty()
            && !rejected.iter().any(|rejected| rejected == candidate)
            && prefix.is_none_or(|prefix| candidate.starts_with(prefix))
    };

    let mut commands = parsed
        .commands
        .iter()
        .map(|command| command.trim())
        .filter(|command| usable(command))
        .unique()
        .take(MAX_COMMANDS)
        .map(str::to_owned)
        .collect_vec();

    let most_likely_action = parsed
        .most_likely_action
        .as_deref()
        .map(str::trim)
        .filter(|action| usable(action))
        .map(str::to_owned)
        .or_else(|| commands.first().cloned())
        .context("custom endpoint returned no usable suggestion")?;
    if !commands.contains(&most_likely_action) {
        commands.insert(0, most_likely_action.clone());
    }

    Ok(GenerateAIInputSuggestionsResponseV2 {
        commands,
        ai_queries: vec![],
        most_likely_action,
    })
}

const SYSTEM_PROMPT: &str = "You predict the next shell command a developer will run in an \
interactive terminal session.\nReply with one JSON object and nothing else:\n\
{\"commands\": [\"<command>\", ...], \"most_likely_action\": \"<command>\"}\nRules:\n\
- At most 3 commands, most likely first; `most_likely_action` is the single best one.\n\
- Every command is a single line, directly runnable in the user's shell, and does not wrap.\n\
- When a typed prefix is given, `most_likely_action` and every command must start with it \
verbatim.\n- Ground suggestions in the working directory, shell, git branch, recent commands and \
their output.\n- Never suggest a command listed as already rejected.\n\
- Answer immediately with the JSON; no explanations, no markdown, no extra keys.";

fn user_prompt(request: &GenerateAIInputSuggestionsRequest) -> String {
    let mut context = String::new();

    if let Some(system_context) = request.system_context.as_deref()
        && !system_context.trim().is_empty()
    {
        let _ = writeln!(context, "{system_context}");
    }

    if let Some(block) = request.block_context.as_deref() {
        if let Some(pwd) = block.pwd.as_deref() {
            let _ = writeln!(context, "working directory: {pwd}");
        }
        if let Some(shell) = block.shell.as_deref() {
            let _ = writeln!(context, "shell: {shell}");
        }
        if let Some(branch) = block.git_branch.as_deref() {
            let _ = writeln!(context, "git branch: {branch}");
        }
        let _ = writeln!(
            context,
            "last command: {}\nexit code: {}",
            block.command.trim(),
            block.exit_code
        );
        let output = truncate(block.output.trim(), BLOCK_OUTPUT_CHARS);
        if !output.is_empty() {
            let _ = writeln!(context, "last command output:\n{output}");
        }
    }

    if !request.history_context.trim().is_empty() {
        let _ = writeln!(
            context,
            "commands run in similar past sessions:\n{}",
            request.history_context.trim()
        );
    }

    let recent_blocks = request
        .context_messages
        .iter()
        .rev()
        .take(RECENT_BLOCK_LIMIT)
        .rev()
        .map(|message| message.trim())
        .filter(|message| !message.is_empty())
        .collect_vec();
    if !recent_blocks.is_empty() {
        let _ = writeln!(
            context,
            "recent terminal blocks:\n{}",
            recent_blocks.join("\n---\n")
        );
    }

    if !request.rejected_suggestions.is_empty() {
        let _ = writeln!(
            context,
            "already rejected: {}",
            request.rejected_suggestions.iter().join(", ")
        );
    }

    let mut prompt = truncate(&context, MAX_CONTEXT_CHARS).to_owned();
    match request.prefix.as_deref() {
        Some(prefix) if !prefix.is_empty() => {
            let _ = write!(
                prompt,
                "\nthe user has typed: {prefix:?}\nComplete that command. `most_likely_action` must \
start with the typed text."
            );
        }
        _ => prompt.push_str(
            "\nthe user has not typed anything yet. Suggest the single most likely next command.",
        ),
    }
    prompt
}

fn truncate(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
#[path = "byo_next_command_tests.rs"]
mod tests;
