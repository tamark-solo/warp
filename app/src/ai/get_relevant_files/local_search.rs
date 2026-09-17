//! Local ranking for the SearchCodebase tool.
//!
//! The outline this ranks is already built locally (`RepoOutlines`); normally only the ranking
//! call is remote. With [`FeatureFlag::LocalByoInference`] the ranking runs against the user's own
//! endpoint instead, so codebase search works without Warp credits or a server-built index.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ai::index::FileSymbols;
use itertools::Itertools;
use serde::Deserialize;
use warp_core::safe_info;

use crate::ai::byo_inference::{self, ByoEndpoint};
use crate::settings::ByoRankingMode;

/// How many candidates are handed to the model, and how many files are returned to the agent.
const MAX_CANDIDATES: usize = 30;
const MAX_RESULTS: usize = 8;
/// Symbols are truncated per candidate so the ranking prompt stays small.
const MAX_SYMBOLS_CHARS: usize = 240;
const MAX_TOKENS: u32 = 200;
/// Ranking is worth more than a fast failure: a slow answer still routes the agent to the right
/// files, so even the shortest remote budget is wider than the one autofill can afford.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RankingPolicy {
    /// Whether the endpoint is asked at all; without it the lexical order is the answer.
    pub(crate) remote: bool,
    pub(crate) timeout: Duration,
    /// Consecutive failures after which the endpoint is left alone for `cooldown`.
    pub(crate) failure_limit: u32,
    pub(crate) cooldown: Duration,
}

impl RankingPolicy {
    pub(crate) fn for_mode(mode: ByoRankingMode) -> Self {
        match mode {
            ByoRankingMode::LexicalOnly => Self {
                remote: false,
                timeout: Duration::ZERO,
                failure_limit: u32::MAX,
                cooldown: Duration::ZERO,
            },
            ByoRankingMode::Balanced => Self {
                remote: true,
                timeout: Duration::from_secs(20),
                failure_limit: 2,
                cooldown: Duration::from_secs(600),
            },
            ByoRankingMode::Patient => Self {
                remote: true,
                timeout: Duration::from_secs(60),
                failure_limit: 2,
                cooldown: Duration::from_secs(600),
            },
        }
    }
}

const SYSTEM_PROMPT: &str = "You pick which files a coding agent should read to answer a \
question about a repository.\nReply with one JSON object and nothing else:\n\
{\"files\": [\"<path>\", ...]}\nRules:\n\
- Use only paths from the candidate list, most relevant first.\n\
- Prefer files that define or implement what the question asks about.\n\
- At most 8 paths; fewer is fine when only a few are relevant.\n\
- No explanations, no markdown, no extra keys.";

/// Health of the remote ranking call. Ranking runs on the AI runtime, which has no model of its
/// own to hang this off, and the endpoint is either slow or it is not — so the state is global.
static RANKING_CIRCUIT: Mutex<RankingCircuit> = Mutex::new(RankingCircuit::new());

#[derive(Debug)]
struct RankingCircuit {
    failures: u32,
    suspended_until_ms: u64,
}

impl RankingCircuit {
    const fn new() -> Self {
        Self {
            failures: 0,
            suspended_until_ms: 0,
        }
    }

    fn is_suspended(&self, now_ms: u64) -> bool {
        now_ms < self.suspended_until_ms
    }

    fn note_failure(&mut self, policy: &RankingPolicy, now_ms: u64) {
        self.failures += 1;
        if self.failures >= policy.failure_limit {
            self.failures = 0;
            self.suspended_until_ms = now_ms + policy.cooldown.as_millis() as u64;
        }
    }

    fn note_success(&mut self) {
        self.failures = 0;
        self.suspended_until_ms = 0;
    }
}

fn ranking_circuit() -> MutexGuard<'static, RankingCircuit> {
    RANKING_CIRCUIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Ranks `files` for `query`, returning repository-relative paths, most relevant first.
pub(crate) async fn rank(
    endpoint: &ByoEndpoint,
    policy: RankingPolicy,
    query: &str,
    partial_path_segments: Option<&[String]>,
    files: &[FileSymbols],
) -> Vec<String> {
    let start = instant::Instant::now();
    let candidates = rank_candidates(query, partial_path_segments, files);
    let candidate_count = candidates.len();
    if candidates.len() <= MAX_RESULTS {
        return candidates;
    }

    if !policy.remote {
        let ranked = candidates.into_iter().take(MAX_RESULTS).collect_vec();
        safe_info!(
            safe: (
                "event=byo_local_search outcome=local candidates={candidate_count} results={}",
                ranked.len()
            ),
            full: (
                "event=byo_local_search outcome=local endpoint={} model={} candidates={candidate_count} results={}",
                endpoint.endpoint_name(),
                endpoint.model(),
                ranked.len()
            )
        );
        return ranked;
    }

    // An endpoint that just failed twice is left alone instead of being asked again on the next
    // search: the model call is what costs the agent time, and the lexical order is what it falls
    // back to anyway.
    if ranking_circuit().is_suspended(epoch_millis()) {
        let ranked = candidates.iter().take(MAX_RESULTS).cloned().collect_vec();
        safe_info!(
            safe: (
                "event=byo_local_search outcome=suspended candidates={candidate_count} results={}",
                ranked.len()
            ),
            full: (
                "event=byo_local_search outcome=suspended endpoint={} model={} candidates={candidate_count} results={}",
                endpoint.endpoint_name(),
                endpoint.model(),
                ranked.len()
            )
        );
        return ranked;
    }

    let prompt = ranking_prompt(query, &candidates, files);
    let ranked =
        match byo_inference::complete(endpoint, SYSTEM_PROMPT, &prompt, MAX_TOKENS, policy.timeout)
            .await
        {
            Ok(raw) => match parse_ranking(&raw, &candidates) {
                Some(ranked) => {
                    ranking_circuit().note_success();
                    ranked
                }
                None => {
                    ranking_circuit().note_failure(&policy, epoch_millis());
                    safe_info!(
                        safe: ("event=byo_local_search outcome=unparsed"),
                        full: ("event=byo_local_search outcome=unparsed response={raw:?}")
                    );
                    candidates.clone()
                }
            },
            Err(err) => {
                ranking_circuit().note_failure(&policy, epoch_millis());
                safe_info!(
                    safe: ("event=byo_local_search outcome=error"),
                    full: (
                        "event=byo_local_search outcome=error endpoint={} model={} error={err:#}",
                        endpoint.endpoint_name(),
                        endpoint.model()
                    )
                );
                candidates.clone()
            }
        };

    // The model's order leads; the heuristic order fills the remaining slots so a terse answer
    // still gives the agent something to read.
    let ranked = ranked
        .into_iter()
        .chain(candidates)
        .unique()
        .take(MAX_RESULTS)
        .collect_vec();
    safe_info!(
        safe: (
            "event=byo_local_search outcome=ranked candidates={candidate_count} results={} latency_ms={}",
            ranked.len(),
            start.elapsed().as_millis()
        ),
        full: (
            "event=byo_local_search outcome=ranked endpoint={} model={} candidates={candidate_count} results={} query_len={} latency_ms={}",
            endpoint.endpoint_name(),
            endpoint.model(),
            ranked.len(),
            query.len(),
            start.elapsed().as_millis()
        )
    );
    ranked
}

/// Orders candidate files by lexical overlap with the query, without any model call.
pub(super) fn rank_candidates(
    query: &str,
    partial_path_segments: Option<&[String]>,
    files: &[FileSymbols],
) -> Vec<String> {
    let tokens = query_tokens(query);
    let mut scored = files
        .iter()
        .map(|file| (score_file(&tokens, partial_path_segments, file), file))
        .collect_vec();
    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.symbols.len().cmp(&left.1.symbols.len()))
            .then_with(|| left.1.path.cmp(&right.1.path))
    });

    // When nothing overlaps the query, the symbol-richest files still give the model a choice.
    let any_match = scored.iter().any(|(score, _)| *score > 0);
    scored
        .into_iter()
        .filter(|(score, _)| !any_match || *score > 0)
        .take(MAX_CANDIDATES)
        .map(|(_, file)| file.path.clone())
        .collect()
}

fn score_file(
    tokens: &[String],
    partial_path_segments: Option<&[String]>,
    file: &FileSymbols,
) -> i32 {
    let path = file.path.to_lowercase();
    let symbols = file.symbols.to_lowercase();
    let mut score = 0;
    for token in tokens {
        if path.contains(token.as_str()) {
            score += 3;
        }
        if symbols.contains(token.as_str()) {
            score += 2;
        }
    }
    if let Some(segments) = partial_path_segments {
        score += segments
            .iter()
            .filter(|segment| file.path.contains(segment.as_str()))
            .count() as i32
            * 5;
    }
    score
}

/// Splits a query into lowercase search tokens, dropping stopwords and single characters.
fn query_tokens(query: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the",
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "code",
        "define",
        "defined",
        "does",
        "file",
        "files",
        "for",
        "from",
        "function",
        "functions",
        "handle",
        "handled",
        "handles",
        "handling",
        "how",
        "implement",
        "implementation",
        "implemented",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "repo",
        "repository",
        "that",
        "the",
        "this",
        "to",
        "use",
        "used",
        "uses",
        "using",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    ];

    split_identifiers(query)
        .into_iter()
        .filter(|token| token.len() > 1 && !STOPWORDS.contains(&token.as_str()))
        .unique()
        .collect()
}

/// Splits text into lowercase words, breaking camelCase and snake_case boundaries.
fn split_identifiers(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut previous_was_lowercase = false;
    for character in text.chars() {
        if character.is_alphanumeric() {
            if character.is_uppercase() && previous_was_lowercase && !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            previous_was_lowercase = character.is_lowercase();
            current.push(character.to_ascii_lowercase());
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            previous_was_lowercase = false;
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn ranking_prompt(query: &str, candidates: &[String], files: &[FileSymbols]) -> String {
    let symbols_by_path: HashMap<&str, &str> = files
        .iter()
        .map(|file| (file.path.as_str(), file.symbols.as_str()))
        .collect();

    let mut prompt = String::from("Question about this repository:\n");
    prompt.push_str(query.trim());
    prompt.push_str("\n\nCandidate files:\n");
    for path in candidates {
        prompt.push_str(path);
        prompt.push('\n');
        if let Some(symbols) = symbols_by_path.get(path.as_str()) {
            let symbols = symbols.trim();
            if !symbols.is_empty() {
                prompt.push_str("  ");
                prompt.push_str(truncate(symbols, MAX_SYMBOLS_CHARS));
                prompt.push('\n');
            }
        }
    }
    prompt
}

#[derive(Deserialize)]
struct RawRanking {
    #[serde(default)]
    files: Vec<String>,
}

fn parse_ranking(raw: &str, candidates: &[String]) -> Option<Vec<String>> {
    let json = byo_inference::extract_json_object(raw)?;
    let parsed: RawRanking = serde_json::from_str(json).ok()?;
    let allowed: HashSet<&str> = candidates.iter().map(String::as_str).collect();
    let ranked = parsed
        .files
        .into_iter()
        .map(|path| path.trim().to_owned())
        .filter(|path| allowed.contains(path.as_str()))
        .unique()
        .collect_vec();
    (!ranked.is_empty()).then_some(ranked)
}

fn truncate(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
#[path = "local_search_tests.rs"]
mod tests;
