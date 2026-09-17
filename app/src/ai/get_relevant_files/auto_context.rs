//! Files attached to a user query automatically, ranked locally.
//!
//! This is the cheap half of local retrieval: no model call and no server round trip, so it can
//! run on the submit path without adding latency. The model-ranked variant stays behind the
//! `search_codebase` tool, where the agent opts in per search.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ai::agent::action_result::{AnyFileContent, FileContext};
use ai::index::FileSymbols;
use warp_core::features::FeatureFlag;
use warp_core::safe_info;
use warpui::{AppContext, SingletonEntity as _};

use super::local_search;
use crate::ai::agent::AIAgentContext;
use crate::ai::outline::{OutlineStatus, RepoOutlines};

/// How many files are attached per query, and the size budget for each file's contents.
const MAX_FILES: usize = 3;
const MAX_FILE_CHARS: usize = 6_000;

/// Builds the file context to attach to a user query, or an empty vec when the feature is off or
/// no repo outline is ready yet.
pub(crate) fn context_for_query(
    query: &str,
    working_directory: Option<&str>,
    app: &AppContext,
) -> Vec<AIAgentContext> {
    if !FeatureFlag::LocalByoInference.is_enabled() {
        return Vec::new();
    }
    let Some(working_directory) = working_directory else {
        return Vec::new();
    };
    let Some((OutlineStatus::Complete(outline), base_path)) =
        RepoOutlines::as_ref(app).get_outline(Path::new(working_directory))
    else {
        return Vec::new();
    };

    let file_symbols = outline.to_file_symbols(None);
    let candidate_count = file_symbols.len();
    let attached: Vec<AIAgentContext> = select_files(query, &file_symbols)
        .into_iter()
        .filter_map(|path| file_context(&base_path, &path))
        .collect();

    if !attached.is_empty() {
        let chars: usize = attached
            .iter()
            .map(|context| match context {
                AIAgentContext::File(file) => file.content.len(),
                _ => 0,
            })
            .sum();
        safe_info!(
            safe: (
                "event=byo_auto_context files={} chars={chars} candidates={candidate_count}",
                attached.len()
            ),
            full: (
                "event=byo_auto_context files={} chars={chars} candidates={candidate_count} paths={:?}",
                attached.len(),
                attached
                    .iter()
                    .filter_map(|context| match context {
                        AIAgentContext::File(file) => Some(file.file_name.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            )
        );
    }
    attached
}

/// The repository-relative paths to attach, most relevant first.
fn select_files(query: &str, files: &[FileSymbols]) -> Vec<String> {
    local_search::rank_candidates(query, None, files)
        .into_iter()
        .take(MAX_FILES)
        .collect()
}

fn file_context(base_path: &Path, relative_path: &str) -> Option<AIAgentContext> {
    let path = base_path.join(relative_path);
    let content = read_capped(&path, MAX_FILE_CHARS)?;
    if content.trim().is_empty() {
        return None;
    }
    let last_modified = path.metadata().ok().and_then(|meta| meta.modified().ok());
    Some(AIAgentContext::File(FileContext::new(
        relative_path.to_owned(),
        AnyFileContent::StringContent(content),
        None,
        last_modified,
    )))
}

/// Reads at most `max_chars` characters, so a large file cannot blow the request budget.
fn read_capped(path: &Path, max_chars: usize) -> Option<String> {
    let mut buffer = Vec::new();
    File::open(path)
        .ok()?
        .take(max_chars as u64 * 4)
        .read_to_end(&mut buffer)
        .ok()?;
    let text = String::from_utf8(buffer).ok()?;
    Some(match text.char_indices().nth(max_chars) {
        Some((idx, _)) => text[..idx].to_owned(),
        None => text,
    })
}

#[cfg(test)]
#[path = "auto_context_tests.rs"]
mod tests;
