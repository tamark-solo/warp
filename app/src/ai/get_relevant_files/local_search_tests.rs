use super::*;

fn file(path: &str, symbols: &str) -> FileSymbols {
    FileSymbols {
        path: path.to_owned(),
        symbols: symbols.to_owned(),
    }
}

#[test]
fn splits_camel_case_and_snake_case_into_tokens() {
    assert_eq!(
        split_identifiers("getUserToken refresh_auth_token"),
        vec!["get", "user", "token", "refresh", "auth", "token"]
    );
}

#[test]
fn query_tokens_drop_stopwords_and_duplicates() {
    let tokens = query_tokens("Where is the auth token handled in the auth code?");

    assert_eq!(tokens, vec!["auth", "token"]);
}

#[test]
fn prefilter_ranks_path_and_symbol_matches_above_unrelated_files() {
    let files = vec![
        file("src/ui/button.rs", "struct Button"),
        file("src/auth/token.rs", "fn refresh_token"),
        file("docs/readme.md", "intro"),
    ];

    let ranked = rank_candidates("how does refresh_token work", None, &files);

    assert_eq!(
        ranked.first().map(String::as_str),
        Some("src/auth/token.rs")
    );
    assert!(
        !ranked.contains(&"docs/readme.md".to_owned()),
        "unmatched files are dropped: {ranked:?}"
    );
}

#[test]
fn prefilter_boosts_files_matching_partial_path_segments() {
    let files = vec![
        file("src/other/thing.rs", "fn thing"),
        file("src/auth/thing.rs", "fn thing"),
    ];
    let segments = vec!["auth".to_owned()];

    let ranked = rank_candidates("thing", Some(&segments), &files);

    assert_eq!(
        ranked.first().map(String::as_str),
        Some("src/auth/thing.rs")
    );
}

#[test]
fn prefilter_falls_back_to_symbol_richest_files_when_nothing_matches() {
    let files = vec![
        file("a.rs", ""),
        file("b.rs", "fn one\nfn two\nfn three"),
        file("c.rs", "fn only"),
    ];

    let ranked = rank_candidates("zzz", None, &files);

    assert_eq!(ranked.first().map(String::as_str), Some("b.rs"));
    assert_eq!(ranked.len(), 3, "the model still gets candidates");
}

#[test]
fn parse_ranking_keeps_only_candidate_paths_in_order() {
    let candidates = vec!["src/a.rs".to_owned(), "src/b.rs".to_owned()];
    let raw =
        "```json\n{\"files\": [\"src/b.rs\", \"src/evil.rs\", \"src/a.rs\", \"src/b.rs\"]}\n```";

    let ranked = parse_ranking(raw, &candidates).expect("ranking parses");

    assert_eq!(ranked, vec!["src/b.rs", "src/a.rs"]);
}

#[test]
fn parse_ranking_rejects_responses_without_usable_paths() {
    let candidates = vec!["src/a.rs".to_owned()];

    assert!(parse_ranking("I cannot help with that", &candidates).is_none());
    assert!(parse_ranking("{\"files\": [\"src/other.rs\"]}", &candidates).is_none());
}

#[test]
fn ranking_prompt_lists_candidates_with_truncated_symbols() {
    let files = vec![
        file("src/a.rs", "fn a"),
        file("src/b.rs", &"x".repeat(MAX_SYMBOLS_CHARS + 50)),
    ];
    let candidates = vec!["src/a.rs".to_owned(), "src/b.rs".to_owned()];

    let prompt = ranking_prompt("where is a", &candidates, &files);

    assert!(prompt.contains("where is a"));
    assert!(prompt.contains("src/a.rs\n  fn a"));
    assert!(
        prompt.len() < MAX_SYMBOLS_CHARS * 2,
        "symbols are truncated per candidate: {} chars",
        prompt.len()
    );
}

fn balanced_policy() -> RankingPolicy {
    RankingPolicy::for_mode(ByoRankingMode::Balanced)
}

#[test]
fn ranking_policy_presets_trade_latency_for_endpoint_use() {
    let lexical = RankingPolicy::for_mode(ByoRankingMode::LexicalOnly);
    assert!(!lexical.remote, "the local preset never calls the endpoint");

    let balanced = balanced_policy();
    let patient = RankingPolicy::for_mode(ByoRankingMode::Patient);
    assert!(
        balanced.remote && patient.remote,
        "both remote presets ask the endpoint"
    );
    assert!(
        patient.timeout > balanced.timeout,
        "the patient preset waits longer per call: {:?} vs {:?}",
        patient.timeout,
        balanced.timeout
    );
}

#[test]
fn ranking_circuit_suspends_only_after_consecutive_failures() {
    let policy = balanced_policy();
    let mut circuit = RankingCircuit::new();
    let now = 1_000;

    circuit.note_failure(&policy, now);
    assert!(
        !circuit.is_suspended(now),
        "a single failure keeps the endpoint in use"
    );

    circuit.note_failure(&policy, now);
    assert!(
        circuit.is_suspended(now),
        "consecutive failures suspend remote ranking"
    );
}

#[test]
fn ranking_circuit_resumes_after_the_cooldown() {
    let policy = balanced_policy();
    let mut circuit = RankingCircuit::new();
    circuit.note_failure(&policy, 0);
    circuit.note_failure(&policy, 0);
    let cooldown_ms = policy.cooldown.as_millis() as u64;

    assert!(circuit.is_suspended(cooldown_ms - 1));
    assert!(!circuit.is_suspended(cooldown_ms));
}

#[test]
fn ranking_circuit_success_clears_failures_and_suspension() {
    let policy = balanced_policy();
    let mut circuit = RankingCircuit::new();
    circuit.note_failure(&policy, 0);
    circuit.note_success();
    circuit.note_failure(&policy, 0);
    assert!(
        !circuit.is_suspended(0),
        "a success between failures resets the count"
    );

    circuit.note_failure(&policy, 0);
    assert!(circuit.is_suspended(0));
    circuit.note_success();
    assert!(!circuit.is_suspended(0));
}
