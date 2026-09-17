use super::*;

fn file(path: &str, symbols: &str) -> FileSymbols {
    FileSymbols {
        path: path.to_owned(),
        symbols: symbols.to_owned(),
    }
}

#[test]
fn select_files_returns_the_most_relevant_paths_up_to_the_limit() {
    let files = vec![
        file("src/ui/button.rs", "struct Button"),
        file("src/auth/token.rs", "fn refresh_token"),
        file("src/auth/session.rs", "fn refresh_session"),
        file("src/auth/store.rs", "fn refresh_store"),
        file("docs/readme.md", "intro"),
    ];

    let selected = select_files("where is refresh_token used", &files);

    assert_eq!(
        selected,
        vec![
            "src/auth/token.rs".to_owned(),
            "src/auth/session.rs".to_owned(),
            "src/auth/store.rs".to_owned(),
        ]
    );
}

#[test]
fn select_files_returns_nothing_when_no_file_matches_and_none_are_richer() {
    let files = vec![file("a.rs", ""), file("b.rs", "")];

    let selected = select_files("zzz", &files);

    assert!(
        selected.len() <= MAX_FILES,
        "still bounded by the attachment budget: {selected:?}"
    );
}
