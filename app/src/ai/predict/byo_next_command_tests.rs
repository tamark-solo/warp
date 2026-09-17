use super::*;

fn request_with_prefix(prefix: Option<&str>) -> GenerateAIInputSuggestionsRequest {
    GenerateAIInputSuggestionsRequest {
        prefix: prefix.map(str::to_owned),
        ..Default::default()
    }
}

#[test]
fn parses_commands_and_most_likely_action_from_fenced_json() {
    let raw = "```json\n{\"commands\": [\"cargo test\", \"cargo build\"], \
\"most_likely_action\": \"cargo test\"}\n```";

    let response = parse_response(raw, None, &[]).expect("response parses");

    assert_eq!(response.commands, vec!["cargo test", "cargo build"]);
    assert_eq!(response.most_likely_action, "cargo test");
}

#[test]
fn rejects_suggestion_that_does_not_extend_typed_prefix() {
    let raw = "{\"commands\": [\"git status\"], \"most_likely_action\": \"git status\"}";

    let error = parse_response(raw, Some("cargo "), &[]).expect_err("prefix mismatch is unusable");

    assert!(
        error.to_string().contains("no usable suggestion"),
        "unexpected error: {error}"
    );
}

#[test]
fn falls_back_to_first_command_when_most_likely_action_is_missing() {
    let raw = "{\"commands\": [\"cargo check\", \"cargo test\"]}";

    let response = parse_response(raw, None, &[]).expect("response parses");

    assert_eq!(response.most_likely_action, "cargo check");
}

#[test]
fn drops_rejected_and_duplicate_commands() {
    let raw = "{\"commands\": [\"git diff\", \"git status\", \"git status\"], \
\"most_likely_action\": \"git status\"}";
    let rejected = vec!["git diff".to_owned()];

    let response = parse_response(raw, None, &rejected).expect("response parses");

    assert_eq!(response.commands, vec!["git status"]);
}

#[test]
fn reports_error_when_response_has_no_json_object() {
    let error = parse_response("I cannot help with that.", None, &[])
        .expect_err("prose without JSON is unusable");

    assert!(
        error.to_string().contains("no JSON object"),
        "unexpected error: {error}"
    );
}

#[test]
fn zero_state_prompt_asks_for_a_suggestion_without_a_prefix() {
    let prompt = user_prompt(&request_with_prefix(None));

    assert!(prompt.contains("has not typed anything yet"));
}

#[test]
fn prefix_prompt_repeats_the_typed_text() {
    let prompt = user_prompt(&request_with_prefix(Some("cargo te")));

    assert!(prompt.contains("cargo te"));
}

#[test]
fn request_timeout_maps_every_wait_preset() {
    assert_eq!(
        request_timeout(ByoAutofillWait::Off),
        None,
        "the off preset keeps the endpoint out of autofill"
    );
    assert_eq!(
        request_timeout(ByoAutofillWait::TenSeconds),
        Some(Duration::from_secs(10))
    );
    assert_eq!(
        request_timeout(ByoAutofillWait::TwentySeconds),
        Some(Duration::from_secs(20)),
        "the longest preset gives a slow endpoint more room than the default"
    );
}
