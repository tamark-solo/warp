use std::collections::HashMap;
use std::future::Future;
use std::sync::Mutex;

use ai::api_keys::{
    ApiKeyManager, CustomEndpointDefinition, CustomEndpointDefinitions, CustomEndpointId,
    CustomEndpointModel, CustomEndpointSchema,
};
use mockito::{Matcher, Server};
use warpui::{App, SingletonEntity as _};
use warpui_extras::secure_storage;

use super::*;
use crate::auth::AuthStateProvider;
use crate::settings::{AISettings, init_and_register_user_preferences};
use crate::workspaces::user_workspaces::{TeamlessScopeForTest, UserWorkspaces};

fn block_on_tokio<F: Future>(fut: F) -> F::Output {
    tokio::runtime::Runtime::new().unwrap().block_on(fut)
}

/// Wide enough that only a stalled mock could reach it.
const TEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct InMemorySecureStorage {
    values: Mutex<HashMap<String, String>>,
}

impl secure_storage::SecureStorage for InMemorySecureStorage {
    fn write_value(&self, key: &str, value: &str) -> Result<(), secure_storage::Error> {
        match self.values.lock() {
            Ok(mut values) => {
                values.insert(key.to_owned(), value.to_owned());
                Ok(())
            }
            Err(err) => Err(secure_storage::Error::Unknown(anyhow::anyhow!(
                err.to_string()
            ))),
        }
    }

    fn read_value(&self, key: &str) -> Result<String, secure_storage::Error> {
        match self.values.lock() {
            Ok(values) => values
                .get(key)
                .cloned()
                .ok_or(secure_storage::Error::NotFound),
            Err(err) => Err(secure_storage::Error::Unknown(anyhow::anyhow!(
                err.to_string()
            ))),
        }
    }

    fn remove_value(&self, key: &str) -> Result<(), secure_storage::Error> {
        match self.values.lock() {
            Ok(mut values) => {
                values.remove(key);
                Ok(())
            }
            Err(err) => Err(secure_storage::Error::Unknown(anyhow::anyhow!(
                err.to_string()
            ))),
        }
    }
}

fn add_api_key_manager(app: &mut App) {
    app.update(init_and_register_user_preferences);
    app.add_singleton_model(AISettings::new_with_defaults);
    app.add_singleton_model(|_| AuthStateProvider::new_for_test());
    app.add_singleton_model(UserWorkspaces::default_mock);
    app.add_singleton_model(|_| -> secure_storage::Model {
        Box::new(InMemorySecureStorage::default())
    });
    app.add_singleton_model(ApiKeyManager::new);
}

fn configure_endpoint(app: &mut App, base_url: &str, key: Option<&str>, models: &[(&str, &str)]) {
    let id = CustomEndpointId::generated();
    let mut definitions = CustomEndpointDefinitions::default();
    definitions
        .insert(
            id.clone(),
            CustomEndpointDefinition {
                name: "Test".to_owned(),
                base_url: base_url.to_owned(),
                schema: CustomEndpointSchema::default(),
                models: models
                    .iter()
                    .map(|(name, config_key)| CustomEndpointModel {
                        name: (*name).to_owned(),
                        alias: None,
                        config_key: (*config_key).to_owned(),
                    })
                    .collect(),
            },
        )
        .expect("one valid endpoint definition");

    ApiKeyManager::handle(app).update(app, |manager, ctx| {
        manager.set_custom_endpoint_definitions(definitions, ctx);
        manager
            .persist_custom_endpoint_key(id, key.map(str::to_owned), ctx)
            .expect("endpoint key persists");
    });
}

const SINGLE_MODEL: &[(&str, &str)] = &[("test-model", "test-model-key")];

#[test]
fn resolve_picks_configured_endpoint_for_teamless_scope() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(true);
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            SINGLE_MODEL,
        );

        let resolved = app.read(|ctx| resolve(ctx, &TeamlessScopeForTest));

        let resolved = resolved.expect("configured endpoint resolves");
        assert_eq!(resolved.model(), "test-model");
        assert_eq!(resolved.endpoint_name(), "Test");
    });
}

#[test]
fn resolve_skips_endpoint_without_a_key() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(true);
        add_api_key_manager(&mut app);
        configure_endpoint(&mut app, "https://example.com/v1", None, SINGLE_MODEL);

        let resolved = app.read(|ctx| resolve(ctx, &TeamlessScopeForTest));

        assert!(
            resolved.is_none(),
            "keyless endpoints cannot serve requests"
        );
    });
}

#[test]
fn resolve_returns_none_when_feature_is_disabled() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(false);
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            SINGLE_MODEL,
        );

        let resolved = app.read(|ctx| resolve(ctx, &TeamlessScopeForTest));

        assert!(resolved.is_none(), "the flag is the kill switch");
    });
}

#[test]
fn resolve_uses_the_model_configured_for_autofill() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(true);
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            &[("slow-model", "slow-key"), ("fast-model", "fast-key")],
        );
        set_autofill_model(&mut app, Some("fast-key"));

        let resolved = app.read(|ctx| resolve(ctx, &TeamlessScopeForTest));

        let resolved = resolved.expect("endpoint resolves");
        assert_eq!(resolved.model(), "fast-model");
    });
}

#[test]
fn resolve_falls_back_to_first_model_when_configured_model_is_gone() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(true);
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            &[("first-model", "first-key"), ("second-model", "second-key")],
        );
        set_autofill_model(&mut app, Some("removed-key"));

        let resolved = app.read(|ctx| resolve(ctx, &TeamlessScopeForTest));

        let resolved = resolved.expect("endpoint resolves");
        assert_eq!(resolved.model(), "first-model");
    });
}

#[test]
fn selectable_models_lists_every_named_model() {
    App::test((), |mut app| async move {
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            &[("slow-model", "slow-key"), ("fast-model", "fast-key")],
        );

        let choices = app.read(|ctx| selectable_models(ctx, &TeamlessScopeForTest));

        let labels: Vec<(&str, &str)> = choices
            .iter()
            .map(|choice| (choice.endpoint_name.as_str(), choice.model_label.as_str()))
            .collect();
        assert_eq!(labels, vec![("Test", "slow-model"), ("Test", "fast-model")]);
    });
}

#[test]
fn resolve_for_team_uid_uses_the_personal_scope_when_no_team_is_selected() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(true);
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            SINGLE_MODEL,
        );

        let resolved = app.read(|ctx| resolve_for_team_uid(ctx, None));

        let resolved = resolved.expect("personal scope resolves the configured endpoint");
        assert_eq!(resolved.model(), "test-model");
    });
}

#[test]
fn resolve_for_team_uid_returns_none_when_feature_is_disabled() {
    App::test((), |mut app| async move {
        let _flag = FeatureFlag::LocalByoInference.override_enabled(false);
        add_api_key_manager(&mut app);
        configure_endpoint(
            &mut app,
            "https://example.com/v1",
            Some("secret"),
            SINGLE_MODEL,
        );

        let resolved = app.read(|ctx| resolve_for_team_uid(ctx, None));

        assert!(resolved.is_none());
    });
}

fn set_autofill_model(app: &mut App, config_key: Option<&str>) {
    let config_key = config_key.map(str::to_owned);
    app.update(|ctx| {
        AISettings::handle(ctx).update(ctx, |settings, ctx| {
            settings
                .byo_autofill_model
                .set_value(config_key, ctx)
                .expect("autofill model setting persists");
        });
    });
}

fn endpoint_for(server: &Server, schema: CustomEndpointSchema) -> ByoEndpoint {
    ByoEndpoint {
        endpoint: CustomEndpoint {
            name: "test endpoint".to_owned(),
            url: server.url(),
            api_key: "test-key".to_owned(),
            models: vec![CustomEndpointModel {
                name: "test-model".to_owned(),
                alias: None,
                config_key: "test-model-key".to_owned(),
            }],
            schema,
        },
        model: "test-model".to_owned(),
    }
}

#[test]
fn sends_chat_completions_request_with_bearer_auth_and_returns_content() {
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .match_body(Matcher::PartialJson(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ],
        })))
        .with_status(200)
        .with_body(r#"{"choices": [{"message": {"content": "git status"}}]}"#)
        .create();

    let text = block_on_tokio(complete(
        &endpoint_for(&server, CustomEndpointSchema::OpenaiChatCompletions),
        "sys",
        "usr",
        64,
        TEST_TIMEOUT,
    ))
    .expect("completion succeeds");

    assert_eq!(text, "git status");
    mock.assert();
}

#[test]
fn sends_messages_request_with_version_segment_and_api_key_header() {
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/v1/messages")
        .match_header("x-api-key", "test-key")
        .with_status(200)
        .with_body(r#"{"content": [{"type": "text", "text": "cargo test"}]}"#)
        .create();

    let text = block_on_tokio(complete(
        &endpoint_for(&server, CustomEndpointSchema::AnthropicMessages),
        "sys",
        "usr",
        64,
        TEST_TIMEOUT,
    ))
    .expect("completion succeeds");

    assert_eq!(text, "cargo test");
    mock.assert();
}

#[test]
fn surfaces_error_status_with_body_excerpt() {
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_body(r#"{"error": {"message": "invalid api key"}}"#)
        .create();

    let error = block_on_tokio(complete(
        &endpoint_for(&server, CustomEndpointSchema::OpenaiChatCompletions),
        "sys",
        "usr",
        64,
        TEST_TIMEOUT,
    ))
    .expect_err("non-success status is an error");

    assert!(
        error.to_string().contains("401") && error.to_string().contains("invalid api key"),
        "unexpected error: {error}"
    );
    mock.assert();
}

#[test]
fn appends_chat_completions_path_to_base_url() {
    assert_eq!(
        request_url(
            "https://example.com/v1",
            CustomEndpointSchema::OpenaiChatCompletions
        ),
        "https://example.com/v1/chat/completions"
    );
}

#[test]
fn keeps_path_already_present_in_configured_url() {
    assert_eq!(
        request_url(
            "https://example.com/v1/chat/completions",
            CustomEndpointSchema::OpenaiChatCompletions
        ),
        "https://example.com/v1/chat/completions"
    );
}

#[test]
fn adds_version_segment_for_messages_api_without_one() {
    assert_eq!(
        request_url(
            "https://api.anthropic.com",
            CustomEndpointSchema::AnthropicMessages
        ),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        request_url(
            "https://api.anthropic.com/v1/",
            CustomEndpointSchema::AnthropicMessages
        ),
        "https://api.anthropic.com/v1/messages"
    );
}

#[test]
fn extracts_json_object_through_markdown_fence_and_prose() {
    let raw = "Here you go:\n```json\n{\"commands\": [\"ls\"]}\n```\nEnjoy.";

    assert_eq!(extract_json_object(raw), Some("{\"commands\": [\"ls\"]}"));
}

#[test]
fn returns_none_when_no_complete_object_is_present() {
    assert_eq!(extract_json_object("no object here"), None);
    assert_eq!(extract_json_object("{ unterminated"), None);
}
