//! HTTP server: control panel (`/`), REST + SSE (`/api`), MCP (`/mcp`), one port.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::extract::{Query as QueryParams, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use futures::Stream;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::app::{App, Caller};
use crate::config::{AsrMode, Scope, ServerConfig};
use crate::mcp::VoiceMcp;
use crate::transcripts::Query;

const PANEL: &str = include_str!("panel.html");

pub async fn serve(app: Arc<App>, cfg: ServerConfig, shutdown: CancellationToken) -> Result<()> {
    let mut hosts: Vec<String> = vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
    hosts.extend(cfg.allowed_hosts.iter().cloned());
    let mcp_app = app.clone();
    let mcp = StreamableHttpService::new(
        move || Ok(VoiceMcp::new(mcp_app.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default()
            .with_allowed_hosts(hosts)
            .with_sse_keep_alive(Some(Duration::from_secs(15)))
            .with_cancellation_token(shutdown.child_token()),
    );

    let api = Router::new()
        .route("/status", get(status))
        .route("/users", get(users))
        .route("/transcripts", get(transcripts))
        .route("/events", get(events))
        .route("/speak", post(speak))
        .route("/stop", post(stop))
        .route("/message", post(message))
        .route("/asr", post(set_asr));

    let protected = Router::new()
        .nest("/api", api)
        .nest_service("/mcp", mcp)
        .layer(middleware::from_fn_with_state(app.clone(), auth));

    let router = Router::new()
        .route("/", get(|| async { Html(PANEL) }))
        .route("/healthz", get(|| async { "ok" }))
        .merge(protected)
        .with_state(app);

    let listener = tokio::net::TcpListener::bind(&cfg.bind).await?;
    info!("control panel http://{}/  ·  MCP endpoint http://{}/mcp", cfg.bind, cfg.bind);
    axum::serve(listener, router)
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await?;
    Ok(())
}

/// Bearer token from `Authorization`, or `?token=` (for EventSource, which cannot set headers).
fn extract_token(headers: &HeaderMap, query: Option<&str>) -> Option<String> {
    if let Some(v) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok())
        && let Some(t) = v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer "))
    {
        return Some(t.trim().to_string());
    }
    form_urlencoded::parse(query?.as_bytes()).find(|(k, _)| k == "token").map(|(_, v)| v.into_owned())
}

async fn auth(State(app): State<Arc<App>>, mut req: Request, next: Next) -> Response {
    let token = extract_token(req.headers(), req.uri().query());
    match app.authenticate(token.as_deref()) {
        Some(caller) => {
            req.extensions_mut().insert(caller);
            next.run(req).await
        }
        None => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(json!({"error": "missing or invalid bearer token"})),
        )
            .into_response(),
    }
}

type ApiResult = Result<Response, (StatusCode, Json<serde_json::Value>)>;

fn need(c: &Caller, scope: Scope) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if c.has(scope) {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, Json(json!({"error": format!("token lacks '{scope:?}' scope")}))))
    }
}

fn fail(e: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e:#}")})))
}

async fn status(State(app): State<Arc<App>>, Extension(c): Extension<Caller>) -> ApiResult {
    need(&c, Scope::Read)?;
    let mut v = serde_json::to_value(app.status().await).unwrap_or_default();
    v["caller"] = json!({"name": c.name, "scopes": c.scopes});
    Ok(Json(v).into_response())
}

async fn users(State(app): State<Arc<App>>, Extension(c): Extension<Caller>) -> ApiResult {
    need(&c, Scope::Read)?;
    Ok(Json(app.ts.list_users().await.map_err(fail)?).into_response())
}

async fn transcripts(
    State(app): State<Arc<App>>,
    Extension(c): Extension<Caller>,
    QueryParams(q): QueryParams<Query>,
) -> ApiResult {
    need(&c, Scope::Read)?;
    Ok(Json(json!({"events": app.store.query(&q), "latest_seq": app.store.latest_seq()})).into_response())
}

async fn events(
    State(app): State<Arc<App>>,
    Extension(c): Extension<Caller>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<serde_json::Value>)> {
    need(&c, Scope::Read)?;
    let stream = BroadcastStream::new(app.store.live.subscribe()).filter_map(|ev| {
        let ev = ev.ok()?;
        Some(Ok(Event::default().json_data(ev).unwrap_or_default()))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
struct SpeakBody {
    text: String,
    #[serde(default)]
    interrupt: bool,
}

async fn speak(State(app): State<Arc<App>>, Extension(c): Extension<Caller>, Json(b): Json<SpeakBody>) -> ApiResult {
    need(&c, Scope::Speak)?;
    let n = app.tts.speak(&b.text, b.interrupt).map_err(fail)?;
    Ok(Json(json!({"queued": true, "sentences": n})).into_response())
}

async fn stop(State(app): State<Arc<App>>, Extension(c): Extension<Caller>) -> ApiResult {
    need(&c, Scope::Speak)?;
    app.tts.stop().await.map_err(fail)?;
    Ok(Json(json!({"stopped": true})).into_response())
}

#[derive(Deserialize)]
struct MessageBody {
    text: String,
}

async fn message(State(app): State<Arc<App>>, Extension(c): Extension<Caller>, Json(b): Json<MessageBody>) -> ApiResult {
    need(&c, Scope::Speak)?;
    if b.text.trim().is_empty() || b.text.len() > 1024 {
        return Err(fail(anyhow::anyhow!("text must be 1..=1024 bytes")));
    }
    app.ts.send_channel_message(b.text).await.map_err(fail)?;
    Ok(Json(json!({"sent": true})).into_response())
}

#[derive(Deserialize)]
struct AsrBody {
    enabled: Option<bool>,
    mode: Option<AsrMode>,
    #[serde(default)]
    clear_block: bool,
}

async fn set_asr(State(app): State<Arc<App>>, Extension(c): Extension<Caller>, Json(b): Json<AsrBody>) -> ApiResult {
    need(&c, Scope::Admin)?;
    app.asr.set(b.enabled, b.mode, b.clear_block);
    info!(by = %c.name, enabled = ?b.enabled, mode = ?b.mode, "ASR settings changed");
    Ok(Json(app.asr.state()).into_response())
}
