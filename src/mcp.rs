//! MCP server: tools, resources and update notifications.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::service::{RequestContext, SubscriptionContext};
use rmcp::{ErrorData as McpError, Peer, RoleServer, ServerHandler, schemars, tool, tool_handler, tool_router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::broadcast::error::RecvError;

use crate::app::{App, Caller};
use crate::config::Scope;
use crate::transcripts::{LiveEvent, Query};

pub const URI_LATEST: &str = "ts://transcripts/latest";
pub const URI_USERS: &str = "ts://users";
pub const URI_STATUS: &str = "ts://status";
const URI_EVENT_PREFIX: &str = "ts://transcripts/";

#[derive(Clone)]
pub struct VoiceMcp {
    app: Arc<App>,
    tool_router: ToolRouter<VoiceMcp>,
    /// Legacy `resources/subscribe` URIs of this session.
    subscribed: Arc<Mutex<HashSet<String>>>,
    forwarder_started: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SpeakArgs {
    /// Text to speak in the channel.
    pub text: String,
    /// Stop current playback and drop queued speech first. Default false.
    #[serde(default)]
    pub interrupt: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MessageArgs {
    /// Text message for the bot's current channel chat.
    pub text: String,
}

fn caller(ctx: &RequestContext<RoleServer>) -> Option<Caller> {
    ctx.extensions
        .get::<axum::http::request::Parts>()
        .and_then(|p| p.extensions.get::<Caller>())
        .cloned()
}

fn require(ctx: &RequestContext<RoleServer>, scope: Scope) -> Result<(), McpError> {
    match caller(ctx) {
        Some(c) if c.has(scope) => Ok(()),
        _ => Err(McpError::invalid_request(format!("token lacks '{scope:?}' scope"), None)),
    }
}

fn json_result(v: impl serde::Serialize) -> Result<CallToolResult, McpError> {
    let v = serde_json::to_value(v).map_err(|e| McpError::internal_error(e.to_string(), None))?;
    Ok(CallToolResult::success(vec![ContentBlock::text(serde_json::to_string_pretty(&v).unwrap_or_default())]))
}

fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(format!("{e:#}"), None)
}

#[tool_router]
impl VoiceMcp {
    pub fn new(app: Arc<App>) -> Self {
        Self {
            app,
            tool_router: Self::tool_router(),
            subscribed: Default::default(),
            forwarder_started: Default::default(),
        }
    }

    #[tool(description = "List the users in the bot's current TeamSpeak channel.")]
    async fn ts_list_users(&self, ctx: RequestContext<RoleServer>) -> Result<CallToolResult, McpError> {
        require(&ctx, Scope::Read)?;
        json_result(self.app.ts.list_users().await.map_err(internal)?)
    }

    #[tool(description = "Get bot status: connection, channel, playback, ASR mode and usage.")]
    async fn ts_get_status(&self, ctx: RequestContext<RoleServer>) -> Result<CallToolResult, McpError> {
        require(&ctx, Scope::Read)?;
        json_result(self.app.status().await)
    }

    #[tool(description = "Get final voice transcripts, oldest first. Use `after_seq` with the last seen `seq` to poll for new ones.")]
    async fn ts_get_transcripts(
        &self,
        Parameters(q): Parameters<Query>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        require(&ctx, Scope::Read)?;
        let events = self.app.store.query(&q);
        json_result(json!({ "events": events, "latest_seq": self.app.store.latest_seq() }))
    }

    #[tool(description = "Speak text in the TeamSpeak channel via TTS. Requests are queued; set interrupt=true to cut off current speech.")]
    async fn ts_speak(
        &self,
        Parameters(a): Parameters<SpeakArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        require(&ctx, Scope::Speak)?;
        let n = self.app.tts.speak(&a.text, a.interrupt).map_err(internal)?;
        json_result(json!({ "queued": true, "sentences": n }))
    }

    #[tool(description = "Stop the current voice playback and clear queued speech.")]
    async fn ts_stop_speaking(&self, ctx: RequestContext<RoleServer>) -> Result<CallToolResult, McpError> {
        require(&ctx, Scope::Speak)?;
        self.app.tts.stop().await.map_err(internal)?;
        json_result(json!({ "stopped": true }))
    }

    #[tool(description = "Send a text message to the bot's current TeamSpeak channel chat.")]
    async fn ts_send_message(
        &self,
        Parameters(a): Parameters<MessageArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        require(&ctx, Scope::Speak)?;
        if a.text.trim().is_empty() || a.text.len() > 1024 {
            return Err(McpError::invalid_params("text must be 1..=1024 bytes", None));
        }
        self.app.ts.send_channel_message(a.text).await.map_err(internal)?;
        json_result(json!({ "sent": true }))
    }
}

impl VoiceMcp {
    /// URIs affected by a live event.
    fn uris_for(ev: &LiveEvent) -> &'static [&'static str] {
        match ev {
            LiveEvent::Transcript(t) if t.kind == "transcript.final" => &[URI_LATEST, URI_STATUS],
            LiveEvent::UsersChanged => &[URI_USERS],
            _ => &[],
        }
    }

    /// Forward store events as `notifications/resources/updated` (legacy subscribe).
    fn start_forwarder(&self, peer: Peer<RoleServer>) {
        if self.forwarder_started.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let mut rx = self.app.store.live.subscribe();
        let subscribed = self.subscribed.clone();
        tokio::spawn(async move {
            loop {
                let ev = match rx.recv().await {
                    Ok(ev) => ev,
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => return,
                };
                for uri in Self::uris_for(&ev) {
                    if !subscribed.lock().unwrap().contains(*uri) {
                        continue;
                    }
                    if peer.notify_resource_updated(ResourceUpdatedNotificationParam::new(*uri)).await.is_err() {
                        return;
                    }
                }
            }
        });
    }

    fn resource(uri: &str, name: &str, desc: &str) -> Resource {
        Resource::new(uri, name).with_description(desc).with_mime_type("application/json")
    }
}

#[tool_handler]
impl ServerHandler for VoiceMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder().enable_tools().enable_resources().enable_resources_subscribe().build(),
        )
        .with_server_info(Implementation::from_build_env())
        .with_instructions(
            "TeamSpeak voice bridge. Speech of channel members is transcribed per speaker. \
             Subscribe to ts://transcripts/latest (or poll ts_get_transcripts with after_seq) \
             to receive new utterances; reply with ts_speak.",
        )
    }

    async fn list_resources(
        &self,
        _r: Option<PaginatedRequestParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        require(&ctx, Scope::Read)?;
        Ok(ListResourcesResult {
            resources: vec![
                Self::resource(URI_LATEST, "latest-transcripts", "The 50 most recent final transcripts. Updated on every new utterance."),
                Self::resource(URI_USERS, "channel-users", "Users in the bot's channel."),
                Self::resource(URI_STATUS, "bot-status", "Connection, playback, ASR and usage status."),
            ],
            ..Default::default()
        })
    }

    async fn list_resource_templates(
        &self,
        _r: Option<PaginatedRequestParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        require(&ctx, Scope::Read)?;
        Ok(ListResourceTemplatesResult {
            resource_templates: vec![
                ResourceTemplate::new("ts://transcripts/{event_id}", "transcript-event")
                    .with_description("A single transcript event by event_id, e.g. ts://transcripts/evt_12")
                    .with_mime_type("application/json"),
            ],
            ..Default::default()
        })
    }

    async fn read_resource(
        &self,
        r: ReadResourceRequestParams,
        ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        require(&ctx, Scope::Read)?;
        let uri = r.uri.as_str();
        let value = match uri {
            URI_LATEST => json!({
                "events": self.app.store.query(&Query { limit: 50, ..Default::default() }),
                "latest_seq": self.app.store.latest_seq(),
            }),
            URI_USERS => serde_json::to_value(self.app.ts.list_users().await.map_err(internal)?).unwrap_or_default(),
            URI_STATUS => serde_json::to_value(self.app.status().await).unwrap_or_default(),
            _ => match uri.strip_prefix(URI_EVENT_PREFIX).and_then(|id| self.app.store.get(id)) {
                Some(ev) => serde_json::to_value(ev).unwrap_or_default(),
                None => return Err(McpError::resource_not_found("resource_not_found", Some(json!({ "uri": uri })))),
            },
        };
        let text = serde_json::to_string_pretty(&value).unwrap_or_default();
        Ok(ReadResourceResult::new(vec![ResourceContents::text(text, uri)]).into())
    }

    #[allow(deprecated)]
    async fn subscribe(&self, r: SubscribeRequestParams, ctx: RequestContext<RoleServer>) -> Result<(), McpError> {
        require(&ctx, Scope::Read)?;
        self.subscribed.lock().unwrap().insert(r.uri);
        self.start_forwarder(ctx.peer.clone());
        Ok(())
    }

    #[allow(deprecated)]
    async fn unsubscribe(&self, r: UnsubscribeRequestParams, ctx: RequestContext<RoleServer>) -> Result<(), McpError> {
        require(&ctx, Scope::Read)?;
        self.subscribed.lock().unwrap().remove(&r.uri);
        Ok(())
    }

    /// 2026-07-28 `subscriptions/listen`.
    fn accepted_subscription_filter(&self, requested: &SubscriptionFilter) -> Option<SubscriptionFilter> {
        Some(requested.supported_by(&self.get_info().capabilities))
    }

    async fn listen(&self, context: SubscriptionContext) -> Result<(), McpError> {
        require(context.request_context(), Scope::Read)?;
        let accepted: HashSet<String> =
            context.accepted().resource_subscriptions.clone().unwrap_or_default().into_iter().collect();
        let mut rx = self.app.store.live.subscribe();
        loop {
            tokio::select! {
                () = context.cancelled() => return Ok(()),
                ev = rx.recv() => {
                    let ev = match ev {
                        Ok(ev) => ev,
                        Err(RecvError::Lagged(_)) => continue,
                        Err(RecvError::Closed) => return Ok(()),
                    };
                    for uri in Self::uris_for(&ev) {
                        if accepted.contains(*uri) {
                            context.sink().notify_resource_updated(*uri).await
                                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                        }
                    }
                }
            }
        }
    }
}
