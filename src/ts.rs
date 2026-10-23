//! TeamSpeak connection actor.
//!
//! Owns the `tsclientlib::Connection`, splits incoming voice per client before
//! mixing, and paces outgoing Opus frames. Other components talk to it through
//! [`TsHandle`] and receive per-user audio as [`TrackEvent`]s.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use time::OffsetDateTime;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::{MissedTickBehavior, interval, sleep};
use tracing::{debug, error, info, warn};

use tsclientlib::audio::AudioHandler;
use tsclientlib::events::Event;
use tsclientlib::prelude::*;
use tsclientlib::{ClientId, Connection, DisconnectOptions, Identity, MessageTarget, StreamItem};
use tsproto_packets::packets::{AudioData, CodecType, OutAudio};

use crate::audio::{Downsampler, OpusFramer, TS_FRAME};
use crate::config::TeamSpeakConfig;

/// Identity and location of a speaker, captured when an utterance starts.
#[derive(Debug, Clone)]
pub struct SpeakerInfo {
    pub client_id: u16,
    pub uid: Option<String>,
    pub database_id: Option<u64>,
    pub name: String,
    pub channel_id: u64,
    pub channel_name: String,
}

/// Per-user audio, already split before mixing. Audio is 16 kHz mono f32.
#[derive(Debug)]
pub enum TrackEvent {
    Start {
        speaker: SpeakerInfo,
        at: OffsetDateTime,
    },
    Audio {
        client_id: u16,
        samples: Vec<f32>,
    },
    End {
        client_id: u16,
        at: OffsetDateTime,
    },
    /// Channel membership or names changed.
    UsersChanged,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UserInfo {
    pub client_id: u16,
    pub uid: Option<String>,
    pub name: String,
    pub channel_id: u64,
    pub is_self: bool,
    pub talking: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BotStatus {
    pub connected: bool,
    pub own_client_id: Option<u16>,
    pub channel_id: Option<u64>,
    pub channel_name: Option<String>,
    pub playing: bool,
    pub queued_ms: u64,
    pub active_speakers: usize,
}

enum Command {
    /// 48 kHz mono PCM to append to the playback queue.
    Play(Vec<f32>),
    StopPlayback,
    SendChannelMessage(String, oneshot::Sender<Result<()>>),
    ListUsers(oneshot::Sender<Result<Vec<UserInfo>>>),
    Status(oneshot::Sender<BotStatus>),
    Shutdown,
}

#[derive(Clone)]
pub struct TsHandle {
    tx: mpsc::Sender<Command>,
    playing: watch::Receiver<bool>,
}

impl TsHandle {
    pub async fn play(&self, pcm48k: Vec<f32>) -> Result<()> {
        self.send(Command::Play(pcm48k)).await
    }
    pub async fn stop_playback(&self) -> Result<()> {
        self.send(Command::StopPlayback).await
    }
    pub async fn send_channel_message(&self, text: String) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::SendChannelMessage(text, tx)).await?;
        rx.await.context("ts actor gone")?
    }
    pub async fn list_users(&self) -> Result<Vec<UserInfo>> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::ListUsers(tx)).await?;
        rx.await.context("ts actor gone")?
    }
    pub async fn status(&self) -> Result<BotStatus> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Status(tx)).await?;
        rx.await.context("ts actor gone")
    }
    /// True while the bot is sending voice.
    pub fn is_playing(&self) -> bool {
        *self.playing.borrow()
    }
    /// Wait until the playback queue is drained.
    pub async fn wait_idle(&self) {
        let mut rx = self.playing.clone();
        // Playback starts on the next 20 ms tick after queueing.
        tokio::time::sleep(Duration::from_millis(60)).await;
        let _ = rx.wait_for(|p| !*p).await;
    }
    pub async fn shutdown(&self) -> Result<()> {
        self.send(Command::Shutdown).await
    }
    async fn send(&self, cmd: Command) -> Result<()> {
        self.tx
            .send(cmd)
            .await
            .map_err(|_| anyhow::anyhow!("ts actor gone"))
    }
}

/// Spawn the TeamSpeak actor. Returns a handle and the stream of per-user audio.
pub fn spawn(
    cfg: TeamSpeakConfig,
) -> Result<(
    TsHandle,
    mpsc::Receiver<TrackEvent>,
    tokio::task::JoinHandle<()>,
)> {
    let identity = load_or_create_identity(&cfg.identity_file)?;
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (track_tx, track_rx) = mpsc::channel(1024);
    let (playing_tx, playing_rx) = watch::channel(false);
    let task = tokio::spawn(async move {
        let mut actor = Actor::new(cfg, identity, cmd_rx, track_tx, playing_tx);
        actor.run().await;
    });
    Ok((
        TsHandle {
            tx: cmd_tx,
            playing: playing_rx,
        },
        track_rx,
        task,
    ))
}

fn load_or_create_identity(path: &Path) -> Result<Identity> {
    if path.exists() {
        let text = std::fs::read_to_string(path)?;
        let id: Identity = serde_json::from_str(&text)
            .with_context(|| format!("parsing identity {}", path.display()))?;
        info!(
            level = id.level(),
            "loaded identity from {}",
            path.display()
        );
        return Ok(id);
    }
    info!("creating new identity at {}", path.display());
    let id = Identity::create();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    write_private(path, &serde_json::to_string_pretty(&id)?)?;
    Ok(id)
}

#[cfg(unix)]
fn write_private(path: &Path, content: &str) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(content.as_bytes())?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &Path, content: &str) -> Result<()> {
    std::fs::write(path, content)?;
    Ok(())
}

/// Receive-side state for one talking client.
struct Track {
    down: Downsampler,
    out: Vec<f32>,
}

enum Outcome {
    /// Connection ended; reconnect after backoff.
    Reconnect,
    /// Shut down requested.
    Quit,
}

struct Actor {
    cfg: TeamSpeakConfig,
    identity: Identity,
    cmd_rx: mpsc::Receiver<Command>,
    track_tx: mpsc::Sender<TrackEvent>,
    // Receive side
    audio: AudioHandler<ClientId>,
    tracks: HashMap<ClientId, Track>,
    mix_buf: Vec<f32>,
    // Send side
    framer: OpusFramer,
    play_queue: VecDeque<Vec<u8>>,
    playing: bool,
    playing_tx: watch::Sender<bool>,
    // Connection state
    connected: bool,
}

impl Actor {
    fn new(
        cfg: TeamSpeakConfig,
        identity: Identity,
        cmd_rx: mpsc::Receiver<Command>,
        track_tx: mpsc::Sender<TrackEvent>,
        playing_tx: watch::Sender<bool>,
    ) -> Self {
        Self {
            cfg,
            identity,
            cmd_rx,
            track_tx,
            audio: AudioHandler::new(),
            tracks: HashMap::new(),
            mix_buf: vec![0.0; TS_FRAME * 2],
            framer: OpusFramer::new().expect("opus encoder"),
            play_queue: VecDeque::new(),
            playing: false,
            playing_tx,
            connected: false,
        }
    }

    fn set_playing(&mut self, v: bool) {
        self.playing = v;
        self.playing_tx
            .send_if_modified(|p| std::mem::replace(p, v) != v);
    }

    fn is_ignored(&self, con: &Connection, id: ClientId) -> bool {
        if self.cfg.ignore_clients.is_empty() {
            return false;
        }
        let Ok(state) = con.get_state() else {
            return false;
        };
        let Some(c) = state.clients.get(&id) else {
            return false;
        };
        let uid = c.uid.as_ref().map(|u| u.to_string());
        self.cfg
            .ignore_clients
            .iter()
            .any(|x| *x == c.name || Some(x) == uid.as_ref())
    }

    async fn run(&mut self) {
        let mut backoff = Duration::from_secs(2);
        loop {
            let res = self.connect_and_serve().await;
            if self.connected {
                backoff = Duration::from_secs(2);
            }
            match res {
                Ok(Outcome::Quit) => return,
                Ok(Outcome::Reconnect) => {}
                Err(e) => error!("teamspeak connection error: {e:#}"),
            }
            self.on_disconnected();
            warn!("reconnecting in {:?}", backoff);
            // Keep answering commands while waiting.
            let deadline = sleep(backoff);
            tokio::pin!(deadline);
            loop {
                tokio::select! {
                    _ = &mut deadline => break,
                    cmd = self.cmd_rx.recv() => match cmd {
                        None | Some(Command::Shutdown) => return,
                        Some(cmd) => self.handle_offline_command(cmd),
                    },
                }
            }
            backoff = (backoff * 2).min(Duration::from_secs(60));
        }
    }

    fn connect_options(&self) -> tsclientlib::ConnectOptions {
        let c = &self.cfg;
        let mut opts = Connection::build(c.address.clone())
            .name(c.nickname.clone())
            .identity(self.identity.clone())
            .log_commands(c.log_commands);
        if let Some(id) = c.channel_id {
            opts = opts.channel_id(tsclientlib::ChannelId(id));
        } else if let Some(ch) = &c.channel {
            opts = opts.channel(ch.clone());
        }
        if let Some(p) = &c.channel_password {
            opts = opts.channel_password(p.clone());
        }
        if let Some(p) = &c.server_password {
            opts = opts.password(p.clone());
        }
        opts
    }

    async fn connect_and_serve(&mut self) -> Result<Outcome> {
        info!(address = %self.cfg.address, "connecting");
        let mut con = self.connect_options().connect()?;
        let mut tick = interval(Duration::from_millis(20));
        tick.set_missed_tick_behavior(MissedTickBehavior::Burst);

        loop {
            enum Step {
                Event(Option<Result<StreamItem, tsclientlib::Error>>),
                Tick,
                Cmd(Option<Command>),
            }
            let step = {
                let mut events = con.events();
                tokio::select! {
                    ev = events.next() => Step::Event(ev),
                    _ = tick.tick() => Step::Tick,
                    cmd = self.cmd_rx.recv() => Step::Cmd(cmd),
                }
            };
            match step {
                Step::Event(None) => return Ok(Outcome::Reconnect),
                Step::Event(Some(Err(e))) => return Err(e.into()),
                Step::Event(Some(Ok(item))) => self.handle_item(&mut con, item),
                Step::Tick => self.on_tick(&mut con).await,
                Step::Cmd(None) | Step::Cmd(Some(Command::Shutdown)) => {
                    self.disconnect(con).await;
                    return Ok(Outcome::Quit);
                }
                Step::Cmd(Some(cmd)) => self.handle_command(&mut con, cmd),
            }
        }
    }

    async fn disconnect(&mut self, mut con: Connection) {
        // tsclientlib panics if disconnect is driven while the initial handshake
        // is still in progress. In that case simply drop the connection.
        if !self.connected {
            info!("dropping unfinished connection");
            drop(con);
            self.on_disconnected();
            return;
        }
        info!("disconnecting");
        if con.disconnect(DisconnectOptions::new()).is_ok() {
            // Do not use `for_each` here: a busy audio stream can keep it
            // ready forever and starve the timeout. Select the deadline on
            // every event instead.
            let mut events = con.events();
            let deadline = sleep(Duration::from_secs(3));
            tokio::pin!(deadline);
            loop {
                tokio::select! {
                    _ = &mut deadline => break,
                    item = events.next() => {
                        if item.is_none() {
                            break;
                        }
                    }
                }
            }
        }
        drop(con);
        self.on_disconnected();
    }

    fn handle_item(&mut self, con: &mut Connection, item: StreamItem) {
        match item {
            StreamItem::BookEvents(events) => {
                if !self.connected {
                    self.connected = true;
                    self.on_connected(con);
                }
                let mut users_changed = false;
                for e in events {
                    users_changed |= self.log_book_event(con, &e);
                }
                if users_changed {
                    self.emit(TrackEvent::UsersChanged);
                }
            }
            StreamItem::Audio(packet) => {
                let from = match packet.data().data() {
                    AudioData::S2C { from, .. } | AudioData::S2CWhisper { from, .. } => {
                        ClientId(*from)
                    }
                    _ => return,
                };
                match self.audio.handle_packet(from, packet) {
                    Ok(Some(id)) if self.is_ignored(con, id) => {
                        self.audio.get_mut_queues().remove(&id);
                    }
                    Ok(Some(id)) => self.start_track(con, id),
                    Ok(None) => {}
                    Err(e) => debug!(client = from.0, "audio packet dropped: {e}"),
                }
            }
            StreamItem::DisconnectedTemporarily(reason) => {
                warn!(?reason, "temporarily disconnected, library is reconnecting");
                self.on_disconnected();
            }
            StreamItem::AudioChange(change) => debug!(?change, "audio permission change"),
            StreamItem::MessageResult(_, Err(e)) => warn!("command failed: {e:?}"),
            other => debug!(item = ?std::mem::discriminant(&other), "ignored stream item"),
        }
    }

    fn on_connected(&mut self, con: &mut Connection) {
        let Ok(state) = con.get_state() else { return };
        let own = state.own_client;
        let channel = state.clients.get(&own).map(|c| c.channel);
        let channel_name = channel
            .and_then(|id| state.channels.get(&id))
            .map(|c| c.name.clone())
            .unwrap_or_default();
        info!(
            client_id = own.0,
            channel = channel.map(|c| c.0),
            %channel_name,
            server = %state.server.name,
            "connected"
        );
        if !self.cfg.announcement.is_empty() {
            let msg = state.send_message(MessageTarget::Channel, &self.cfg.announcement);
            if let Err(e) = msg.send(con) {
                warn!("failed to send announcement: {e}");
            }
        }
    }

    fn on_disconnected(&mut self) {
        self.connected = false;
        let ids: Vec<ClientId> = self.tracks.keys().copied().collect();
        for id in ids {
            self.end_track(id);
        }
        self.audio.reset();
        self.play_queue.clear();
        self.set_playing(false);
    }

    /// Logs a book event; returns true if channel membership may have changed.
    fn log_book_event(&self, con: &Connection, e: &Event) -> bool {
        use tsclientlib::events::PropertyId;
        match e {
            Event::Message {
                target,
                invoker,
                message,
            } => {
                info!(?target, from = %invoker.name, "text message: {message}");
                false
            }
            Event::PropertyAdded {
                id: PropertyId::Client(id),
                ..
            } => {
                if let Ok(state) = con.get_state()
                    && let Some(c) = state.clients.get(id)
                {
                    info!(client_id = id.0, name = %c.name, channel = c.channel.0, "client appeared");
                }
                true
            }
            Event::PropertyRemoved {
                id: PropertyId::Client(id),
                ..
            } => {
                info!(client_id = id.0, "client left");
                true
            }
            Event::PropertyChanged {
                id: PropertyId::ClientChannel(id),
                ..
            } => {
                if let Ok(state) = con.get_state()
                    && let Some(c) = state.clients.get(id)
                {
                    info!(client_id = id.0, name = %c.name, channel = c.channel.0, "client moved");
                }
                true
            }
            Event::PropertyChanged {
                id: PropertyId::ClientName(_),
                ..
            } => true,
            _ => false,
        }
    }

    fn speaker_info(con: &Connection, id: ClientId) -> SpeakerInfo {
        let mut info = SpeakerInfo {
            client_id: id.0,
            uid: None,
            database_id: None,
            name: format!("client-{}", id.0),
            channel_id: 0,
            channel_name: String::new(),
        };
        if let Ok(state) = con.get_state()
            && let Some(c) = state.clients.get(&id)
        {
            info.uid = c.uid.as_ref().map(|u| u.to_string());
            info.database_id = Some(c.database_id.0);
            info.name = c.name.clone();
            info.channel_id = c.channel.0;
            info.channel_name = state
                .channels
                .get(&c.channel)
                .map(|ch| ch.name.clone())
                .unwrap_or_default();
        }
        info
    }

    fn start_track(&mut self, con: &Connection, id: ClientId) {
        if self.tracks.contains_key(&id) {
            return;
        }
        let down = match Downsampler::new() {
            Ok(d) => d,
            Err(e) => {
                error!("resampler init failed: {e:#}");
                return;
            }
        };
        let speaker = Self::speaker_info(con, id);
        debug!(client_id = id.0, name = %speaker.name, "speaker started");
        self.tracks.insert(
            id,
            Track {
                down,
                out: Vec::new(),
            },
        );
        self.emit(TrackEvent::Start {
            speaker,
            at: OffsetDateTime::now_utc(),
        });
    }

    fn end_track(&mut self, id: ClientId) {
        if let Some(mut t) = self.tracks.remove(&id) {
            if let Err(e) = t.down.flush(&mut t.out) {
                warn!("resampler flush failed: {e:#}");
            }
            if !t.out.is_empty() {
                self.emit(TrackEvent::Audio {
                    client_id: id.0,
                    samples: t.out,
                });
            }
            self.emit(TrackEvent::End {
                client_id: id.0,
                at: OffsetDateTime::now_utc(),
            });
        }
    }

    fn emit(&self, ev: TrackEvent) {
        if let Err(mpsc::error::TrySendError::Full(_)) = self.track_tx.try_send(ev) {
            warn!("track consumer is lagging, dropping audio");
        }
    }

    async fn on_tick(&mut self, con: &mut Connection) {
        // Receive: pull 20 ms per active client, before mixing.
        self.mix_buf.iter_mut().for_each(|s| *s = 0.0);
        let tracks = &mut self.tracks;
        let ended = self
            .audio
            .fill_buffer_with_proc(&mut self.mix_buf, |id, stereo| {
                if let Some(t) = tracks.get_mut(id) {
                    let mono: Vec<f32> = stereo
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|[l, r]| (l + r) * 0.5)
                        .collect();
                    if let Err(e) = t.down.push(&mono, &mut t.out) {
                        warn!("resample failed: {e:#}");
                    }
                }
            });
        let chunks: Vec<(ClientId, Vec<f32>)> = self
            .tracks
            .iter_mut()
            .filter(|(_, t)| !t.out.is_empty())
            .map(|(id, t)| (*id, std::mem::take(&mut t.out)))
            .collect();
        for (id, samples) in chunks {
            self.emit(TrackEvent::Audio {
                client_id: id.0,
                samples,
            });
        }
        for id in ended {
            self.end_track(id);
        }

        // Send: one Opus frame per tick.
        if let Some(frame) = self.play_queue.pop_front() {
            self.set_playing(true);
            let pkt = OutAudio::new(&AudioData::C2S {
                id: 0,
                codec: CodecType::OpusVoice,
                data: &frame,
            });
            if let Err(e) = con.send_audio(pkt) {
                warn!("send_audio failed: {e}");
            }
        } else if self.playing {
            self.set_playing(false);
            self.send_audio_end(con);
        }
    }

    fn send_audio_end(&self, con: &mut Connection) {
        let pkt = OutAudio::new(&AudioData::C2S {
            id: 0,
            codec: CodecType::OpusVoice,
            data: &[],
        });
        let _ = con.send_audio(pkt);
    }

    fn handle_command(&mut self, con: &mut Connection, cmd: Command) {
        match cmd {
            Command::Play(pcm) => {
                let queue = &mut self.play_queue;
                let res = self.framer.push(&pcm, |f| {
                    queue.push_back(f);
                    Ok(())
                });
                let res = res.and_then(|_| {
                    self.framer.finish(|f| {
                        queue.push_back(f);
                        Ok(())
                    })
                });
                if let Err(e) = res {
                    warn!("opus encode failed: {e:#}");
                }
            }
            Command::StopPlayback => {
                self.play_queue.clear();
                if self.playing {
                    self.set_playing(false);
                    self.send_audio_end(con);
                }
            }
            Command::SendChannelMessage(text, reply) => {
                let res = (|| -> Result<()> {
                    let cmd = con.get_state()?.send_message(MessageTarget::Channel, &text);
                    cmd.send(con)?;
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            Command::ListUsers(reply) => {
                let _ = reply.send(self.list_users(con));
            }
            Command::Status(reply) => {
                let _ = reply.send(self.status(Some(con)));
            }
            Command::Shutdown => unreachable!("handled by caller"),
        }
    }

    fn handle_offline_command(&mut self, cmd: Command) {
        match cmd {
            Command::Play(_) | Command::StopPlayback => {}
            Command::SendChannelMessage(_, r) => {
                let _ = r.send(Err(anyhow::anyhow!("not connected")));
            }
            Command::ListUsers(r) => {
                let _ = r.send(Err(anyhow::anyhow!("not connected")));
            }
            Command::Status(r) => {
                let _ = r.send(self.status(None));
            }
            Command::Shutdown => {}
        }
    }

    fn list_users(&self, con: &Connection) -> Result<Vec<UserInfo>> {
        let state = con.get_state()?;
        let own = state.own_client;
        let own_channel = state.clients.get(&own).map(|c| c.channel);
        let mut users: Vec<UserInfo> = state
            .clients
            .values()
            .filter(|c| Some(c.channel) == own_channel)
            .map(|c| UserInfo {
                client_id: c.id.0,
                uid: c.uid.as_ref().map(|u| u.to_string()),
                name: c.name.clone(),
                channel_id: c.channel.0,
                is_self: c.id == own,
                talking: self.tracks.contains_key(&c.id),
            })
            .collect();
        users.sort_by_key(|u| u.client_id);
        Ok(users)
    }

    fn status(&self, con: Option<&Connection>) -> BotStatus {
        let state = con.and_then(|c| c.get_state().ok());
        let own = state.map(|s| s.own_client);
        let channel = state
            .zip(own)
            .and_then(|(s, o)| s.clients.get(&o))
            .map(|c| c.channel);
        let channel_name = state
            .zip(channel)
            .and_then(|(s, ch)| s.channels.get(&ch))
            .map(|c| c.name.clone());
        BotStatus {
            connected: self.connected && state.is_some(),
            own_client_id: own.map(|o| o.0),
            channel_id: channel.map(|c| c.0),
            channel_name,
            playing: self.playing,
            queued_ms: self.play_queue.len() as u64 * 20,
            active_speakers: self.tracks.len(),
        }
    }
}
