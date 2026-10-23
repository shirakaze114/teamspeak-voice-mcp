# tsvoice_mcp

一个运行在 Headless Linux VPS 上的 TeamSpeak 3 Voice Bot：把频道里**每个人的独立语音**转成文字，通过 MCP 提供给已有的 Agent Harness，并通过 MCP Tools 在频道里发文字或语音。

本项目不包含 Agent Harness、LLM 调度、Prompt 管理或长期记忆。

## 已实现

- TeamSpeak 3 连接、指定频道加入、持久化身份、断线重连。
- 在混音前按 ClientId 分离音频，不使用混合音频做主要的说话人分离。
- 每个用户独立的 Silero VAD：过滤短噪声、合并短暂停顿、减少发送给 ASR 的静音。
- 两种百炼 ASR：
  - `qwen-audio-3.0-asr-flash`：HTTP 非实时模式，价格较低，话说完后返回最终结果。
  - `qwen-audio-3.0-asr-flash-streaming`：WebSocket 流式模式，说话过程中返回中间结果。
- 本地用量统计、每日/累计硬上限、免费额度耗尽后的封锁保护。
- MCP Streamable HTTP、Bearer Token 权限、Tools、Resources 和资源更新通知。
- 与 MCP **共用同一个端口**的 Web 控制台：成员状态、实时转写、用量、ASR 切换、播放控制。
- 本地命令 TTS 和 OpenAI 兼容 TTS；播放队列、打断、停止、重采样和 Opus 回传。
- 转写保留策略、可选 JSONL 持久化、调试 WAV 录音。

已在实际 TS3 服务器上验证连接、按人分轨、HTTP ASR、streaming ASR 中间结果，以及 macOS 本地 `say` TTS。Linux 上只需把本地 TTS 命令换成 `espeak-ng`、Piper 等。

## 数据流

```text
TeamSpeak 3
    ↓ Opus 语音包（带 ClientId）
按用户解码、保留独立音轨
    ↓ 48 kHz PCM → 16 kHz mono
每个用户独立的 Silero VAD
    ↓ 只发送有效语音段
qwen-audio-3.0-asr-flash（HTTP）
或 qwen-audio-3.0-asr-flash-streaming（WebSocket）
    ↓ transcript.partial / transcript.final
MCP Resources / Tools / SSE / Web 控制台
    ↕ Existing Agent Harness
MCP ts_speak
    ↓ 本地命令 TTS 或 OpenAI 兼容 TTS
重采样 → Opus 20 ms 帧 → TeamSpeak 3
```

## 环境要求

- Rust 1.88+（项目使用 edition 2024）。
- Linux VPS 无需图形界面或虚拟声卡。
- 构建 Opus 时需要 `cmake`，或者 `pkg-config` 加系统 libopus。
- Silero 模型由 `silero` crate 捆绑，不需要单独下载模型。
- 一个 TeamSpeak 3 服务器和可加入频道的身份。
- 使用云端 ASR 时，需要北京地域的百炼 API Key；免费额度也需要 API Key。

## 快速开始

### 1. 准备配置

```bash
cp config.example.toml config.toml
chmod 600 config.toml
```

编辑 `config.toml`：

- `teamspeak.address`：服务器地址。
- `teamspeak.channel_id`：出生频道 ID；当前测试配置使用 `458`。
- `server.tokens`：控制台和 MCP 的访问 Token。
- `asr`：ASR 模式和 Key 来源。
- `tts`：本地命令或 OpenAI 兼容服务。

`config.toml` 已被 `.gitignore` 忽略，不要提交真实密码、Token 或 API Key。

### 2. 配置百炼 Key

推荐使用单独的本机文件：

```bash
printf '%s' '你的百炼 API Key' > dashscope-key
chmod 600 dashscope-key
```

配置：

```toml
[asr]
enabled = true
mode = "http"
api_key_file = "dashscope-key"
```

`dashscope-key` 已加入 `.gitignore`。Key 来源优先级为：显式 `api_key` > `api_key_file` > `DASHSCOPE_API_KEY`。也可以使用环境变量：

```bash
export DASHSCOPE_API_KEY='你的百炼 API Key'
```

程序只在启动时读取 Key，不会把 Key 写入日志。

### 3. 配置 TeamSpeak 密码

密码不写入仓库，启动时传入：

```bash
export TS_SERVER_PASSWORD='你的服务器密码'
# 如果频道有密码：
export TS_CHANNEL_PASSWORD='你的频道密码'
```

### 4. 启动

```bash
cargo run --release -- -c config.toml
```

生产环境建议直接运行编译后的二进制：

```bash
cargo build --release
TS_SERVER_PASSWORD='你的服务器密码' \
  ./target/release/tsvoice_mcp -c config.toml
```

启动后：

- Web 控制台：`http://127.0.0.1:8787/`
- MCP：`http://127.0.0.1:8787/mcp`

`Ctrl-C` 会停止 Bot；TS 和 HTTP 关闭等待都有时间上限。日常运行不需要 `--console`，面板和 HTTP API 已足够调试。

## Web 控制台

打开 `http://127.0.0.1:8787/`，输入 `config.toml` 中 `[[server.tokens]]` 的 Token。

控制台功能：

- Bot 连接状态、当前频道、播放队列。
- 频道成员及正在说话的用户。
- 每个用户独立的转写记录。
- **最新转写在顶部，旧记录向下排列**；中间结果更新同一条记录。
- ASR 启用开关、HTTP/streaming 切换、封锁解除。
- 今日/累计音频秒数、请求次数、失败次数和估算原价。
- 文字发言、TTS 发言、打断和停止播放。

HTTP 非实时模式只显示最终结果；要看逐字中间结果，在 ASR 区域切换到 `streaming`。streaming 价格更高，测试完可以切回 `http`。

如果服务在远程 VPS 上，不要直接把无 TLS 的面板暴露到公网，可以使用 SSH 转发：

```bash
ssh -L 8787:127.0.0.1:8787 user@your-vps
```

然后打开本机的 `http://127.0.0.1:8787/`。

## ASR 配置

### HTTP 非实时模式（默认、便宜）

```toml
[asr]
mode = "http"
http_model = "qwen-audio-3.0-asr-flash"
```

VAD 判定一段话说完后，程序把这段 16 kHz WAV 通过一次 HTTP 请求提交，得到最终文字。适合省钱和不需要中间结果的场景。

### streaming 流式模式（有中间结果）

```toml
[asr]
mode = "streaming"
streaming_model = "qwen-audio-3.0-asr-flash-streaming"
max_sentence_silence_ms = 2000
```

说话过程中建立 WebSocket，发送 16 kHz PCM 二进制帧，接收 `transcript.partial`，结束后接收 `transcript.final`。每个用户有独立的识别任务。

### 常用选项

```toml
[asr]
language_hints = ["zh"]
# hotwords = { "TeamSpeak" = 3, "百炼" = 4 }
context_messages = 0
request_timeout_s = 30

[asr.budget]
daily_limit_seconds = 3600  # 每日本地硬上限，0 为不限
total_limit_seconds = 0     # 累计本地硬上限，0 为不限
usage_file = "data/usage.json"
```

本地 VAD 会直接减少上传流量和计费音频时长。`data/usage.json` 保存每个模型的每日和累计用量。

建议在百炼控制台同时打开**免费额度用完即停**。程序收到 `AllocationQuota.FreeTierOnly`、欠费或鉴权失败等错误时会封锁 ASR，需要在面板中确认后手动解除。

价格和免费额度以阿里云官方页面为准：

- [Qwen Audio ASR 文档](https://help.aliyun.com/zh/model-studio/qwen-audio-3.0-asr-flash)
- [百炼模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)

## 本地 TTS（不需要 TTS Key）

### macOS `say`

```toml
[tts]
kind = "local"
command = "say"
args = ["-v", "Tingting", "-o", "{output}", "--file-format=WAVE", "--data-format=LEI16@22050", "{text}"]
format = "wav"
request_timeout_s = 30
```

### Linux `espeak-ng`

先安装 `espeak-ng`，然后：

```toml
[tts]
kind = "local"
command = "espeak-ng"
args = ["-v", "zh", "-w", "{output}", "{text}"]
format = "wav"
request_timeout_s = 30
```

`{text}` 会替换成当前句子，`{output}` 会替换成临时音频文件路径。若不写 `{output}`，程序读取命令 stdout 作为音频。命令直接执行，不经过 shell，文本不会变成 shell 命令。

也可以使用 Piper 或其他能输出 WAV/PCM 的本地程序。`format = "pcm"` 时，音频按 `sample_rate` 指定的采样率解释为 s16le mono。

### OpenAI 兼容 TTS

如果已有服务兼容 `POST {base_url}/audio/speech`：

```toml
[tts]
kind = "openai"
base_url = "http://127.0.0.1:8880/v1"
model = "tts-1"
voice = "alloy"
format = "wav"
# api_key = "..."  # 也可用 TTS_API_KEY
```

长文本会按句子拆分，第一句生成后就可以开始播放；`ts_stop_speaking` 会停止播放并丢弃排队内容。

## MCP 接口

传输：Streamable HTTP。

```text
POST http://127.0.0.1:8787/mcp
Authorization: Bearer <server.tokens 中的 Token>
```

### Tools

| Tool | 作用 | 所需权限 |
|---|---|---|
| `ts_list_users` | 获取 Bot 当前频道成员 | `read` |
| `ts_get_status` | 连接、频道、播放、ASR 和用量状态 | `read` |
| `ts_get_transcripts` | 查询最终转写，支持 `after_seq` 游标 | `read` |
| `ts_speak` | 通过 TTS 在频道说话，支持 `interrupt` | `speak` |
| `ts_stop_speaking` | 停止播放并清空队列 | `speak` |
| `ts_send_message` | 发送当前频道文字消息 | `speak` |

`admin` 包含全部权限，并能调用面板的 ASR 控制接口。

`ts_get_transcripts` 支持：`limit`、`after_seq`、`since`、`client_id`、`channel_id`。Agent 可以保存最后一条事件的 `seq`，下次用 `after_seq` 只取新事件。

### Resources

- `ts://transcripts/latest`：最近 50 条最终转写。
- `ts://users`：当前频道成员。
- `ts://status`：Bot 和 ASR 状态。
- `ts://transcripts/{event_id}`：按事件 ID 读取一条转写。

支持资源订阅和 `notifications/resources/updated`。通知只表示数据更新，不会强制触发 LLM 推理；是否启动 Agent Run 由外部 Harness 决定。

### 转写事件示例

```json
{
  "event_id": "evt_12",
  "seq": 12,
  "type": "transcript.final",
  "utterance_id": "utt_...",
  "channel_id": 458,
  "channel_name": "停尸房",
  "speaker": {
    "client_id": 1123,
    "name": "ayami",
    "uid": "...",
    "database_id": 38
  },
  "text": "测试一二三，今天天气不错。",
  "timestamp": "2026-10-23T18:00:00Z",
  "end_timestamp": "2026-10-23T18:00:02Z",
  "duration_ms": 2000,
  "model": "qwen-audio-3.0-asr-flash"
}
```

## HTTP API

所有 `/api` 接口都需要 Bearer Token：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/status` | 完整状态 |
| GET | `/api/users` | 频道成员 |
| GET | `/api/transcripts` | 转写查询 |
| GET | `/api/events` | SSE 实时事件 |
| POST | `/api/speak` | TTS 发言 |
| POST | `/api/stop` | 停止播放 |
| POST | `/api/message` | 频道文字 |
| POST | `/api/asr` | ASR 开关和模式 |

示例：

```bash
export PANEL_TOKEN='你的控制台 Token'
curl -H "Authorization: Bearer $PANEL_TOKEN" \
  http://127.0.0.1:8787/api/transcripts?limit=20

curl -H "Authorization: Bearer $PANEL_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","interrupt":false}' \
  http://127.0.0.1:8787/api/speak
```

## 配置与安全

- `config.toml`、`dashscope-key`、`data/identity.json` 都不要提交到 git，建议权限为 600。
- 默认只绑定 `127.0.0.1:8787`。
- 不要在非回环地址上设置 `server.auth_disabled = true`。
- 对外提供 MCP 时，使用 TLS 反向代理、独立 Token 和最小权限 scopes，并配置 `server.allowed_hosts`。
- `read`、`speak`、`admin` 分离，避免只读 Agent 调用语音播放。
- 转写默认只保存在内存，可配置保留时间和最大条数；`transcripts.persist = true` 时写入 JSONL。
- 当前本机配置按个人偏好使用 `teamspeak.announcement = ""`，加入频道不自动发文字。生产环境如需告知成员语音会被转写，可设置提示文本。
- Bot 不会因为自己的声音被服务器回传而直接识别自己；使用外放时，可通过 `vad.suppress_while_bot_speaking = true` 降低回声循环风险。

## 常用排障

**面板提示缺 API Key**：检查 `dashscope-key` 是否存在、路径是否相对于启动时的工作目录，以及文件是否有换行以外的额外内容。

**ASR 返回 `AllocationQuota.FreeTierOnly`**：免费额度已耗尽或触发限制。程序会停止继续调用；确认额度和计费设置后再解除封锁。

**TTS 没有声音**：先直接运行配置中的本地命令，确认能生成 WAV；再检查 `format`、采样率和 Bot 当前频道。

**Bot 被移动到别的频道**：服务器成员可以拖动 Bot，程序不会强行拉回。重启后会回到 `teamspeak.channel_id` 指定的出生频道。

**远程打不开面板**：确认绑定地址和防火墙；优先使用 SSH 端口转发，不要把 HTTP 面板直接暴露到公网。

## 开发与测试

```bash
cargo fmt -- --check
cargo test
cargo clippy --all-targets -- -D warnings
```

测试包含音频重采样、Opus 分帧、WAV 解码、VAD 相关逻辑、转写查询、用量上限，以及不访问外网的 HTTP/WebSocket ASR 协议模拟。

主要模块：

- `src/ts.rs`：TeamSpeak 连接、按人音轨、播放和重连。
- `src/pipeline.rs`：每用户 VAD 和 ASR 流水线。
- `src/asr.rs`：百炼 HTTP/WebSocket 客户端。
- `src/transcripts.rs`：事件、保留策略、持久化和广播。
- `src/mcp.rs`：MCP Tools、Resources、订阅。
- `src/web.rs`、`src/panel.html`：同端口 REST/SSE 和控制台。
- `src/tts.rs`、`src/audio.rs`：TTS、解码、重采样、Opus。
- `src/usage.rs`、`src/config.rs`、`src/app.rs`：额度、配置和共享状态。
