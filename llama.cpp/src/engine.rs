//! The model and the conversation held in it.
//!
//! llama.h leaves generation to the caller: apply the chat template, tokenize, decode, then
//! sample a token and decode it, until the model ends its turn.

use std::time::Instant;

use futures::lock::Mutex;
use serde::{Deserialize, Serialize};
use wasip3::filesystem::preopens::get_directories;
use wasip3::filesystem::types::{Descriptor, DescriptorFlags, DescriptorType, OpenFlags, PathFlags};
use wit_bindgen::StreamWriter;

use crate::cosmonic::llama_cpp::api::{self as llama, ChatMessage, Context, ContextParams, Model, ModelParams, Sampler};

const MODELS_DIR: &str = "/models";
/// Loaded at startup, and used when a request doesn't name a model.
pub const DEFAULT_MODEL: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";

const CONTEXT_LENGTH: u32 = 4096;
const BATCH: u32 = 512;
/// Older messages are dropped from the prompt to leave at least this much room for the reply.
const MIN_REPLY_ROOM: usize = 512;
const MAX_REPLY_TOKENS: usize = 1024;
/// `init-dist` picks a random seed for this value.
const RANDOM_SEED: u32 = u32::MAX;

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A model file in [`MODELS_DIR`].
#[derive(Serialize)]
pub struct ModelFile {
    pub file: String,
    pub bytes: u64,
}

/// The model in memory.
#[derive(Serialize, Clone)]
pub struct LoadedModel {
    pub file: String,
    /// llama.cpp's summary, such as `llama 1B Q4_K - Medium`.
    pub description: String,
    pub bytes: u64,
    /// The context this chat uses, at most [`CONTEXT_LENGTH`].
    pub context: u32,
    /// The context the model was trained with.
    pub trained_context: u32,
    pub load_seconds: f32,
}

/// What `GET /models` returns.
#[derive(Serialize)]
pub struct Models {
    pub default: &'static str,
    pub loaded: Option<LoadedModel>,
    /// The model being loaded right now, if any.
    pub loading: Option<String>,
    pub files: Vec<ModelFile>,
}

/// What a reply streams to the client.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The requested model is loading, after the previous one was freed.
    Loading { model: String },
    /// The requested model finished loading.
    Loaded { model: LoadedModel },
    /// More of the reply.
    Token { text: String },
    Done {
        /// Why it ended: `stop` if the model finished, `length` if it ran out of room.
        finish: &'static str,
        prompt_tokens: usize,
        /// How many of `prompt_tokens` were already in the context, from the previous turn.
        cached_tokens: usize,
        /// How many of the oldest messages didn't fit in the context.
        dropped_messages: usize,
        reply_tokens: usize,
        reply_seconds: f32,
    },
    Error { message: String },
}

/// The response body: one JSON event per line.
pub struct Events {
    tx: StreamWriter<u8>,
}

impl Events {
    pub fn new(tx: StreamWriter<u8>) -> Self {
        Events { tx }
    }

    /// Returns false once the client has gone away.
    async fn send(&mut self, event: Event) -> bool {
        let mut line = serde_json::to_vec(&event).expect("events serialize");
        line.push(b'\n');
        self.tx.write_all(line).await.is_empty()
    }
}

/// One conversation at a time: the model has a single context, and requests take turns on it.
struct Engine {
    // Before `model`, which has to outlive it.
    ctx: Context,
    /// The tokens in `ctx`.
    tokens: Vec<u32>,
    model: Model,
    info: LoadedModel,
}

// Statics rather than `thread_local!`: on wasm32-wasip3 each export call runs as its own task,
// with its own thread-local storage, so state kept there would be gone by the next request.
static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
/// A copy of the engine's `info`, readable while a reply holds the engine.
static LOADED: std::sync::Mutex<Option<LoadedModel>> = std::sync::Mutex::new(None);

/// The file being loaded, readable while the load holds the engine.
static LOADING: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

fn set_loaded(info: Option<LoadedModel>) {
    *LOADED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = info;
}

fn set_loading(file: Option<&str>) {
    *LOADING.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = file.map(Into::into);
}

/// Replaces the model in `slot` with `file`. The old one is freed first, so that two never share
/// GPU memory.
async fn switch_to(slot: &mut Option<Engine>, file: &str) -> Result<(), String> {
    *slot = None;
    set_loaded(None);
    set_loading(Some(file));
    let loaded = Engine::load(file).await;
    set_loading(None);
    *slot = Some(loaded?);
    Ok(())
}

/// Whether `file` can name a model: a `.gguf` file directly in [`MODELS_DIR`].
pub fn is_model_name(file: &str) -> bool {
    file.ends_with(".gguf") && !file.starts_with('.') && !file.contains(['/', '\\'])
}

/// Lists the models in [`MODELS_DIR`], smallest first.
pub async fn models() -> Result<Models, String> {
    let dir = models_dir()?;
    let (entries, listed) = dir.read_directory();
    let entries = entries.collect().await;
    listed.await.map_err(|err| format!("listing {MODELS_DIR}: {err:?}"))?;
    let mut files = Vec::new();
    for entry in entries {
        let is_file = matches!(entry.type_, DescriptorType::RegularFile | DescriptorType::SymbolicLink);
        if !is_file || !is_model_name(&entry.name) {
            continue;
        }
        // A symlink's size is the size of the file it points to.
        let stat = dir.stat_at(PathFlags::SYMLINK_FOLLOW, entry.name.clone()).await;
        let stat = stat.map_err(|err| format!("{MODELS_DIR}/{}: {err:?}", entry.name))?;
        files.push(ModelFile { file: entry.name, bytes: stat.size });
    }
    files.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.file.cmp(&b.file)));
    let loaded = LOADED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
    let loading = LOADING.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
    Ok(Models { default: DEFAULT_MODEL, loaded, loading, files })
}

/// Loads the default model if no model is loaded yet.
pub async fn preload() -> Result<(), String> {
    let mut engine = ENGINE.lock().await;
    if engine.is_none() {
        switch_to(&mut engine, DEFAULT_MODEL).await?;
    }
    Ok(())
}

/// Makes `model` the loaded model, unless it already is.
pub async fn load(model: &str) -> Result<LoadedModel, String> {
    // Waiting here means another model is loading, or a reply is being generated.
    let mut engine = ENGINE.lock().await;
    if engine.as_ref().is_none_or(|engine| engine.info.file != model) {
        switch_to(&mut engine, model).await?;
    }
    Ok(engine.as_ref().expect("loaded above").info.clone())
}

/// Streams the reply to `messages` from `model`, loading it first if another one is loaded.
pub async fn chat(model: String, messages: Vec<Message>, mut events: Events) {
    // Waiting here means a model is loading, or another reply is being generated.
    let mut guard = ENGINE.lock().await;
    if guard.as_ref().is_none_or(|engine| engine.info.file != model) {
        if !events.send(Event::Loading { model: model.clone() }).await {
            return;
        }
        if let Err(message) = switch_to(&mut guard, &model).await {
            events.send(Event::Error { message }).await;
            return;
        }
        let info = guard.as_ref().expect("loaded above").info.clone();
        if !events.send(Event::Loaded { model: info }).await {
            return;
        }
    }
    let engine = guard.as_mut().expect("loaded above");
    if let Err(message) = engine.reply(messages, &mut events).await {
        eprintln!("error: {message}");
        // The context may be out of step with `tokens`.
        engine.ctx.memory_clear(true);
        engine.tokens.clear();
        events.send(Event::Error { message }).await;
    }
}

impl Engine {
    async fn load(name: &str) -> Result<Self, String> {
        if !is_model_name(name) {
            return Err(format!("{name:?} is not a .gguf file in {MODELS_DIR}"));
        }
        let start = Instant::now();
        let dir = models_dir()?;
        let open = async {
            let file =
                dir.open_at(PathFlags::SYMLINK_FOLLOW, name.into(), OpenFlags::empty(), DescriptorFlags::READ).await?;
            let size = file.stat().await?.size;
            Ok::<_, wasip3::filesystem::types::ErrorCode>((file, size))
        };
        let (file, size) = open.await.map_err(|err| format!("{MODELS_DIR}/{name}: {err:?}"))?;

        // The file goes straight from the host to the engine, without passing through this
        // component's memory.
        let (data, _read_result) = file.read_via_stream(0);
        let model = Model::create(data, size, Some(ModelParams { n_gpu_layers: -1 })).await?;
        let n_ctx = match model.n_ctx_train() {
            0 => CONTEXT_LENGTH,
            trained => trained.min(CONTEXT_LENGTH),
        };
        let params = ContextParams { n_ctx, n_batch: BATCH.min(n_ctx), n_ubatch: 0, embeddings: false };
        let ctx = Context::create(&model, Some(params)).await?;
        let info = LoadedModel {
            file: name.into(),
            description: model.description(),
            bytes: size,
            context: n_ctx,
            trained_context: model.n_ctx_train(),
            load_seconds: start.elapsed().as_secs_f32(),
        };
        eprintln!("loaded {name} ({}) in {:.1} s", info.description, info.load_seconds);
        set_loaded(Some(info.clone()));
        Ok(Engine { ctx, tokens: Vec::new(), model, info })
    }

    async fn reply(&mut self, messages: Vec<Message>, events: &mut Events) -> Result<(), String> {
        let (prompt, dropped_messages) = self.prompt(messages)?;
        let cached_tokens = self.reuse_prefix(&prompt);
        self.decode(&prompt[cached_tokens..]).await?;

        let sampler = sampler()?;
        let n_ctx = self.ctx.n_ctx() as usize;
        let start = Instant::now();
        let mut pending = Vec::new();
        let mut reply_tokens = 0;
        let finish = loop {
            let token = sampler.sample(&self.ctx, -1).await?;
            if self.model.is_eog(token) {
                break "stop";
            }
            reply_tokens += 1;
            pending.extend(self.model.token_to_piece(token, 0, false)?);
            let text = complete_characters(&mut pending);
            if !text.is_empty() && !events.send(Event::Token { text }).await {
                return Ok(());
            }
            if reply_tokens >= MAX_REPLY_TOKENS || self.tokens.len() >= n_ctx {
                break "length";
            }
            self.decode(&[token]).await?;
        };

        // A character the reply cut off.
        let text = String::from_utf8_lossy(&pending).into_owned();
        if !text.is_empty() && !events.send(Event::Token { text }).await {
            return Ok(());
        }
        events.send(Event::Done {
            finish,
            prompt_tokens: prompt.len(),
            cached_tokens,
            dropped_messages,
            reply_tokens,
            reply_seconds: start.elapsed().as_secs_f32(),
        })
        .await;
        Ok(())
    }

    /// Tokenizes the chat, dropping its oldest turns until it leaves room for a reply.
    fn prompt(&self, messages: Vec<Message>) -> Result<(Vec<u32>, usize), String> {
        let template = self.model.chat_template(None).ok_or("the model has no chat template")?;
        let mut chat: Vec<ChatMessage> =
            messages.into_iter().map(|m| ChatMessage { role: m.role, content: m.content }).collect();
        let room = (self.ctx.n_ctx() as usize).saturating_sub(MIN_REPLY_ROOM);
        let mut dropped = 0;
        loop {
            let prompt = llama::chat_apply_template(&template, &chat, true)?;
            let tokens = self.model.tokenize(&prompt, true, true)?;
            if tokens.len() <= room {
                return Ok((tokens, dropped));
            }
            let oldest = chat.iter().position(|m| m.role != "system").filter(|i| i + 1 < chat.len());
            let Some(oldest) = oldest else {
                return Err(format!("the message is too long: {} tokens, the limit is {room}", tokens.len()));
            };
            chat.remove(oldest);
            dropped += 1;
        }
    }

    /// Keeps what the context shares with `prompt`, and returns how many tokens that is. The last
    /// token of the prompt is always decoded again, as sampling needs its logits.
    fn reuse_prefix(&mut self, prompt: &[u32]) -> usize {
        let shared = self.tokens.iter().zip(prompt).take_while(|(held, wanted)| held == wanted).count();
        let shared = shared.min(prompt.len().saturating_sub(1));
        // With sliding-window attention, the first token after the prefix attends to the n-swa
        // before it, which the context may have dropped.
        let oldest_needed = (shared as i64 - i64::from(self.model.n_swa())).max(0);
        let oldest_held = i64::from(self.ctx.memory_seq_pos_min(0));
        let kept = shared > 0
            && oldest_held >= 0
            && oldest_held <= oldest_needed
            && self.ctx.memory_seq_rm(0, shared as i32, -1);
        if kept {
            self.tokens.truncate(shared);
            shared
        } else {
            self.ctx.memory_clear(true);
            self.tokens.clear();
            0
        }
    }

    async fn decode(&mut self, tokens: &[u32]) -> Result<(), String> {
        for chunk in tokens.chunks(self.ctx.n_batch() as usize) {
            self.ctx.decode(chunk.to_vec()).await.map_err(|code| format!("decode failed with code {code}"))?;
            self.tokens.extend_from_slice(chunk);
        }
        Ok(())
    }
}

fn models_dir() -> Result<Descriptor, String> {
    get_directories()
        .into_iter()
        .find(|(_, path)| path == MODELS_DIR)
        .map(|(dir, _)| dir)
        .ok_or_else(|| format!("{MODELS_DIR} is not mounted"))
}

/// llama.cpp's default sampling, from `common_params_sampling`.
fn sampler() -> Result<Sampler, String> {
    let chain = Sampler::chain_init();
    chain.chain_add(Sampler::init_top_k(40))?;
    chain.chain_add(Sampler::init_top_p(0.95, 1))?;
    chain.chain_add(Sampler::init_min_p(0.05, 1))?;
    chain.chain_add(Sampler::init_temp(0.8))?;
    chain.chain_add(Sampler::init_dist(RANDOM_SEED))?;
    Ok(chain)
}

/// Takes the whole characters off the front of `bytes`, as a token can end partway through one.
/// Bytes that can never become valid UTF-8 come out as U+FFFD.
fn complete_characters(bytes: &mut Vec<u8>) -> String {
    let mut text = String::new();
    loop {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                text.push_str(valid);
                bytes.clear();
                return text;
            }
            Err(err) => {
                let valid = err.valid_up_to();
                text.push_str(std::str::from_utf8(&bytes[..valid]).expect("checked valid"));
                match err.error_len() {
                    Some(invalid) => {
                        text.push(char::REPLACEMENT_CHARACTER);
                        bytes.drain(..valid + invalid);
                    }
                    None => {
                        bytes.drain(..valid);
                        return text;
                    }
                }
            }
        }
    }
}
