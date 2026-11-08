//! The model and the conversation held in it.
//!
//! llama.h leaves generation to the caller: apply the chat template, tokenize, decode, then
//! sample a token and decode it, until the model ends its turn.

use std::future::IntoFuture;
use std::rc::Rc;
use std::time::Instant;

use futures::lock::Mutex;
use serde::{Deserialize, Serialize};
use wasip3::filesystem::preopens::get_directories;
use wasip3::filesystem::types::{DescriptorFlags, OpenFlags, PathFlags};
use wit_bindgen::StreamWriter;

use crate::cosmonic::llama_cpp::api::{
    self as llama, ChatMessage, Context, ContextParams, Model, ModelParams, Sampler,
};

const MODELS_DIR: &str = "/models";
const MODEL_FILE: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";

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

/// What a reply streams to the client.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The model is still loading.
    Loading,
    /// More of the reply.
    Token {
        text: String,
    },
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
    Error {
        message: String,
    },
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
}

thread_local! {
    static ENGINE: Rc<Mutex<Option<Engine>>> = Rc::default();
}

/// Loads the model if it isn't yet.
pub async fn preload() -> Result<(), String> {
    let engine = ENGINE.with(Rc::clone);
    let mut engine = engine.lock().await;
    if engine.is_none() {
        *engine = Some(Engine::load().await?);
    }
    Ok(())
}

/// Streams the reply to `messages`.
pub async fn chat(messages: Vec<Message>, mut events: Events) {
    let engine = ENGINE.with(Rc::clone);
    // Waiting here means the model is loading, or another reply is being generated.
    let mut guard = engine.lock().await;
    if guard.is_none() {
        if !events.send(Event::Loading).await {
            return;
        }
        match Engine::load().await {
            Ok(loaded) => *guard = Some(loaded),
            Err(message) => {
                events.send(Event::Error { message }).await;
                return;
            }
        }
    }
    let engine = guard.as_mut().expect("loaded above");
    if let Err(message) = engine.reply(messages, &mut events).await {
        eprintln!("error: {message}");
        // The context may be out of step with `tokens`.
        engine.ctx.memory_clear(true).await;
        engine.tokens.clear();
        events.send(Event::Error { message }).await;
    }
}

impl Engine {
    async fn load() -> Result<Self, String> {
        let start = Instant::now();
        let (dir, _) = get_directories()
            .into_iter()
            .find(|(_, path)| path == MODELS_DIR)
            .ok_or_else(|| format!("{MODELS_DIR} is not mounted"))?;
        let open = async {
            let file = dir
                .open_at(
                    PathFlags::SYMLINK_FOLLOW,
                    MODEL_FILE.into(),
                    OpenFlags::empty(),
                    DescriptorFlags::READ,
                )
                .await?;
            let size = file.stat().await?.size;
            Ok::<_, wasip3::filesystem::types::ErrorCode>((file, size))
        };
        let (file, size) = open
            .await
            .map_err(|err| format!("{MODELS_DIR}/{MODEL_FILE}: {err:?}"))?;

        // The file goes straight from the host to the engine, without passing through this
        // component's memory.
        let (data, read_result) = file.read_via_stream(0);
        let (mut loaded, model) =
            Model::create(data, size, Some(ModelParams { n_gpu_layers: -1 })).await;
        // The engine reports how much of the file it has loaded; nothing here shows it.
        let progress = async { while loaded.next().await.is_some() {} };
        let ((), model, read) =
            futures::future::join3(progress, model.into_future(), read_result.into_future()).await;
        // A failed read ends the stream early, so the engine's error would only say the file
        // was short.
        read.map_err(|err| format!("{MODELS_DIR}/{MODEL_FILE}: {err:?}"))?;
        let model = model?;
        let n_ctx = match model.n_ctx_train() {
            0 => CONTEXT_LENGTH,
            trained => trained.min(CONTEXT_LENGTH),
        };
        let params = ContextParams {
            n_ctx,
            n_batch: BATCH.min(n_ctx),
            n_ubatch: 0,
            embeddings: false,
        };
        let ctx = Context::create(&model, Some(params)).await?;
        eprintln!(
            "loaded {} in {:.1} s",
            model.description(),
            start.elapsed().as_secs_f32()
        );
        Ok(Engine {
            ctx,
            tokens: Vec::new(),
            model,
        })
    }

    async fn reply(&mut self, messages: Vec<Message>, events: &mut Events) -> Result<(), String> {
        let (prompt, dropped_messages) = self.prompt(messages)?;
        let cached_tokens = self.reuse_prefix(&prompt).await;
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
        events
            .send(Event::Done {
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
        let template = self
            .model
            .chat_template(None)
            .ok_or("the model has no chat template")?;
        let mut chat: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let room = (self.ctx.n_ctx() as usize).saturating_sub(MIN_REPLY_ROOM);
        let mut dropped = 0;
        loop {
            let prompt = llama::chat_apply_template(&template, &chat, true)?;
            let tokens = self.model.tokenize(&prompt, true, true)?;
            if tokens.len() <= room {
                return Ok((tokens, dropped));
            }
            let oldest = chat
                .iter()
                .position(|m| m.role != "system")
                .filter(|i| i + 1 < chat.len());
            let Some(oldest) = oldest else {
                return Err(format!(
                    "the message is too long: {} tokens, the limit is {room}",
                    tokens.len()
                ));
            };
            chat.remove(oldest);
            dropped += 1;
        }
    }

    /// Keeps what the context shares with `prompt`, and returns how many tokens that is. The last
    /// token of the prompt is always decoded again, as sampling needs its logits.
    async fn reuse_prefix(&mut self, prompt: &[u32]) -> usize {
        let shared = self
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(held, wanted)| held == wanted)
            .count();
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
            self.ctx.memory_clear(true).await;
            self.tokens.clear();
            0
        }
    }

    async fn decode(&mut self, tokens: &[u32]) -> Result<(), String> {
        for chunk in tokens.chunks(self.ctx.n_batch() as usize) {
            self.ctx
                .decode(chunk.to_vec())
                .await
                .map_err(|code| format!("decode failed with code {code}"))?;
            self.tokens.extend_from_slice(chunk);
        }
        Ok(())
    }
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
