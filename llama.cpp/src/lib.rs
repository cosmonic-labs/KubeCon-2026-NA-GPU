//! A chatbot over HTTP, running llama.cpp on the GPU through the
//! cosmonic:llama-cpp interface.
//!
//! - `GET /` serves the chat page.
//! - `GET /models` lists the models in `/models` with their sizes, the default, and the loaded one,
//!   see [`engine::Models`].
//! - `POST /load` takes `{"model"}`, makes it the loaded model, freeing the previous one, and
//!   returns it once it's loaded, see [`engine::LoadedModel`].
//! - `POST /chat` takes `{"model"?, "messages": [{"role", "content"}]}` and streams the reply as one
//!   JSON event per line, see [`engine::Event`]. Without `model`, it uses [`engine::DEFAULT_MODEL`].
//!
//! It exports both `wasi:cli/run` and `wasi:http/handler`, and runs as a wasmCloud service: one
//! instance lives as long as the workload and answers every request, so the model stays loaded.

wit_bindgen::generate!({
    path: "wit",
    world: "chat",
    generate_all,
});

mod engine;

use serde::Deserialize;
use wasip3::http::types::{ErrorCode, Fields, Method, Request, Response};
use wasip3::{wit_future, wit_stream as http_stream};
use wit_bindgen::StreamResult;

const INDEX_HTML: &str = include_str!("index.html");
const MAX_BODY: usize = 1 << 20;

struct Component;

wasip3::cli::command::export!(Component);
wasip3::http::service::export!(Component);

impl wasip3::exports::cli::run::Guest for Component {
    async fn run() -> Result<(), ()> {
        // Load up front, so that the first message doesn't wait for it. A failure is retried, and
        // reported, by the next message.
        if let Err(err) = engine::preload().await {
            eprintln!("failed to load the model: {err}");
        }
        // Returning would end the service.
        loop {
            wasip3::clocks::monotonic_clock::wait_for(u64::MAX).await;
        }
    }
}

impl wasip3::exports::http::handler::Guest for Component {
    async fn handle(request: Request) -> Result<Response, ErrorCode> {
        let target = request.get_path_with_query().unwrap_or_default();
        let path = target.split('?').next().unwrap_or_default();
        match (request.get_method(), path) {
            (Method::Get, "/") => respond(200, "text/html; charset=utf-8", INDEX_HTML.into()),
            (Method::Get, "/models") => models().await,
            (Method::Post, "/load") => load(request).await,
            (Method::Post, "/chat") => chat(request).await,
            _ => respond(404, "text/plain; charset=utf-8", "not found\n".into()),
        }
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<engine::Message>,
}

#[derive(Deserialize)]
struct LoadRequest {
    model: String,
}

async fn load(request: Request) -> Result<Response, ErrorCode> {
    let Some(body) = read_body(request).await else {
        return respond(413, "text/plain; charset=utf-8", "request too large\n".into());
    };
    let model = match serde_json::from_slice::<LoadRequest>(&body) {
        Ok(LoadRequest { model }) if engine::is_model_name(&model) => model,
        Ok(_) => {
            return respond(
                400,
                "text/plain; charset=utf-8",
                "the model must be the name of a .gguf file in /models\n".into(),
            );
        }
        Err(err) => return respond(400, "text/plain; charset=utf-8", format!("{err}\n")),
    };
    match engine::load(&model).await {
        Ok(loaded) => respond(200, "application/json", serde_json::to_string(&loaded).expect("model serializes")),
        Err(err) => respond(500, "text/plain; charset=utf-8", format!("{err}\n")),
    }
}

async fn models() -> Result<Response, ErrorCode> {
    match engine::models().await {
        Ok(models) => respond(200, "application/json", serde_json::to_string(&models).expect("models serialize")),
        Err(err) => respond(500, "text/plain; charset=utf-8", format!("{err}\n")),
    }
}

async fn chat(request: Request) -> Result<Response, ErrorCode> {
    let Some(body) = read_body(request).await else {
        return respond(413, "text/plain; charset=utf-8", "request too large\n".into());
    };
    let (model, messages) = match parse(&body) {
        Ok(request) => request,
        Err(err) => return respond(400, "text/plain; charset=utf-8", format!("{err}\n")),
    };

    let (tx, rx) = http_stream::new();
    let (response, _sent) = new_response("application/x-ndjson", rx)?;
    // Sent from a task of its own, as the host only starts reading the body once it has the head.
    wit_bindgen::spawn_local(engine::chat(model, messages, engine::Events::new(tx)));
    Ok(response)
}

/// Returns the model to use, and the messages.
fn parse(body: &[u8]) -> Result<(String, Vec<engine::Message>), String> {
    let request: ChatRequest = serde_json::from_slice(body).map_err(|err| err.to_string())?;
    let model = request.model.unwrap_or_else(|| engine::DEFAULT_MODEL.into());
    if !engine::is_model_name(&model) {
        return Err("the model must be the name of a .gguf file in /models".into());
    }
    if request.messages.iter().any(|m| !["system", "user", "assistant"].contains(&m.role.as_str())) {
        return Err("a message's role must be system, user or assistant".into());
    }
    if request.messages.last().is_none_or(|m| m.role != "user") {
        return Err("the last message must be from the user".into());
    }
    Ok((model, request.messages))
}

/// Reads the request body, or returns `None` if it is over [`MAX_BODY`].
async fn read_body(request: Request) -> Option<Vec<u8>> {
    let (_result_tx, result_rx) = wit_future::new(|| Ok(()));
    let (mut stream, _trailers) = Request::consume_body(request, result_rx);
    let mut body = Vec::new();
    loop {
        let (status, chunk) = stream.read(Vec::with_capacity(64 << 10)).await;
        body.extend_from_slice(&chunk);
        if body.len() > MAX_BODY {
            return None;
        }
        if !matches!(status, StreamResult::Complete(_)) {
            return Some(body);
        }
    }
}

fn respond(status: u16, content_type: &str, body: String) -> Result<Response, ErrorCode> {
    let (mut tx, rx) = http_stream::new();
    let (response, _sent) = new_response(content_type, rx)?;
    response
        .set_status_code(status)
        .map_err(|()| ErrorCode::InternalError(Some("invalid status code".into())))?;
    wit_bindgen::spawn_local(async move {
        tx.write_all(body.into_bytes()).await;
    });
    Ok(response)
}

fn new_response(
    content_type: &str,
    body: wit_bindgen::StreamReader<u8>,
) -> Result<(Response, wit_bindgen::FutureReader<Result<(), ErrorCode>>), ErrorCode> {
    let headers = Fields::from_list(&[("content-type".to_string(), content_type.as_bytes().to_vec())])
        .map_err(|err| ErrorCode::InternalError(Some(format!("invalid headers: {err}"))))?;
    let (_trailers_tx, trailers_rx) = wit_future::new(|| Ok(None));
    Ok(Response::new(headers, Some(body), trailers_rx))
}
