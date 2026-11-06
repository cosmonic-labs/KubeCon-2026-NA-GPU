//! An HTTP server that generates images from text prompts with SD-Turbo, through the
//! onnxruntime:onnxruntime interface.
//!
//! - `GET /` serves a page with a prompt box.
//! - `GET /generate?prompt=<text>[&seed=<n>]` returns a 512x512 PNG.
//!
//! The models are read from `/models`, see the README for how to get them.

wit_bindgen::generate!({
    world: "example:sd-turbo/sd-turbo",
    // The onnxruntime:onnxruntime package is resolved from wit/deps.
    path: "wit",
    generate_all,
});

mod pipeline;
mod tokenizer;

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use pipeline::Pipeline;
use wasip3::http::types::{ErrorCode, Fields, Request, Response};
use wasip3::http_compat::{BodyWriter, http_from_wasi_request};

const MODELS_DIR: &str = "/models";

thread_local! {
    // Loading the models takes seconds, so they are loaded on the first request and kept for the next ones.
    static PIPELINE: RefCell<Option<Rc<Pipeline>>> = const { RefCell::new(None) };
}

struct Component;

wasip3::http::service::export!(Component);

impl wasip3::exports::http::handler::Guest for Component {
    async fn handle(request: Request) -> Result<Response, ErrorCode> {
        let request = http_from_wasi_request(request)?;
        let path = request.uri().path().to_string();
        let query = request.uri().query().unwrap_or("").to_string();
        drop(request);

        match path.as_str() {
            "/" => respond(200, "text/html; charset=utf-8", INDEX_HTML.as_bytes().to_vec()),
            "/generate" => generate(&query).await,
            _ => respond(404, "text/plain; charset=utf-8", b"not found\n".to_vec()),
        }
    }
}

async fn generate(query: &str) -> Result<Response, ErrorCode> {
    let Some(prompt) = query_param(query, "prompt").filter(|p| !p.trim().is_empty()) else {
        return respond(400, "text/plain; charset=utf-8", b"missing ?prompt=\n".to_vec());
    };
    let seed = query_param(query, "seed").and_then(|s| s.parse().ok()).unwrap_or_else(time_seed);

    let result = match pipeline().await {
        Ok(pipeline) => pipeline.generate(&prompt, seed).await,
        Err(err) => Err(err),
    };
    match result {
        Ok(png) => respond(200, "image/png", png),
        Err(err) => {
            eprintln!("error: {err}");
            respond(500, "text/plain; charset=utf-8", format!("{err}\n").into_bytes())
        }
    }
}

async fn pipeline() -> Result<Rc<Pipeline>, String> {
    if let Some(pipeline) = PIPELINE.with_borrow(Option::clone) {
        return Ok(pipeline);
    }
    let pipeline = Rc::new(Pipeline::load(Path::new(MODELS_DIR)).await?);
    PIPELINE.set(Some(pipeline.clone()));
    Ok(pipeline)
}

/// Builds a complete response. The body is a stream, written from a spawned task after the response head is
/// handed back to the host.
fn respond(status: u16, content_type: &str, body: Vec<u8>) -> Result<Response, ErrorCode> {
    let headers = Fields::from_list(&[("content-type".to_string(), content_type.as_bytes().to_vec())])
        .map_err(|err| ErrorCode::InternalError(Some(format!("invalid headers: {err}"))))?;

    let (mut writer, body_rx, result_rx) = BodyWriter::new();
    let (response, _transmit) = Response::new(headers, Some(body_rx), result_rx);
    response.set_status_code(status).map_err(|()| ErrorCode::InternalError(Some("invalid status code".into())))?;

    wit_bindgen::spawn_local(async move {
        let frame = http_body::Frame::data(bytes::Bytes::from(body));
        let _ = writer.send_frame(frame).await;
        drop(writer.stream_writer);
        let _ = writer.result_writer.write(Ok(None)).await;
    });

    Ok(response)
}

fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        (k == key).then(|| {
            let v = v.replace('+', " ");
            urlencoding::decode(&v).map(|c| c.into_owned()).unwrap_or(v)
        })
    })
}

fn time_seed() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64)
}

const INDEX_HTML: &str = include_str!("index.html");
