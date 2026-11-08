# llama.cpp chat

A chatbot served over HTTP from a WebAssembly component. The component runs [llama.cpp](https://github.com/ggml-org/llama.cpp) on the host's GPU through [`llama.wit`](https://github.com/cosmonic-labs/llama.wit) and `wasi:webgpu`.

## Run

Put the model in `models/`. It's gitignored, as it's 800 MB:

```shell
curl -L --create-dirs -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Then start it, and open http://localhost:8000/:

```shell
wash dev
```

You need `wash`, `wac`, Rust (`rust-toolchain.toml` pins the nightly and its `wasm32-wasip3` target), and a GPU.

To use another GGUF with a chat template, change `MODEL_FILE` in [`src/engine.rs`](src/engine.rs).

## How it works

- [`src/lib.rs`](src/lib.rs) serves the page and `POST /chat`, which streams the reply as one JSON event per line.
- [`src/engine.rs`](src/engine.rs) holds the model and does what `llama.h` leaves to the caller: templating, decoding, sampling, and reusing the tokens a new prompt shares with the last one.
- [`src/index.html`](src/index.html) is the chat page. It keeps the conversation and sends all of it with each message.
- [`vendor/llama-cpp.wasm`](vendor/) is the llama.cpp engine, built from `llama.wit`, and [`vendor/wit/llama-cpp/`](vendor/wit/llama-cpp/) is its interface, which [`wkg.toml`](wkg.toml) points `wash build` at. The build plugs the engine into this component with `wac`.

The component exports both `wasi:cli/run` and `wasi:http/handler`, and `service_file` in [`.wash/config.yaml`](.wash/config.yaml) runs it as a service. An HTTP component gets a new instance per request and would reload the model for every message. A service has one instance, so the model and the conversation's tokens stay loaded.

There is one context, so one reply generates at a time and other requests wait their turn.
