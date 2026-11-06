# ONNX Runtime image generator

Generates images from text prompts with SD-Turbo. A Rust component runs the model through ONNX Runtime, which uses the GPU through `wasi:webgpu`.

## Run it

You need [`wash`](https://wasmcloud.com/docs/installation), Rust and [`wac`](https://github.com/bytecodealliance/wac). `rust-toolchain.toml` pins the nightly and its `wasm32-wasip3` target, which rustup installs on the first build.

```bash
cargo install wac-cli
```

Download the model (about 2.5 GB). Its [license](https://huggingface.co/schmuell/sd-turbo-ort-web/blob/main/LICENSE) is non-commercial.

```bash
base=https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1
for f in text_encoder/model.onnx unet/model.onnx vae_decoder/model.onnx tokenizer/vocab.json tokenizer/merges.txt; do
  curl -fL --create-dirs -o "models/$f" "$base/$f"
done
```

Start the demo, then open <http://localhost:8000>:

```bash
wash dev
```
