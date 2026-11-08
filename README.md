# KubeCon 2026 NA: GPU

Demos from our talk **GPU Portability Was Broken, But Now There's WebGPU**, by Bailey Hayes and Mendy Berger, at Cloud Native AI + Inference Day North America 2026, co-located with KubeCon + CloudNativeCon North America. Each one runs machine learning on the GPU from inside a WebAssembly component on [wasmCloud](https://github.com/wasmCloud/wasmCloud), a CNCF incubating project, using `wasi:webgpu`.

- [Slides](https://docs.google.com/presentation/d/1XSuhDqWWACD6UbhuxEaW_ebqrtHIU-ECw8BWaz27JJE/edit?usp=sharing)
- [Demo videos](https://drive.google.com/drive/folders/1Bwp5kfoMeG3n4zAbteudTW1h-HJkEXA1?usp=sharing): recordings of each demo running, if you'd rather watch than run them.
- [GPU benchmarks](https://arewefastyet.wasmcloud.com/gpu.html): one model on one GPU three ways (the vendor backend, WebGPU as a native binary, and the same WebGPU build as a Wasm component), then several models taking turns on one GPU.

| Demo | What it does |
|---|---|
| [onnx-runtime](onnx-runtime/) | Generates images from text prompts with ONNX Runtime and SD-Turbo. |
| [llama.cpp](llama.cpp/) | Runs a chatbot with llama.cpp. |
| [tfjs](tfjs/) | Transfers the style of one image onto another with TensorFlow.js. |

All of them need [`wash`](https://wasmcloud.com/docs/installation) and a machine with a GPU.

## Run them on Kubernetes

Each demo runs locally with `wash dev`. To run wasmCloud workloads on a Kubernetes cluster, see the [wasmCloud Kubernetes docs](https://wasmcloud.com/docs/kubernetes/).

## Learn more

- [wasi-gfx](https://github.com/WebAssembly/wasi-gfx): the WASI proposals behind `wasi:webgpu`.
- [wasmCloud community calls](https://wasmcloud.com/community/): every Wednesday, with agendas, notes and recordings.
- [wasi-gfx meetings](https://github.com/wasi-gfx): every Tuesday, 17:00 to 18:00 UTC, open to anyone.

## License

The code is [Apache-2.0](LICENSE). The models and images keep their own licenses:

- SD-Turbo, for onnx-runtime, is [non-commercial](https://huggingface.co/schmuell/sd-turbo-ort-web/blob/main/LICENSE).
- Llama 3.2, for llama.cpp, is under the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt).
- The tfjs models and images are credited in their `ATTRIBUTION.md` files.
