# KubeCon 2026 NA: GPU

Demos from our KubeCon + CloudNativeCon North America 2026 talk. Each one runs machine learning on the GPU from inside a WebAssembly component on wasmCloud, using `wasi:webgpu`.

| Demo | What it does |
|---|---|
| [onnx-runtime](onnx-runtime/) | Generates images from text prompts with ONNX Runtime and SD-Turbo. |
| [llama.cpp](llama.cpp/) | Runs a chatbot with llama.cpp. |
| [tfjs](tfjs/) | Transfers the style of one image onto another with TensorFlow.js. |

All of them need [`wash`](https://wasmcloud.com/docs/installation) and a machine with a GPU.
