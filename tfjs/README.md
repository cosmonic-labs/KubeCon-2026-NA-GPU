# TensorFlow.js style transfer

Repaints one image in the style of another. TensorFlow.js runs the model in a long-lived wasmCloud service, using the GPU through `wasi:webgpu`, and an HTTP component serves the web page.

## Run it

You need [`wash`](https://wasmcloud.com/docs/installation) and Node.js 20 or later.

```bash
npm install
wash dev
```

Open <http://localhost:8000>, pick a content image and a style image, and click **Stylize**.
