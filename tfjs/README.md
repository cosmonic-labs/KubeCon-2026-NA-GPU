# TensorFlow.js style transfer

Repaints one image in the style of another. TensorFlow.js runs the model in a long-lived wasmCloud service, using the GPU through `wasi:webgpu`, and an HTTP component serves the web page.

## Run it

You need [`wash`](https://wasmcloud.com/docs/installation) and Node.js 20 or later.

```sh
npm install
wash dev
```

Open <http://localhost:8000>, pick a content image and a style image, and click **Stylize**.

Or stylize from another terminal. This sends one of the bundled photos and styles as data URLs, as the page does:

```sh
img() { printf '{"dataUrl":"data:image/jpeg;base64,%s","width":%s,"height":%s}' "$(base64 < "component/src/static/images/$1" | tr -d '\n')" "$2" "$3"; }
printf '{"contentImage":%s,"styleImage":%s}' "$(img content/statue-of-liberty.jpg 250 252)" "$(img style/pen-and-ink.jpg 334 272)" > stylize.json
curl -sS -o stylized.jpg -H 'content-type: application/json' --data-binary @stylize.json http://localhost:8000/stylize
file stylized.jpg
```
