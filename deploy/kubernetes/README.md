# Run the demos on Kubernetes

Runs the [ONNX Runtime](../../onnx-runtime/) and [TensorFlow.js](../../tfjs/) demos on a local [kind](https://kind.sigs.k8s.io/) cluster with the wasmCloud runtime operator, from the components CI publishes to `ghcr.io/cosmonic-labs/kubecon-2026-na-gpu`. Nothing is built but the host image.

You need Docker, [`kind`](https://kind.sigs.k8s.io/docs/user/quick-start/#installation), `kubectl` and [`helm`](https://helm.sh/docs/intro/install/). Give Docker at least 10 GB of memory for SD-Turbo; with both demos loaded the kind node uses about 8.5 GB.

## Run it

Download the SD-Turbo models first if you want that demo; it reads them from `onnx-runtime/models`. Skip this and `up.sh` deploys the TensorFlow.js demo alone.

```sh
cd onnx-runtime
base=https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1
for f in text_encoder/model.onnx unet/model.onnx vae_decoder/model.onnx tokenizer/vocab.json tokenizer/merges.txt; do
  curl -fL --create-dirs -o "models/$f" "$base/$f"
done
cd ..
```

Then, from the repository root:

```sh
deploy/kubernetes/up.sh
```

| Demo | URL |
|---|---|
| TensorFlow.js style transfer | <http://style.localhost:8000> |
| SD-Turbo image generation | <http://sd-turbo.localhost:8000> |

`PORT=9000 deploy/kubernetes/up.sh` uses another port. `deploy/kubernetes/down.sh` deletes the cluster.

## What it sets up

| File | What it is |
|---|---|
| [`up.sh`](up.sh) | Creates the `kubecon-gpu` kind cluster, builds and loads the host image, installs the operator chart and applies the workloads. Safe to run again. |
| [`Dockerfile`](Dockerfile) | The wasmCloud host (`ghcr.io/wasmcloud/wash:2.12.1`) plus Mesa's lavapipe Vulkan driver. |
| [`values.yaml`](values.yaml) | The [runtime operator chart](https://github.com/wasmCloud/wasmCloud/tree/main/charts/runtime-operator) 2.12.1, with one host group, `gpu`: `wasi:webgpu` on, the wasm proposals the demos need, HTTP on NodePort 30080, `/models` mounted and allowed, 10 GiB of memory. |
| [`tfjs.yaml`](tfjs.yaml) | The style transfer workload: an HTTP component and the long-running service that holds the models, at `style.localhost`. |
| [`sd-turbo.yaml`](sd-turbo.yaml) | The image generation workload at `sd-turbo.localhost`, granted `/models` read-only. |

Each workload is a `WorkloadDeployment` on the `gpu` host group. The host serves every workload on one port and picks one by the request's `Host` header; browsers send `*.localhost` to your machine.

Each manifest lists `wasi:webgpu` under `hostInterfaces`. `wash dev` adds it for you from the component's imports, but on Kubernetes the host links only what the workload asks for: leave it out and the workload fails with `instance export 'gpu' has the wrong type`.

## The GPU

kind runs Kubernetes inside a Docker container, which has no GPU of its own. The host image carries lavapipe, a Vulkan driver that runs on the CPU, so `wasi:webgpu` has an adapter on any machine. Everything works, but slowly. On an Apple silicon Mac, the style transfer service warms up in about 11 seconds and then stylizes an image in about 5; SD-Turbo loads its models in about 30 seconds and takes about 6 minutes per image. On a machine with a GPU, `wash dev` in each demo's directory is much faster.

To use an NVIDIA GPU, run the cluster on a Linux machine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and pass the GPU to the kind node, or use a cluster with GPU nodes and the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html). Then add `nodeSelector`, `tolerations` and an `nvidia.com/gpu` resource limit to the `gpu` host group in `values.yaml`.

## Troubleshooting

```sh
kubectl --context kind-kubecon-gpu get workloaddeployments
kubectl --context kind-kubecon-gpu -n wasmcloud logs deploy/hostgroup-gpu -f
```
