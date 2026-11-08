#!/usr/bin/env bash
# Run the demos on a local kind cluster with wasmCloud:
#
#   deploy/kubernetes/up.sh
#
# 1. a one-node kind cluster, kubecon-gpu, with onnx-runtime/models mounted
#    into the node and NodePort 30080 mapped to localhost:$PORT (8000)
# 2. the wasmCloud host image with lavapipe (Dockerfile), loaded into kind
# 3. the wasmCloud runtime operator chart, with values.yaml
# 4. the TensorFlow.js demo, and SD-Turbo when its models are downloaded
#
# Running it again updates the image, the chart and the workloads in place.
# The components come from ghcr.io/cosmonic-labs/kubecon-2026-na-gpu, built
# from main by .github/workflows/components.yml.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CLUSTER="${CLUSTER:-kubecon-gpu}"
CTX="kind-$CLUSTER"
PORT="${PORT:-8000}"
CHART_VERSION=2.12.1
IMAGE=localhost/wasmcloud-host-lavapipe:2.12.1
MODELS="${MODELS:-$ROOT/onnx-runtime/models}"

for tool in docker kind kubectl helm; do
  command -v "$tool" >/dev/null || { echo "up.sh needs $tool on PATH" >&2; exit 1; }
done

# kind mounts the models directory when it creates the node, so it has to
# exist first. Downloading the models later is fine: the mount is live.
mkdir -p "$MODELS"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config - <<YAML
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: $PORT
        listenAddress: 127.0.0.1
    extraMounts:
      - hostPath: $MODELS
        containerPath: /models
        readOnly: true
YAML
fi

docker build -t "$IMAGE" "$HERE"
# With Docker Desktop's containerd image store, a built image is a
# multi-platform index that `kind load docker-image` rejects; save just this
# machine's platform instead.
if ! kind load docker-image "$IMAGE" --name "$CLUSTER" >/dev/null 2>&1; then
  archive="$(mktemp "${TMPDIR:-/tmp}/wasmcloud-host.XXXXXX")"
  trap 'rm -f "$archive"' EXIT
  docker save --platform "$(docker version -f '{{.Server.Os}}/{{.Server.Arch}}')" -o "$archive" "$IMAGE"
  kind load image-archive "$archive" --name "$CLUSTER"
fi

helm --kube-context "$CTX" upgrade --install wasmcloud \
  oci://ghcr.io/wasmcloud/charts/runtime-operator --version "$CHART_VERSION" \
  --namespace wasmcloud --create-namespace \
  -f "$HERE/values.yaml" --wait --timeout 5m
# A rebuilt image under the same tag needs a new pod to take effect.
kubectl --context "$CTX" -n wasmcloud rollout restart deploy/hostgroup-gpu >/dev/null
kubectl --context "$CTX" -n wasmcloud rollout status deploy/hostgroup-gpu --timeout 5m

kubectl --context "$CTX" apply -f "$HERE/tfjs.yaml"
if [ -f "$MODELS/unet/model.onnx" ]; then
  kubectl --context "$CTX" apply -f "$HERE/sd-turbo.yaml"
else
  echo "No models in onnx-runtime/models yet: skipping SD-Turbo."
  echo "Download them (onnx-runtime/README.md) and run up.sh again."
fi

echo
echo "Waiting for the workloads to be ready (the first pull takes a minute)..."
kubectl --context "$CTX" wait workloaddeployment --all --for=condition=Ready --timeout 10m
echo
echo "  TensorFlow.js style transfer   http://style.localhost:$PORT"
[ -f "$MODELS/unet/model.onnx" ] && echo "  SD-Turbo image generation      http://sd-turbo.localhost:$PORT"
echo
echo "Host logs: kubectl --context $CTX -n wasmcloud logs deploy/hostgroup-gpu -f"
