# The wasmCloud host with a Vulkan driver it can find in any kind node.
#
# ghcr.io/wasmcloud/wash already has wasi:webgpu built in, but no Vulkan
# loader or driver, so wgpu finds no adapter. Mesa's lavapipe is a software
# Vulkan driver: it runs the demos on the CPU, on any machine, slowly.
FROM ghcr.io/wasmcloud/wash:2.12.1
RUN apk add --no-cache mesa vulkan-loader-libs
