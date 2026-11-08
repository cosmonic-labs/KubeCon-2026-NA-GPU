#!/usr/bin/env bash
# Delete the kind cluster up.sh made. The downloaded models stay.
set -euo pipefail
kind delete cluster --name "${CLUSTER:-kubecon-gpu}"
