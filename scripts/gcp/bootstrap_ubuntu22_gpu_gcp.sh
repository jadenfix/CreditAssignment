#!/usr/bin/env bash
set -euo pipefail

# Install gcloud CLI if not already present.
# On GCP VMs running ubuntu-2204-lts it is pre-installed; this guard makes the
# script safe to run on any Ubuntu 22.04 host.
if ! command -v gcloud &>/dev/null; then
  echo "gcloud not found; installing google-cloud-cli..."
  sudo apt-get update
  sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y google-cloud-cli
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/check_driver.sh"
"$SCRIPT_DIR/install_docker_ubuntu.sh"
"$SCRIPT_DIR/install_nvidia_toolkit_ubuntu.sh"
"$SCRIPT_DIR/configure_docker_runtime.sh"
"$SCRIPT_DIR/validate_cuda_container.sh"
