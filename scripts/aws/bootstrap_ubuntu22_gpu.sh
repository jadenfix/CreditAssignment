#!/usr/bin/env bash
set -euo pipefail

scripts/aws/check_driver.sh
scripts/aws/install_docker_ubuntu.sh
scripts/aws/install_nvidia_toolkit_ubuntu.sh
scripts/aws/configure_docker_runtime.sh
scripts/aws/validate_cuda_container.sh

