#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

