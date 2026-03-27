#!/usr/bin/env bash
set -euo pipefail

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
