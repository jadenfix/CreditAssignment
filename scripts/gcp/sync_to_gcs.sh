#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <local-path> <gcs-uri>" >&2
  exit 1
fi

gcloud storage rsync --recursive "$1" "$2"
