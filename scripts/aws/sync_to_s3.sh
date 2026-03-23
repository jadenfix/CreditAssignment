#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <local-path> <s3-uri>" >&2
  exit 1
fi

aws s3 sync "$1" "$2"

