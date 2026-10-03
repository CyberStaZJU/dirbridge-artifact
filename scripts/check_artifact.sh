#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m compileall -q .
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
find . -name '*.pyc' -type f -delete

if grep -RInE '47\.107|/home/|WorkBuddy|workbuddy|cyb@|Hunan|hnu|Yibo|Zhizhong|Chao Zhang|Keqin|Kenli|相关工作|skills|\.workbuddy' . --exclude-dir=.git --exclude='check_artifact.sh'; then
  echo "Potential private or non-anonymous content found. Please inspect the matches above." >&2
  exit 1
fi

echo "Artifact check passed."
