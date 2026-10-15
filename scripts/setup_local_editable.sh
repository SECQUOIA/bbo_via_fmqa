#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v uv >/dev/null 2>&1; then
    uv pip install -e "$repo_root/fmqa" -e "$repo_root/bbo_via_fmqa"
else
    python -m pip install -e "$repo_root/fmqa" -e "$repo_root/bbo_via_fmqa"
fi
