#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" - <<'PY'
import ast
import pathlib
import subprocess

root = pathlib.Path.cwd()
files = sorted(root.rglob('*.py'))
files = [p for p in files if '.git' not in p.parts and '__pycache__' not in p.parts]
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
for path in sorted((root / 'scripts').glob('*.sh')):
    subprocess.run(['bash', '-n', str(path)], check=True)
print(f'Syntax check passed: {len(files)} Python files; shell entry points parse successfully.')
PY
