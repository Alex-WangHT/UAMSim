#!/usr/bin/env bash
# Install only the validation framework into the caller's Python environment.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
"$PYTHON" -m pip install -e "$ROOT[sitl]"
echo "Framework installed. Headless rendering requires EGL/Mesa; video export requires FFmpeg."
