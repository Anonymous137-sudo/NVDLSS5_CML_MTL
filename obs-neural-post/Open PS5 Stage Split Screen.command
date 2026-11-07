#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
exec "$project_dir/tools/open-stage-split-screen.sh"
