#!/usr/bin/env bash

# Exit immediately on error (-e), on unset variables (-u),
# and if any command in a pipeline fails (pipefail).
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run-block.sh <model.zip> <output-dir>

Arguments:
  model.zip     Zip archive containing model.onnx and ncconfig.json
  output-dir    Directory where output artifacts will be written

Example:
  run-block.sh /artifacts/model.zip /out
EOF
  exit 1
}

# Expect exactly two positional arguments.
[[ $# -eq 2 ]] || usage

model_zip="$1"
output_dir="$2"

# Provide early, readable errors for missing input paths.
[[ -f "$model_zip" ]] || { echo "Input archive not found: $model_zip" >&2; exit 2; }
[[ -d "$output_dir" ]] || { echo "Output directory not found: $output_dir" >&2; exit 2; }

# Construct the command as an array to preserve argument boundaries.
cmd=(
  conversion_block
  "$model_zip"
  "$output_dir"
)

# Echo the command for visibility/debugging.
echo "Running: ${cmd[*]}"

# Execute the conversion.
"${cmd[@]}"
