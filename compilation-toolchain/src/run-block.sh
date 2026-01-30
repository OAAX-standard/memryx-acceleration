#!/usr/bin/env bash

# Exit immediately on error (-e), on unset variables (-u),
# and if any command in a pipeline fails (pipefail).
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1


# Print usage information and exit.
usage() {
  cat >&2 <<'EOF'
Usage:
  run-block.sh --onnx-path <path> --output-dir <dir> [--num_chips <n>] [--extensions <ext>]

Examples:
  run-block.sh --onnx-path model.onnx --output-dir out
  run-block.sh --onnx-path model.onnx --output-dir out --num_chips 8
  run-block.sh --onnx-path model.onnx --output-dir out --extensions "foo=1,bar=2"
EOF
  exit 1
}

# Default values for optional arguments.
num_chips=4
extensions=""

# Required arguments (initialized empty for validation later).
onnx_path=""
output_dir=""

# Parse command-line arguments.
# Long-option parsing is implemented manually for portability.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx-path)
      [[ $# -ge 2 ]] || usage
      onnx_path="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || usage
      output_dir="$2"
      shift 2
      ;;
    --num_chips|--num-chips)
      [[ $# -ge 2 ]] || usage
      num_chips="$2"
      shift 2
      ;;
    --extensions)
      [[ $# -ge 2 ]] || usage
      extensions="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

# Validate required arguments.
[[ -n "$onnx_path" && -n "$output_dir" ]] || usage

# Resolve the directory containing this script and switch to it.
# This ensures relative paths behave consistently.
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

# Construct the conversion command as an array to preserve argument boundaries.
cmd=(
  conversion_block
  --onnx-path "$onnx_path"
  --output-dir "$output_dir"
  --num_chips "$num_chips"
)

# Append extensions only if provided.
if [[ -n "$extensions" ]]; then
  cmd+=(--extensions "$extensions")
fi

# Echo the command for visibility/debugging.
echo "Running: ${cmd[*]}"

# Execute the conversion.
"${cmd[@]}"
