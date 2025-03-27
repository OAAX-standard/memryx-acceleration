#!/usr/bin/env bash

set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <onnx-path> <output-dir>" >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
cd $DIR || exit 1

onnx_path="$1"
output_dir="$2"

echo "Installing the MemryX SDK. Please enter your credentials when prompted."
pip install --extra-index-url https://developer.memryx.com/pip memryx==0.10.0

conversion_block --onnx-path "$onnx_path" --output-dir "$output_dir"
