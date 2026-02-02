# ONNX → DFP MemryX Conversion Toolchain (Docker)

This toolchain provides a **Docker-based workflow** for compiling ONNX models into MemryX
artifacts (`.dfp` + `chain.json`) using the MemryX Neural Compiler.

The toolchain is designed to be:
- reproducible
- isolated from host Python environments
- easy to run on supported Linux systems
- compatible with MemryX 2.x tooling

---

## Supported Platforms

### Host system
- Linux (x86_64 or ARM64)
- Docker installed and running

### Container environment
- Debian-based Linux
- Python **3.9 – 3.12**
- MemryX SDK (installed at image build time)

---

## Repository Structure (relevant)

```
compilation-toolchain/
├── Dockerfile
├── build-toolchain.sh
├── requirements.txt
├── setup.py
├── conversion_block/
│   ├── main.py
│   ├── utils.py
│   ├── logger.py
│   └── __init__.py
└── src/
    └── run-block.sh
```

---

## Build the Toolchain Image

From the `compilation-toolchain` directory:

```bash
./build-toolchain.sh
```

This will:
1. Build the Docker image `onnx-to-memryx:latest`
2. Save it as a tarball under `build/onnx-to-memryx-latest.tar`

You can later load the image with:

```bash
docker load -i build/onnx-to-memryx-latest.tar
```

---

## Prepare Input and Output Directories

Create directories on the host for inputs and outputs:

```bash
mkdir -p artifacts out
```

---

## Prepare the Input Model Archive

The toolchain expects a **single zip archive** as input instead of a raw ONNX file.

The archive **must** contain:
- One or more `.onnx` model files
- A required `ncconfig.json` file describing NeuralCompiler configuration

### Example (single model)

```
model.zip
├── ncconfig.json
├── model.onnx
```

### Example (co-mapped models)

```
model.zip
├── ncconfig.json
├── model_a.onnx
├── model_b.onnx
```

All model paths listed in `ncconfig.json` must be **relative to the root of the archive**.

---

## ncconfig.json

`ncconfig.json` is a **flat JSON object** whose keys are passed directly to
`NeuralCompiler.set_config()`.

### Minimal example

```json
{
  "models": ["model.onnx"],
  "num_chips": 4,
  "chip_gen": "mx3",
  "autocrop": true
}
```

### Co-mapping example

```json
{
  "models": ["model_a.onnx", "model_b.onnx"],
  "num_chips": 8,
  "effort": "hard"
}
```

### Using compiler extensions

```json
{
  "models": ["model.onnx"],
  "extensions": [
    "Yolov10"
  ]
}
```

Notes:
- `models` **must** be a non-empty list
- `extensions`, if provided, **must** be a JSON list
- All configuration validation is performed by the NeuralCompiler itself

For a full list of supported parameters, refer to:
https://developer.memryx.com/tools/neural_compiler.html

---

## Run the Conversion

The container expects **two positional arguments**:

```
<model_archive.zip> <output_dir>
```

Example:

```bash
docker run --rm -it   -v "$(pwd)/artifacts:/artifacts"   -v "$(pwd)/out:/out"   onnx-to-memryx:latest   /artifacts/model.zip   /out
```

---

## Toolchain-Enforced Behavior

The following settings are **controlled by the toolchain** and cannot be overridden
via `ncconfig.json`:

| Setting | Behavior |
|------|--------|
| `models` | Resolved from the extracted archive |
| `dfp_fname` | Automatically generated with a timestamp |
| `num_processes` | Forced to `(CPU cores - 2)`, minimum 1 |

All other configuration values are passed unchanged to the NeuralCompiler.

---

## Output Layout

For a model named `model.onnx`, outputs are written as:

```
out/
└── model/
    └── model.zip
```

The output zip contains:
- Compiled `.dfp`
- Generated `chain.json`
- Optional `*_pre.onnx` / `*_post.onnx` (if emitted)

Artifact filenames inside the zip are prefixed with a **human-readable UTC timestamp** to ensure uniqueness:

```
20260130_154233_model.dfp
20260130_154233_model_pre.onnx
20260130_154233_model_post.onnx
chain.json
```

---

## Notes on Rebuilding

Any changes to:
- `conversion_block/*.py`
- `run-block.sh`
- `Dockerfile`
- `requirements.txt`

require rebuilding the Docker image:

```bash
docker build -t onnx-to-memryx:latest .
```

---

## Design Notes

- The toolchain runs entirely inside Docker to avoid host dependency conflicts
- Each conversion run uses an isolated temporary workspace
- Output artifacts are copied and packaged to avoid filename collisions
- The zip-based input contract enables reproducible, configuration-driven compilation

---

## Troubleshooting

### Docker not found

Ensure Docker is installed and the daemon is running:

```bash
docker info
```
