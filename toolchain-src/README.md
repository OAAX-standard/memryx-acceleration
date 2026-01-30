# ONNX → DFP MemryX Conversion Toolchain (Docker)

This toolchain provides a **Docker-based workflow** for compiling ONNX models into MemryX artifacts (`.dfp` + `chain.json`) using the MemryX Neural Compiler.

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

Copy your ONNX model into `artifacts/`:

```bash
cp /absolute/path/to/model.onnx artifacts/model.onnx
```

---

## Run the Conversion

Run the container and invoke the conversion tool:

```bash
docker run --rm -it \
  -v "$(pwd)/artifacts:/artifacts" \
  -v "$(pwd)/out:/out" \
  onnx-to-memryx:latest \
  --onnx-path /artifacts/model.onnx \
  --output-dir /out \
  --num_chips 4
```

### Optional arguments

#### Number of chips
```bash
--num_chips 2
```

#### Compiler extensions
Extensions are passed as a list and support multiple formats:

```bash
--extensions foo
--extensions foo --extensions bar
--extensions "foo,bar"
--extensions '["foo","bar"]'
```
For more details about MemryX Neural Compiler Extensions please read [MX Neural Compiler Page](https://developer.memryx.com/tools/neural_compiler.html#neural-compiler-extensions)

---

## Output Layout

For a model named `model.onnx`, outputs are written as:

```
out/
└── model/
    └── model.zip
```

The zip file contains:
- compiled `.dfp`
- generated `chain.json`
- optional `*_pre.onnx` / `*_post.onnx` (if emitted by the compiler)

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

- The toolchain runs entirely inside Docker to avoid host dependency conflicts.
- The MemryX SDK is installed at **image build time** for faster repeated conversions.
- Each conversion run uses an isolated temporary workspace inside the container.
- Outputs are copied and packaged to avoid filename collisions across runs.

---

## Troubleshooting

### Docker not found
Ensure Docker is installed and the daemon is running:

```bash
docker info
```