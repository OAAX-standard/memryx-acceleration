set -e

cd "$(dirname "$0")" || exit 1

rm -rf build 2&> /dev/null || true
mkdir build

# Build the toolchain as a Docker image
docker build -t onnx-to-memryx:latest .

# Save the Docker image as a tarball
docker save onnx-to-memryx:latest -o ./build/onnx-to-memryx-latest.tar

# You can run the conversion toolchain using the following command:
#docker load  -i ./build/onnx-to-memryx-latest.tar
#docker run -v ./memryx-deps:/app/memryx-deps -v ./artifacts:/app2 onnx-to-memryx:latest /app2/model.zip /app2

