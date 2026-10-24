# Contributing to OAAX MemryX Acceleration

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (bug reports, commit messages and DCO sign-off, code style).

## How to contribute

1. **Open an issue** to discuss the bug or feature before writing code.
2. **Fork the repository** and create a branch from `release`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run the tests** locally (see below) to confirm nothing is broken.
5. **Sign off every commit** (`git commit -s`) and open a pull request.

## Development environment

The repository has two parts: the runtime in `runtime-src/` (C++, CMake) and the conversion toolchain in
`toolchain-src/` (Docker, see its [README](toolchain-src/README.md)). There is no CI and no automated test suite yet,
so describe in your PR how you built and verified your change.
