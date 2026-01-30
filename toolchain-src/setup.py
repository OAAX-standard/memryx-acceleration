from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _read_requirements(path: Path) -> list[str]:
    reqs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="conversion_block",
    version="1.0.0",
    description="Compile ONNX models into MemryX artifacts",
    packages=find_packages(),
    python_requires=">=3.9,<3.13",
    install_requires=_read_requirements(ROOT / "requirements.txt"),
    entry_points={"console_scripts": ["conversion_block=conversion_block.main:cli"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
    ],
)
