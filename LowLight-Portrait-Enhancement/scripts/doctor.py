#!/usr/bin/env python3
"""Repository health check for local setup."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import candidate_weight_paths, resolve_onnx_path, resolve_weights_path

SUPPORTED_PYTHON = ((3, 10), (3, 12))
REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "onnxsim": "onnxsim",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "einops": "einops",
}


def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def print_status(label: str, ok: bool, detail: str) -> None:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] {label}: {detail}")


def check_python_version() -> bool:
    current = sys.version_info[:2]
    supported = SUPPORTED_PYTHON[0] <= current <= SUPPORTED_PYTHON[1]
    detail = (
        f"{platform.python_version()} "
        f"(supported: {SUPPORTED_PYTHON[0][0]}.{SUPPORTED_PYTHON[0][1]}-"
        f"{SUPPORTED_PYTHON[1][0]}.{SUPPORTED_PYTHON[1][1]})"
    )
    print_status("Python", supported, detail)
    return supported


def check_packages() -> bool:
    missing = [package for package, module in REQUIRED_PACKAGES.items() if not is_installed(module)]
    if missing:
        print_status("Packages", False, f"missing {', '.join(missing)}")
        print("      Install with: pip install -r requirements.txt")
        return False

    print_status("Packages", True, "all required Python packages are installed")
    return True


def check_assets() -> bool:
    weight_path = resolve_weights_path()
    onnx_path = resolve_onnx_path()
    weight_candidates = candidate_weight_paths()

    weight_ok = weight_path.exists()
    onnx_ok = onnx_path.exists()

    if weight_ok:
        print_status("Weights", True, str(weight_path))
    else:
        print_status("Weights", False, f"not found; searched {', '.join(str(path) for path in weight_candidates)}")

    if onnx_ok:
        print_status("ONNX", True, str(onnx_path))
    else:
        print_status("ONNX", False, f"not exported yet; target path is {onnx_path}")
        print("      Generate it with: python deploy/export_onnx.py --simplify --verify")

    return weight_ok


def main() -> int:
    print("LowLight-Portrait-Enhancement doctor\n")
    python_ok = check_python_version()
    packages_ok = check_packages()
    assets_ok = check_assets()

    if python_ok and packages_ok and assets_ok:
        print("\nEnvironment looks ready.")
        print("Run: python tests/test_retinexformer.py --image data/LOL/lol_dataset/eval15/low/1.png")
        return 0

    print("\nEnvironment is not ready yet.")
    print("Recommended bootstrap:")
    print("  conda env create -f environment.yml")
    print("  conda activate lowlight-portrait-enhancement")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
