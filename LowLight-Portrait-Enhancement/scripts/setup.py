#!/usr/bin/env python3
"""Bootstrap local development for LowLight-Portrait-Enhancement."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import MODELS_DIR, PROJECT_ROOT, standard_directories

REFERENCES_DIR = PROJECT_ROOT.parent / "references"

RETINEXFORMER_REPO = "https://github.com/caiyuanhao1998/Retinexformer.git"
RETINEXFORMER_MIRROR = "https://gitclone.com/github.com/caiyuanhao1998/Retinexformer.git"
SUPPORTED_PYTHON = ((3, 10), (3, 12))

WEIGHTS = {
    "LOL_v2_synthetic": {
        "filename": "LOL_v2_synthetic.pth",
        "gdrive_id": "1Hj5k3BLVBIpoEBEvLaqAUxrY93X8jkem",
        "description": "Default synthetic checkpoint",
    },
    "LOL_v2_real": {
        "filename": "LOL_v2_real.pth",
        "gdrive_id": "1WYQjBqpGgEqPwPWkEzJgbbMONZsNBEbw",
        "description": "Real-scene checkpoint",
    },
    "LOL_v1": {
        "filename": "LOL_v1.pth",
        "gdrive_id": "1tfYsOMrwn75miJipjLCf_fMPlfWRQr-m",
        "description": "Legacy LOL-v1 checkpoint",
    },
}


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_step(index: int, message: str) -> None:
    print(f"\n[{index}] {message}")


def check_python_version() -> bool:
    current = sys.version_info[:2]
    supported = SUPPORTED_PYTHON[0] <= current <= SUPPORTED_PYTHON[1]
    supported_text = (
        f"{SUPPORTED_PYTHON[0][0]}.{SUPPORTED_PYTHON[0][1]}-"
        f"{SUPPORTED_PYTHON[1][0]}.{SUPPORTED_PYTHON[1][1]}"
    )
    print(f"Python: {sys.version.split()[0]} (supported: {supported_text})")
    if not supported:
        print("Warning: PyTorch wheels are typically unavailable for this Python version.")
        print("         Create a 3.10 environment with: conda env create -f environment.yml")
    return supported


def ensure_directories() -> None:
    directories = standard_directories() + [
        PROJECT_ROOT / "deploy" / "cpp" / "build",
        REFERENCES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            relative = directory.relative_to(PROJECT_ROOT)
            print(f"  ensured {relative}")
        except ValueError:
            print(f"  ensured {directory}")


def check_git() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_command(command: list[str], cwd: Path | None = None) -> bool:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        return False
    return True


def clone_reference(use_mirror: bool) -> bool:
    target_dir = REFERENCES_DIR / "Retinexformer"
    if target_dir.exists():
        print(f"  reference already present: {target_dir}")
        return True

    if not check_git():
        print("  git is not available; skipping reference checkout")
        return False

    repo_url = RETINEXFORMER_MIRROR if use_mirror else RETINEXFORMER_REPO
    print(f"  cloning {repo_url}")
    if run_command(["git", "clone", "--depth", "1", repo_url, str(target_dir)]):
        return True

    if not use_mirror:
        print("  primary clone failed, retrying with mirror")
        return run_command(["git", "clone", "--depth", "1", RETINEXFORMER_MIRROR, str(target_dir)])

    return False


def try_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def print_manual_weight_links(destination: Path) -> None:
    print("  automatic download unavailable. Download a checkpoint manually:")
    print("    https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV")
    print("    https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2")
    print(f"  then place it under: {destination}")


def download_weights(weight_key: str) -> bool:
    weight_info = WEIGHTS[weight_key]
    destination = MODELS_DIR / weight_info["filename"]
    if destination.exists():
        print(f"  weights already present: {destination}")
        return True

    if not try_import("gdown"):
        print("  gdown is not installed")
        print_manual_weight_links(MODELS_DIR)
        return False

    import gdown

    url = f"https://drive.google.com/uc?id={weight_info['gdrive_id']}"
    print(f"  downloading {weight_info['description']} -> {destination}")
    gdown.download(url, str(destination), quiet=False)
    success = destination.exists()
    if not success:
        print_manual_weight_links(MODELS_DIR)
    return success


def check_python_dependencies() -> bool:
    required = {
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

    missing = [package for package, module in required.items() if not try_import(module)]
    if missing:
        print(f"  missing packages: {', '.join(missing)}")
        print("  install with: pip install -r requirements.txt")
        return False

    print("  Python dependencies look installed")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap local development")
    parser.add_argument("--skip-weights", action="store_true", help="Skip checkpoint download")
    parser.add_argument(
        "--fetch-reference",
        action="store_true",
        help="Clone the upstream RetinexFormer repository into ../references",
    )
    parser.add_argument("--mirror", action="store_true", help="Use the gitclone mirror for reference checkout")
    parser.add_argument(
        "--weight",
        choices=list(WEIGHTS.keys()),
        default="LOL_v2_synthetic",
        help="Checkpoint to download into models/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_header("LowLight-Portrait-Enhancement bootstrap")
    print(f"Project root: {PROJECT_ROOT}")

    print_step(1, "Checking Python runtime")
    python_ok = check_python_version()

    print_step(2, "Ensuring standard directories")
    ensure_directories()

    print_step(3, "Checking Python dependencies")
    deps_ok = check_python_dependencies()

    print_step(4, "Preparing model weights")
    weights_ok = True
    if args.skip_weights:
        print("  skipped checkpoint download")
    else:
        weights_ok = download_weights(args.weight)

    print_step(5, "Optional reference checkout")
    reference_ok = True
    if args.fetch_reference:
        reference_ok = clone_reference(args.mirror)
    else:
        print("  skipped; use --fetch-reference if you need the upstream repo")

    print_header("Bootstrap summary")
    print(f"Python version compatible: {'yes' if python_ok else 'no'}")
    print(f"Python dependencies ready: {'yes' if deps_ok else 'no'}")
    print(f"Checkpoint available: {'yes' if weights_ok else 'no'}")
    print(f"Reference checkout ready: {'yes' if reference_ok else 'no'}")

    if python_ok and deps_ok:
        print("\nNext steps:")
        print("  python scripts/doctor.py")
        print("  python tests/test_retinexformer.py --image data/LOL/lol_dataset/eval15/low/1.png")
        print("  python deploy/export_onnx.py --simplify --verify")
        return 0

    print("\nRecommended environment setup:")
    print("  conda env create -f environment.yml")
    print("  conda activate lowlight-portrait-enhancement")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
