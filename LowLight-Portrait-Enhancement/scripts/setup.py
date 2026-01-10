#!/usr/bin/env python3
"""
LowLight-Portrait-Enhancement 环境配置脚本

在新电脑上克隆项目后运行此脚本，自动完成：
1. 克隆 RetinexFormer 官方仓库
2. 下载预训练权重
3. 创建必要的目录结构
4. 验证环境配置

使用方法:
    python scripts/setup.py [--skip-weights] [--mirror]

参数:
    --skip-weights  跳过权重下载（如果网络不好，可手动下载）
    --mirror        使用镜像加速克隆
"""

import os
import sys
import subprocess
import argparse
import urllib.request
import hashlib
from pathlib import Path


# 项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
REFERENCES_DIR = PROJECT_ROOT.parent / "references"
MODELS_DIR = PROJECT_ROOT / "deploy" / "models"

# RetinexFormer 仓库
RETINEXFORMER_REPO = "https://github.com/caiyuanhao1998/Retinexformer.git"
RETINEXFORMER_MIRROR = "https://gitclone.com/github.com/caiyuanhao1998/Retinexformer.git"

# 预训练权重配置
WEIGHTS = {
    "LOL_v2_synthetic": {
        "filename": "LOL_v2_synthetic.pth",
        "gdrive_id": "1Hj5k3BLVBIpoEBEvLaqAUxrY93X8jkem",
        "description": "LOL-v2 Synthetic (PSNR 29.04, 推荐)",
    },
    "LOL_v2_real": {
        "filename": "LOL_v2_real.pth",
        "gdrive_id": "1WYQjBqpGgEqPwPWkEzJgbbMONZsNBEbw",
        "description": "LOL-v2 Real (PSNR 27.71)",
    },
    "LOL_v1": {
        "filename": "LOL_v1.pth",
        "gdrive_id": "1tfYsOMrwn75miJipjLCf_fMPlfWRQr-m",
        "description": "LOL-v1 (PSNR 27.18)",
    },
}

# 下载链接（备用）
DOWNLOAD_LINKS = """
预训练权重下载链接:

Google Drive (所有权重):
  https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV

百度网盘:
  https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2
  提取码: cyh2

下载后请放到: {models_dir}
"""


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step, text):
    """打印步骤"""
    print(f"\n[{step}] {text}")


def run_command(cmd, cwd=None, check=True):
    """运行命令"""
    print(f"  > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        print(f"  错误: {result.stderr}")
        return False
    return True


def check_git():
    """检查 git 是否可用"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clone_retinexformer(use_mirror=False):
    """克隆 RetinexFormer 仓库"""
    target_dir = REFERENCES_DIR / "Retinexformer"

    if target_dir.exists():
        print(f"  RetinexFormer 已存在: {target_dir}")
        return True

    # 创建 references 目录
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    # 选择仓库 URL
    repo_url = RETINEXFORMER_MIRROR if use_mirror else RETINEXFORMER_REPO
    print(f"  克隆地址: {repo_url}")

    # 尝试克隆
    cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
    if run_command(cmd):
        print("  克隆成功!")
        return True

    # 如果失败，尝试镜像
    if not use_mirror:
        print("  尝试使用镜像...")
        cmd = ["git", "clone", "--depth", "1", RETINEXFORMER_MIRROR, str(target_dir)]
        if run_command(cmd, check=False):
            print("  镜像克隆成功!")
            return True

    return False


def download_from_gdrive(file_id, destination):
    """从 Google Drive 下载文件"""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        return destination.exists()
    except ImportError:
        print("  需要安装 gdown: pip install gdown")
        return False
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def download_weights(weight_key="LOL_v2_synthetic"):
    """下载预训练权重"""
    weight_info = WEIGHTS.get(weight_key)
    if not weight_info:
        print(f"  未知权重: {weight_key}")
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / weight_info["filename"]

    if destination.exists():
        print(f"  权重已存在: {destination}")
        return True

    print(f"  下载 {weight_info['description']}...")

    # 尝试用 gdown 下载
    if download_from_gdrive(weight_info["gdrive_id"], destination):
        print(f"  下载成功: {destination}")
        return True

    # 下载失败，提供手动下载链接
    print(DOWNLOAD_LINKS.format(models_dir=MODELS_DIR))
    return False


def create_directories():
    """创建必要的目录结构"""
    dirs = [
        PROJECT_ROOT / "deploy" / "models",
        PROJECT_ROOT / "deploy" / "cpp" / "include",
        PROJECT_ROOT / "deploy" / "cpp" / "src",
        PROJECT_ROOT / "deploy" / "scripts",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "benchmarks" / "results",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  {d.relative_to(PROJECT_ROOT)}/")

    return True


def check_python_deps():
    """检查 Python 依赖"""
    required = ["torch", "torchvision", "onnx", "cv2", "numpy"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            # cv2 对应 opencv-python
            if pkg == "cv2":
                missing.append("opencv-python")
            else:
                missing.append(pkg)

    if missing:
        print(f"  缺少依赖: {', '.join(missing)}")
        print(f"  安装命令: pip install {' '.join(missing)}")
        return False

    print("  所有依赖已安装")
    return True


def verify_setup():
    """验证环境配置"""
    issues = []

    # 检查 RetinexFormer
    retinexformer_dir = REFERENCES_DIR / "Retinexformer"
    if not retinexformer_dir.exists():
        issues.append("RetinexFormer 仓库未克隆")

    # 检查权重文件
    weights_found = list(MODELS_DIR.glob("*.pth")) if MODELS_DIR.exists() else []
    if not weights_found:
        issues.append("未找到预训练权重文件")

    return issues


def main():
    parser = argparse.ArgumentParser(description="LowLight-Portrait-Enhancement 环境配置")
    parser.add_argument("--skip-weights", action="store_true", help="跳过权重下载")
    parser.add_argument("--mirror", action="store_true", help="使用镜像加速克隆")
    parser.add_argument("--weight", choices=list(WEIGHTS.keys()),
                       default="LOL_v2_synthetic", help="选择权重版本")
    args = parser.parse_args()

    print_header("LowLight-Portrait-Enhancement 环境配置")
    print(f"项目目录: {PROJECT_ROOT}")

    # Step 1: 检查 git
    print_step(1, "检查 Git...")
    if not check_git():
        print("  错误: Git 未安装或不可用")
        sys.exit(1)
    print("  Git 可用")

    # Step 2: 创建目录
    print_step(2, "创建目录结构...")
    create_directories()

    # Step 3: 克隆 RetinexFormer
    print_step(3, "克隆 RetinexFormer 仓库...")
    if not clone_retinexformer(use_mirror=args.mirror):
        print("  警告: 克隆失败，请手动克隆:")
        print(f"    git clone {RETINEXFORMER_REPO} {REFERENCES_DIR / 'Retinexformer'}")

    # Step 4: 下载权重
    if not args.skip_weights:
        print_step(4, "下载预训练权重...")
        if not download_weights(args.weight):
            print("  请手动下载权重文件")
    else:
        print_step(4, "跳过权重下载 (--skip-weights)")

    # Step 5: 检查 Python 依赖
    print_step(5, "检查 Python 依赖...")
    check_python_deps()

    # Step 6: 验证
    print_step(6, "验证配置...")
    issues = verify_setup()

    print_header("配置完成")

    if issues:
        print("待处理事项:")
        for issue in issues:
            print(f"  - {issue}")
        print(DOWNLOAD_LINKS.format(models_dir=MODELS_DIR))
    else:
        print("环境配置完成! 可以开始开发了。")
        print("\n下一步:")
        print("  1. 测试模型: python tests/test_retinexformer.py")
        print("  2. 导出 ONNX: python deploy/export_onnx.py")


if __name__ == "__main__":
    main()
