"""准备 INT8 量化校准图片"""
import os
import cv2
import glob

def main():
    # 源图片目录
    src_dir = "../data/LOL/lol_dataset/eval15/low"
    # 目标校准目录
    dst_dir = "calibration"
    # 目标尺寸 (与 ONNX 导出时一致)
    target_size = (512, 512)

    os.makedirs(dst_dir, exist_ok=True)

    # 获取所有 png 图片 (排除 enhanced 结果)
    patterns = [os.path.join(src_dir, "*.png")]
    images = []
    for p in patterns:
        images.extend(glob.glob(p))

    # 过滤掉 enhanced 图片
    images = [img for img in images if "enhanced" not in img]

    print(f"Found {len(images)} images for calibration")

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        # Resize to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # 保存为校准图片
        dst_path = os.path.join(dst_dir, f"calib_{i:03d}.png")
        cv2.imwrite(dst_path, img_resized)
        print(f"Saved: {dst_path}")

    # 生成图片列表文件 (ncnn2table 需要)
    list_file = os.path.join(dst_dir, "imagelist.txt")
    with open(list_file, "w") as f:
        for i in range(len(images)):
            f.write(f"calib_{i:03d}.png\n")

    print(f"\nCalibration images prepared: {len(images)}")
    print(f"Image list saved to: {list_file}")

if __name__ == "__main__":
    main()
