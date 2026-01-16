#include "tiling_manager.h"
#include <cmath>
#include <iostream>

/**
 * @brief 构造函数
 * @param tile_size 单个 tile 的尺寸 (默认 512x512)
 * @param overlap 相邻 tiles 之间的重叠像素数 (默认 32px)
 */
TilingManager::TilingManager(int tile_size, int overlap)
    : tile_size_(tile_size), overlap_(overlap) {
    std::cout << "TilingManager initialized: tile_size=" << tile_size_
              << ", overlap=" << overlap_ << std::endl;
}

/**
 * @brief 将大图像分割为多个 tiles
 * @param image 输入图像
 * @return 包含所有 tiles 的向量
 *
 * 算法说明:
 * 1. 计算 stride = tile_size - overlap (相邻 tile 起点的间距)
 * 2. 计算需要的 tile 数量: num_x, num_y
 * 3. 遍历每个 tile 位置，提取数据
 * 4. 如果 tile 尺寸不足，使用 BORDER_REFLECT 填充到 tile_size
 */
std::vector<Tile> TilingManager::split(const cv::Mat& image) {
    std::vector<Tile> tiles;

    // 计算 tile 的步长 (相邻 tile 起点的间距)
    int stride = tile_size_ - overlap_;

    // 计算需要的 tile 数量
    // 公式: ceil((image_size - overlap) / stride)
    int num_x = std::ceil((float)(image.cols - overlap_) / stride);
    int num_y = std::ceil((float)(image.rows - overlap_) / stride);

    std::cout << "Splitting image " << image.cols << "x" << image.rows
              << " into " << num_x << "x" << num_y << " tiles" << std::endl;

    // 遍历每个 tile 位置
    for (int row = 0; row < num_y; row++) {
        for (int col = 0; col < num_x; col++) {
            // 计算 tile 在原图中的位置
            int x = col * stride;
            int y = row * stride;

            // 计算实际的 tile 尺寸 (边界 tile 可能小于 tile_size)
            int w = std::min(tile_size_, image.cols - x);
            int h = std::min(tile_size_, image.rows - y);

            // 提取 tile 数据
            cv::Rect roi(x, y, w, h);
            cv::Mat tile_data = image(roi).clone();

            // 如果 tile 尺寸不足 tile_size，进行填充
            // 使用 BORDER_REFLECT 模式保证边界连续性
            if (w < tile_size_ || h < tile_size_) {
                cv::copyMakeBorder(tile_data, tile_data,
                    0, tile_size_ - h,      // 上下填充
                    0, tile_size_ - w,      // 左右填充
                    cv::BORDER_REFLECT);
            }

            // 保存 tile 信息
            tiles.push_back({roi, tile_data, row, col});
        }
    }

    return tiles;
}

/**
 * @brief 计算 tile 的混合权重掩码
 * @param tile_w tile 的宽度
 * @param tile_h tile 的高度
 * @return 混合权重掩码 (CV_32FC1, 值域 [0,1])
 *
 * 算法说明:
 * 1. 初始化全 1 的掩码
 * 2. 在重叠区域 (overlap_) 创建线性渐变
 * 3. 边界处权重从 0 渐变到 1，中心区域权重为 1
 * 4. 这样在 merge 时，重叠区域会平滑过渡
 */
cv::Mat TilingManager::compute_blend_mask(int tile_w, int tile_h) {
    // 创建全 1 的掩码
    cv::Mat mask(tile_h, tile_w, CV_32FC1, cv::Scalar(1.0f));

    // 在重叠区域创建线性渐变
    // 上边界: 权重从 0 → 1
    for (int y = 0; y < std::min(overlap_, tile_h); y++) {
        float alpha = (float)y / overlap_;
        mask.row(y) *= alpha;
    }

    // 下边界: 权重从 1 → 0
    for (int y = 0; y < std::min(overlap_, tile_h); y++) {
        float alpha = (float)y / overlap_;
        int row_idx = tile_h - 1 - y;
        if (row_idx >= 0) {
            mask.row(row_idx) *= alpha;
        }
    }

    // 左边界: 权重从 0 → 1
    for (int x = 0; x < std::min(overlap_, tile_w); x++) {
        float alpha = (float)x / overlap_;
        mask.col(x) *= alpha;
    }

    // 右边界: 权重从 1 → 0
    for (int x = 0; x < std::min(overlap_, tile_w); x++) {
        float alpha = (float)x / overlap_;
        int col_idx = tile_w - 1 - x;
        if (col_idx >= 0) {
            mask.col(col_idx) *= alpha;
        }
    }

    return mask;
}

/**
 * @brief 将处理后的 tiles 融合回完整图像
 * @param tiles 处理后的 tiles 向量
 * @param original_size 原始图像尺寸
 * @return 融合后的完整图像
 *
 * 算法说明:
 * 1. 创建浮点数累加缓冲区 (result) 和权重和缓冲区 (weight_sum)
 * 2. 遍历每个 tile:
 *    a. 计算该 tile 的混合权重掩码
 *    b. 将 tile 数据乘以权重后累加到 result
 *    c. 将权重累加到 weight_sum
 * 3. 最后用 weight_sum 归一化 result
 * 4. 转换回 uint8 格式
 */
cv::Mat TilingManager::merge(const std::vector<Tile>& tiles, cv::Size original_size) {
    // 创建浮点数累加缓冲区
    cv::Mat result = cv::Mat::zeros(original_size, CV_32FC3);
    cv::Mat weight_sum = cv::Mat::zeros(original_size, CV_32FC1);

    std::cout << "Merging " << tiles.size() << " tiles..." << std::endl;

    // 遍历每个 tile
    for (const auto& tile : tiles) {
        // 计算该 tile 的混合权重掩码
        cv::Mat blend_mask = compute_blend_mask(tile.roi.width, tile.roi.height);

        // 将 tile 数据转换为浮点数
        cv::Mat tile_float;
        tile.data(cv::Rect(0, 0, tile.roi.width, tile.roi.height)).convertTo(tile_float, CV_32FC3);

        // 提取目标区域
        cv::Mat result_roi = result(tile.roi);
        cv::Mat weight_roi = weight_sum(tile.roi);

        // 累加加权的 tile 数据
        for (int y = 0; y < tile.roi.height; y++) {
            for (int x = 0; x < tile.roi.width; x++) {
                float weight = blend_mask.at<float>(y, x);
                cv::Vec3f pixel = tile_float.at<cv::Vec3f>(y, x);

                result_roi.at<cv::Vec3f>(y, x) += pixel * weight;
                weight_roi.at<float>(y, x) += weight;
            }
        }
    }

    // 归一化: result / weight_sum
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            float weight = weight_sum.at<float>(y, x);
            if (weight > 0) {
                result.at<cv::Vec3f>(y, x) /= weight;
            }
        }
    }

    // 转换回 uint8
    cv::Mat result_u8;
    result.convertTo(result_u8, CV_8UC3);

    std::cout << "Merge completed" << std::endl;
    return result_u8;
}
