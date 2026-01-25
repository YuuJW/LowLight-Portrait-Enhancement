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

        // 累加加权的 tile 数据 (使用 OpenCV 矩阵运算优化)
        // 将单通道掩码扩展为三通道
        std::vector<cv::Mat> mask_channels(3, blend_mask);
        cv::Mat mask_3ch;
        cv::merge(mask_channels, mask_3ch);

        // 矩阵乘法: tile_float * mask_3ch
        cv::Mat weighted_tile;
        cv::multiply(tile_float, mask_3ch, weighted_tile);

        // 累加到结果缓冲区
        result_roi += weighted_tile;
        weight_roi += blend_mask;
    }

    // 归一化: result / weight_sum (使用 OpenCV 矩阵运算优化)
    // 将单通道权重扩展为三通道
    std::vector<cv::Mat> weight_channels(3);
    cv::split(weight_sum, weight_channels);  // 先 split 以获得正确的通道数
    for (int c = 0; c < 3; c++) {
        weight_channels[c] = weight_sum;  // 每个通道都使用相同的权重
    }
    cv::Mat weight_3ch;
    cv::merge(weight_channels, weight_3ch);

    // 逐元素除法，避免除零
    cv::Mat mask = weight_3ch > 0;  // 创建非零掩码
    cv::divide(result, weight_3ch, result, 1.0, -1);  // 执行除法
    result.setTo(0, ~mask);  // 将除零位置设为 0

    // 转换回 uint8
    cv::Mat result_u8;
    result.convertTo(result_u8, CV_8UC3);

    std::cout << "Merge completed" << std::endl;
    return result_u8;
}
