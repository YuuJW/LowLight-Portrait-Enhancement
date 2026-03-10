/**
 * @file tiling_manager.h
 * @brief 图像分块（Tiling）管理器
 *
 * 负责将大图分割为固定尺寸的 tiles，并在推理后将 tiles 融合回完整图像
 * 使用重叠区域和线性融合技术消除边界伪影
 */

#ifndef TILING_MANAGER_H
#define TILING_MANAGER_H

#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Tile 结构体
 *
 * 表示图像的一个分块，包含位置信息和数据
 */
struct Tile {
    cv::Rect roi;       ///< 在原图中的位置和尺寸 (x, y, width, height)
    cv::Mat data;       ///< tile 数据 (通常是 512x512, BGR 格式)
    int row, col;       ///< tile 在网格中的索引 (row, col)
};

/**
 * @brief 图像分块管理器
 *
 * 功能:
 * 1. split(): 将大图分割为固定尺寸的 tiles（带重叠区域）
 * 2. merge(): 将处理后的 tiles 融合回完整图像（线性融合）
 *
 * 算法原理:
 * - 分割时，相邻 tiles 之间有 overlap 像素的重叠
 * - 融合时，在重叠区域使用线性权重进行融合
 * - 边界处权重从 0 渐变到 1，中心区域权重为 1
 * - 这样可以消除 tile 边界的伪影
 *
 * 使用示例:
 * @code
 * TilingManager manager(512, 32);  // 512x512 tiles, 32px overlap
 * auto tiles = manager.split(large_image);
 * // ... 处理每个 tile ...
 * cv::Mat result = manager.merge(tiles, large_image.size());
 * @endcode
 */
class TilingManager {
public:
    /**
     * @brief 构造函数
     * @param tile_size 单个 tile 的尺寸（正方形边长，默认 512）
     * @param overlap 相邻 tiles 之间的重叠像素数（默认 32）
     * @param verbose 是否输出详细日志（默认 true）
     *
     * 说明:
     * - tile_size 应与模型训练尺寸一致（RetinexFormer 使用 512）
     * - overlap 越大，边界伪影越少，但计算量越大
     * - 推荐 overlap = tile_size / 16 (例如 512/16 = 32)
     */
    TilingManager(int tile_size = 512, int overlap = 32, bool verbose = true);

    /**
     * @brief 将大图分割为多个 tiles
     * @param image 输入图像（任意尺寸）
     * @return tiles 向量，每个 tile 包含位置信息和数据
     *
     * 算法说明:
     * 1. 计算 stride = tile_size - overlap（相邻 tile 起点的间距）
     * 2. 计算需要的 tile 数量: num_x, num_y
     * 3. 遍历每个 tile 位置，提取数据
     * 4. 如果 tile 尺寸不足，使用 BORDER_REFLECT 填充到 tile_size
     *
     * 注意:
     * - 边界 tile 可能小于 tile_size，会自动填充
     * - 填充使用 BORDER_REFLECT 模式保证边界连续性
     */
    std::vector<Tile> split(const cv::Mat& image);

    /**
     * @brief 将处理后的 tiles 融合回完整图像
     * @param tiles 处理后的 tiles 向量
     * @param original_size 原始图像尺寸
     * @return 融合后的完整图像
     *
     * 算法说明:
     * 1. 创建浮点数累加缓冲区 (result) 和权重和缓冲区 (weight_sum)
     * 2. 遍历每个 tile:
     *    a. 计算该 tile 的混合权重掩码（边界渐变）
     *    b. 将 tile 数据乘以权重后累加到 result
     *    c. 将权重累加到 weight_sum
     * 3. 最后用 weight_sum 归一化 result
     * 4. 转换回 uint8 格式
     *
     * 优化说明:
     * - 使用 OpenCV 矩阵运算替代逐像素循环
     * - 性能提升约 2-5 倍
     */
    cv::Mat merge(const std::vector<Tile>& tiles, cv::Size original_size);

private:
    int tile_size_;     ///< Tile 尺寸（正方形边长）
    int overlap_;       ///< 重叠像素数
    bool verbose_;      ///< 是否输出详细日志

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
    cv::Mat compute_blend_mask(
        int tile_w,
        int tile_h,
        bool touches_top,
        bool touches_bottom,
        bool touches_left,
        bool touches_right
    );
};

#endif
