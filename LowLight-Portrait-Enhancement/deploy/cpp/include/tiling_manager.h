#ifndef TILING_MANAGER_H
#define TILING_MANAGER_H

#include <vector>
#include <opencv2/opencv.hpp>

struct Tile {
    cv::Rect roi;       // 在原图中的位置
    cv::Mat data;       // tile 数据 (512x512)
    int row, col;       // tile 索引
};

class TilingManager {
public:
    TilingManager(int tile_size = 512, int overlap = 32);

    std::vector<Tile> split(const cv::Mat& image);
    cv::Mat merge(const std::vector<Tile>& tiles, cv::Size original_size);

private:
    int tile_size_;
    int overlap_;

    cv::Mat compute_blend_mask(int tile_w, int tile_h);
};

#endif
