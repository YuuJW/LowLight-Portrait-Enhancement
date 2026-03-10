#include "tiling_manager.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

TilingManager::TilingManager(int tile_size, int overlap, bool verbose)
    : tile_size_(tile_size), overlap_(overlap), verbose_(verbose) {
    if (tile_size_ <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    if (overlap_ < 0 || overlap_ >= tile_size_) {
        throw std::invalid_argument("overlap must be in [0, tile_size)");
    }

    if (verbose_) {
        std::cout << "TilingManager initialized: tile_size=" << tile_size_
                  << ", overlap=" << overlap_ << std::endl;
    }
}

std::vector<Tile> TilingManager::split(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("input image is empty");
    }

    std::vector<Tile> tiles;
    int stride = tile_size_ - overlap_;
    int num_x = std::ceil(static_cast<float>(image.cols - overlap_) / stride);
    int num_y = std::ceil(static_cast<float>(image.rows - overlap_) / stride);

    if (verbose_) {
        std::cout << "Splitting image " << image.cols << "x" << image.rows
                  << " into " << num_x << "x" << num_y << " tiles" << std::endl;
    }

    for (int row = 0; row < num_y; row++) {
        for (int col = 0; col < num_x; col++) {
            int x = col * stride;
            int y = row * stride;
            int w = std::min(tile_size_, image.cols - x);
            int h = std::min(tile_size_, image.rows - y);

            cv::Rect roi(x, y, w, h);
            cv::Mat tile_data = image(roi).clone();

            if (w < tile_size_ || h < tile_size_) {
                cv::copyMakeBorder(
                    tile_data,
                    tile_data,
                    0,
                    tile_size_ - h,
                    0,
                    tile_size_ - w,
                    cv::BORDER_REFLECT
                );
            }

            tiles.push_back({roi, tile_data, row, col});
        }
    }

    return tiles;
}

cv::Mat TilingManager::compute_blend_mask(
    int tile_w,
    int tile_h,
    bool touches_top,
    bool touches_bottom,
    bool touches_left,
    bool touches_right
) {
    cv::Mat mask(tile_h, tile_w, CV_32FC1, cv::Scalar(1.0f));
    const int vertical_fade = std::min(overlap_, std::max(tile_h - 1, 0));
    const int horizontal_fade = std::min(overlap_, std::max(tile_w - 1, 0));

    if (!touches_top && vertical_fade > 0) {
        for (int y = 0; y < vertical_fade; y++) {
            float alpha = static_cast<float>(y + 1) / static_cast<float>(vertical_fade + 1);
            mask.row(y) *= alpha;
        }
    }

    if (!touches_bottom && vertical_fade > 0) {
        for (int y = 0; y < vertical_fade; y++) {
            float alpha = static_cast<float>(y + 1) / static_cast<float>(vertical_fade + 1);
            mask.row(tile_h - 1 - y) *= alpha;
        }
    }

    if (!touches_left && horizontal_fade > 0) {
        for (int x = 0; x < horizontal_fade; x++) {
            float alpha = static_cast<float>(x + 1) / static_cast<float>(horizontal_fade + 1);
            mask.col(x) *= alpha;
        }
    }

    if (!touches_right && horizontal_fade > 0) {
        for (int x = 0; x < horizontal_fade; x++) {
            float alpha = static_cast<float>(x + 1) / static_cast<float>(horizontal_fade + 1);
            mask.col(tile_w - 1 - x) *= alpha;
        }
    }

    return mask;
}

cv::Mat TilingManager::merge(const std::vector<Tile>& tiles, cv::Size original_size) {
    cv::Mat result = cv::Mat::zeros(original_size, CV_32FC3);
    cv::Mat weight_sum = cv::Mat::zeros(original_size, CV_32FC1);

    if (verbose_) {
        std::cout << "Merging " << tiles.size() << " tiles..." << std::endl;
    }

    for (const auto& tile : tiles) {
        const bool touches_top = tile.roi.y == 0;
        const bool touches_bottom = tile.roi.y + tile.roi.height >= original_size.height;
        const bool touches_left = tile.roi.x == 0;
        const bool touches_right = tile.roi.x + tile.roi.width >= original_size.width;

        cv::Mat blend_mask = compute_blend_mask(
            tile.roi.width,
            tile.roi.height,
            touches_top,
            touches_bottom,
            touches_left,
            touches_right
        );

        cv::Mat tile_float;
        tile.data(cv::Rect(0, 0, tile.roi.width, tile.roi.height)).convertTo(tile_float, CV_32FC3);

        cv::Mat result_roi = result(tile.roi);
        cv::Mat weight_roi = weight_sum(tile.roi);

        std::vector<cv::Mat> mask_channels(3, blend_mask);
        cv::Mat mask_3ch;
        cv::merge(mask_channels, mask_3ch);

        cv::Mat weighted_tile;
        cv::multiply(tile_float, mask_3ch, weighted_tile);

        result_roi += weighted_tile;
        weight_roi += blend_mask;
    }

    std::vector<cv::Mat> weight_channels(3, weight_sum);
    cv::Mat weight_3ch;
    cv::merge(weight_channels, weight_3ch);

    cv::Mat non_zero_mask = weight_3ch > 0;
    cv::divide(result, weight_3ch, result, 1.0, -1);
    result.setTo(0, ~non_zero_mask);

    cv::Mat result_u8;
    result.convertTo(result_u8, CV_8UC3);

    if (verbose_) {
        std::cout << "Merge completed" << std::endl;
    }
    return result_u8;
}
