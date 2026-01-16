#ifndef RETINEXFORMER_ENGINE_H
#define RETINEXFORMER_ENGINE_H

#include <memory>
#include <opencv2/opencv.hpp>
#include "ncnn_wrapper.h"
#include "tiling_manager.h"
#include "thread_pool.h"

/**
 * @brief RetinexFormer 推理引擎主类
 *
 * 功能:
 * 1. 协调 NCNNWrapper, TilingManager, ThreadPool 三个模块
 * 2. 提供简洁的对外接口 enhance()
 * 3. 自动处理大图像的分块和并行推理
 */
class RetinexFormerEngine {
public:
    /**
     * @brief 构造函数
     * @param param_path NCNN 模型参数文件路径 (.param)
     * @param bin_path NCNN 模型权重文件路径 (.bin)
     * @param num_threads 线程池线程数 (默认 4)
     */
    RetinexFormerEngine(
        const std::string& param_path,
        const std::string& bin_path,
        int num_threads = 4
    );

    /**
     * @brief 增强图像 (主接口)
     * @param input 输入的低光图像
     * @return 增强后的图像
     */
    cv::Mat enhance(const cv::Mat& input);

private:
    std::unique_ptr<NCNNWrapper> ncnn_wrapper_;
    std::unique_ptr<TilingManager> tiling_manager_;
    std::unique_ptr<ThreadPool> thread_pool_;
};

#endif
