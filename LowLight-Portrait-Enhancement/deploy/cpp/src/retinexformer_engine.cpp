#include "retinexformer_engine.h"
#include <iostream>

/**
 * @brief 构造函数 - 初始化所有模块
 * @param param_path NCNN 模型参数文件路径 (.param)
 * @param bin_path NCNN 模型权重文件路径 (.bin)
 * @param num_threads 线程池线程数 (默认 4)
 * @throws std::runtime_error 如果模型加载失败
 *
 * 初始化顺序:
 * 1. NCNNWrapper - 加载 NCNN 模型
 * 2. TilingManager - 配置分块参数 (512x512, 32px overlap)
 * 3. ThreadPool - 创建指定数量的工作线程
 */
RetinexFormerEngine::RetinexFormerEngine(
    const std::string& param_path,
    const std::string& bin_path,
    int num_threads
) {
    std::cout << "Initializing RetinexFormerEngine..." << std::endl;

    // 1. 创建 NCNN 推理包装器
    // 如果模型加载失败，会抛出 std::runtime_error 异常
    ncnn_wrapper_ = std::make_unique<NCNNWrapper>(param_path, bin_path);

    // 2. 创建 Tiling 管理器
    // tile_size = 512 (与 ONNX 导出时的输入尺寸一致)
    // overlap = 32 (重叠区域，用于平滑边界)
    tiling_manager_ = std::make_unique<TilingManager>(512, 32);

    // 3. 创建线程池
    // 线程数建议设置为 CPU 核心数的 50%-100%
    thread_pool_ = std::make_unique<ThreadPool>(num_threads);

    std::cout << "RetinexFormerEngine initialized with " << num_threads << " threads" << std::endl;
}

/**
 * @brief 增强图像的主函数
 * @param input 输入的低光图像 (BGR 格式，任意尺寸)
 * @return 增强后的图像 (BGR 格式，与输入尺寸相同)
 *
 * 处理流程:
 * 1. 图像分块 - 将大图分割为 512x512 的 tiles (带 32px 重叠)
 * 2. 并行推理 - 使用线程池并行处理所有 tiles
 * 3. 结果收集 - 等待所有推理任务完成
 * 4. 图像融合 - 使用加权混合融合 tiles 为完整图像
 *
 * 性能特点:
 * - 支持任意尺寸输入 (自动分块)
 * - 多线程并行加速
 * - 边界平滑融合 (无接缝)
 */
cv::Mat RetinexFormerEngine::enhance(const cv::Mat& input) {
    std::cout << "\n=== Starting enhancement ===" << std::endl;
    std::cout << "Input size: " << input.cols << "x" << input.rows << std::endl;

    // 步骤1: 图像分块
    // 将大图分割为 512x512 的 tiles，相邻 tiles 重叠 32px
    auto tiles = tiling_manager_->split(input);
    std::cout << "Split into " << tiles.size() << " tiles" << std::endl;

    // 步骤2: 并行推理所有 tiles
    // 使用线程池并行处理，提高处理速度
    std::vector<std::future<cv::Mat>> futures;
    for (auto& tile : tiles) {
        // 提交推理任务到线程池
        // 使用 lambda 捕获 tile.data 的副本，避免引用失效
        futures.push_back(
            thread_pool_->enqueue([this, tile_data = tile.data]() {
                return ncnn_wrapper_->inference(tile_data);
            })
        );
    }
    std::cout << "Submitted " << futures.size() << " inference tasks" << std::endl;

    // 步骤3: 收集所有推理结果
    // 阻塞等待所有线程完成推理
    std::cout << "Waiting for inference results..." << std::endl;
    for (size_t i = 0; i < tiles.size(); i++) {
        tiles[i].data = futures[i].get();  // 阻塞等待第 i 个任务完成
    }
    std::cout << "All inference tasks completed" << std::endl;

    // 步骤4: 融合 tiles 为完整图像
    // 使用加权混合算法，在重叠区域进行平滑过渡
    cv::Mat result = tiling_manager_->merge(tiles, input.size());

    std::cout << "=== Enhancement completed ===" << std::endl;
    return result;
}
