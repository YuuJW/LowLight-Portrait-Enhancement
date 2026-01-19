#include "retinexformer_engine.h"
#include <iostream>

#ifdef USE_ONNXRUNTIME
/**
 * @brief Constructor for ONNX Runtime backend
 */
RetinexFormerEngine::RetinexFormerEngine(
    const std::string& model_path,
    int num_threads
) {
    std::cout << "Initializing RetinexFormerEngine (ONNX Runtime backend)..." << std::endl;

    // 1. Create ONNX Runtime wrapper
    inference_wrapper_ = std::make_unique<OnnxWrapper>(model_path);

    // 2. Create Tiling manager (512x512 tiles, 32px overlap)
    tiling_manager_ = std::make_unique<TilingManager>(512, 32);

    // 3. Create thread pool
    thread_pool_ = std::make_unique<ThreadPool>(num_threads);

    std::cout << "RetinexFormerEngine initialized with " << num_threads << " threads" << std::endl;
}
#else
/**
 * @brief Constructor for NCNN backend
 */
RetinexFormerEngine::RetinexFormerEngine(
    const std::string& param_path,
    const std::string& bin_path,
    int num_threads
) {
    std::cout << "Initializing RetinexFormerEngine (NCNN backend)..." << std::endl;

    // 1. Create NCNN wrapper
    inference_wrapper_ = std::make_unique<NCNNWrapper>(param_path, bin_path);

    // 2. Create Tiling manager (512x512 tiles, 32px overlap)
    tiling_manager_ = std::make_unique<TilingManager>(512, 32);

    // 3. Create thread pool
    thread_pool_ = std::make_unique<ThreadPool>(num_threads);

    std::cout << "RetinexFormerEngine initialized with " << num_threads << " threads" << std::endl;
}
#endif

/**
 * @brief Enhance image
 */
cv::Mat RetinexFormerEngine::enhance(const cv::Mat& input) {
    std::cout << "\n=== Starting enhancement ===" << std::endl;
    std::cout << "Input size: " << input.cols << "x" << input.rows << std::endl;

    // Step 1: Split into tiles
    auto tiles = tiling_manager_->split(input);
    std::cout << "Split into " << tiles.size() << " tiles" << std::endl;

    // Step 2: Parallel inference
    std::vector<std::future<cv::Mat>> futures;
    for (auto& tile : tiles) {
        futures.push_back(
            thread_pool_->enqueue([this, tile_data = tile.data]() {
                std::lock_guard<std::mutex> lock(inference_mutex_);
                return inference_wrapper_->inference(tile_data);
            })
        );
    }
    std::cout << "Submitted " << futures.size() << " inference tasks" << std::endl;

    // Step 3: Collect results
    std::cout << "Waiting for inference results..." << std::endl;
    for (size_t i = 0; i < tiles.size(); i++) {
        tiles[i].data = futures[i].get();
    }
    std::cout << "All inference tasks completed" << std::endl;

    // Step 4: Merge tiles
    cv::Mat result = tiling_manager_->merge(tiles, input.size());

    std::cout << "=== Enhancement completed ===" << std::endl;
    return result;
}
