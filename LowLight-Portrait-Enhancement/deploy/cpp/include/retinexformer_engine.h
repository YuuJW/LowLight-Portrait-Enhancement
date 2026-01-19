#ifndef RETINEXFORMER_ENGINE_H
#define RETINEXFORMER_ENGINE_H

#include <memory>
#include <opencv2/opencv.hpp>

// Backend selection: USE_ONNXRUNTIME or USE_NCNN
#ifdef USE_ONNXRUNTIME
#include "onnx_wrapper.h"
#else
#include "ncnn_wrapper.h"
#endif

#include "tiling_manager.h"
#include "thread_pool.h"

/**
 * @brief RetinexFormer inference engine
 *
 * Features:
 * 1. Coordinates inference wrapper, TilingManager, ThreadPool
 * 2. Provides simple enhance() interface
 * 3. Handles large image tiling and parallel inference
 */
class RetinexFormerEngine {
public:
#ifdef USE_ONNXRUNTIME
    /**
     * @brief Constructor for ONNX Runtime backend
     * @param model_path ONNX model file path (.onnx)
     * @param num_threads Thread pool size (default 4)
     */
    RetinexFormerEngine(
        const std::string& model_path,
        int num_threads = 4
    );
#else
    /**
     * @brief Constructor for NCNN backend
     * @param param_path NCNN param file path (.param)
     * @param bin_path NCNN bin file path (.bin)
     * @param num_threads Thread pool size (default 4)
     */
    RetinexFormerEngine(
        const std::string& param_path,
        const std::string& bin_path,
        int num_threads = 4
    );
#endif

    /**
     * @brief Enhance image (main interface)
     * @param input Low-light input image
     * @return Enhanced image
     */
    cv::Mat enhance(const cv::Mat& input);

private:
#ifdef USE_ONNXRUNTIME
    std::unique_ptr<OnnxWrapper> inference_wrapper_;
#else
    std::unique_ptr<NCNNWrapper> inference_wrapper_;
#endif
    std::unique_ptr<TilingManager> tiling_manager_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::mutex inference_mutex_;  // Protect inference wrapper
};

#endif
