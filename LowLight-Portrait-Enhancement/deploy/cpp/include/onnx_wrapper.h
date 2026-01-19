#ifndef ONNX_WRAPPER_H
#define ONNX_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @brief ONNX Runtime wrapper for RetinexFormer inference
 */
class OnnxWrapper {
public:
    /**
     * @brief Constructor - load ONNX model
     * @param model_path Path to ONNX model file
     */
    OnnxWrapper(const std::string& model_path);

    /**
     * @brief Run inference on a single tile
     * @param input_tile Input tile (512x512, BGR)
     * @return Enhanced tile (512x512, BGR)
     */
    cv::Mat inference(const cv::Mat& input_tile);

private:
    /**
     * @brief Preprocess input image
     * @param bgr BGR image [0,255]
     * @return CHW float tensor [0,1]
     */
    std::vector<float> preprocess(const cv::Mat& bgr);

    /**
     * @brief Postprocess model output
     * @param output CHW float tensor [0,1]
     * @param h Output height
     * @param w Output width
     * @return BGR image [0,255]
     */
    cv::Mat postprocess(const std::vector<float>& output, int h, int w);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

#endif // ONNX_WRAPPER_H
