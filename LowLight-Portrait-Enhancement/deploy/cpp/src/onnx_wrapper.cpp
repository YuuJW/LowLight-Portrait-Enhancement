#include "onnx_wrapper.h"
#include <iostream>

OnnxWrapper::OnnxWrapper(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RetinexFormer") {

    // Session options
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load model
    std::cout << "Loading ONNX model: " << model_path << std::endl;

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    // Get input info
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = input_tensor_info.GetShape();

    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_ptr.get();

    // Get output info
    Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = output_tensor_info.GetShape();

    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name_ptr.get();

    std::cout << "ONNX model loaded successfully" << std::endl;
    std::cout << "  Input: " << input_name_ << " [";
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        std::cout << input_shape_[i];
        if (i < input_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "  Output: " << output_name_ << " [";
    for (size_t i = 0; i < output_shape_.size(); ++i) {
        std::cout << output_shape_[i];
        if (i < output_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

std::vector<float> OnnxWrapper::preprocess(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    std::vector<float> input_tensor;
    input_tensor.reserve(3 * bgr.rows * bgr.cols);

    for (int c = 0; c < 3; ++c) {
        float* ptr = (float*)channels[c].data;
        input_tensor.insert(input_tensor.end(), ptr, ptr + bgr.rows * bgr.cols);
    }

    return input_tensor;
}

cv::Mat OnnxWrapper::postprocess(const std::vector<float>& output, int h, int w) {
    cv::Mat result(h, w, CV_32FC3);

    // CHW to HWC + RGB to BGR
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            result.at<cv::Vec3f>(y, x)[0] = output[2 * h * w + idx]; // B <- R channel (idx 2)
            result.at<cv::Vec3f>(y, x)[1] = output[1 * h * w + idx]; // G
            result.at<cv::Vec3f>(y, x)[2] = output[0 * h * w + idx]; // R <- B channel (idx 0)
        }
    }

    // [0,1] -> [0,255]
    result *= 255.0f;
    result.convertTo(result, CV_8UC3);

    return result;
}

cv::Mat OnnxWrapper::inference(const cv::Mat& input_tile) {
    // Preprocess
    std::vector<float> input_tensor = preprocess(input_tile);

    // Prepare input
    std::vector<int64_t> input_dims = {1, 3, input_tile.rows, input_tile.cols};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_dims.data(), input_dims.size()
    );

    // Run inference
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor_ort, 1,
        output_names, 1
    );

    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_dims = output_info.GetShape();

    int out_h = output_dims[2];
    int out_w = output_dims[3];
    size_t output_size = 3 * out_h * out_w;

    std::vector<float> output_vec(output_data, output_data + output_size);

    // Postprocess
    return postprocess(output_vec, out_h, out_w);
}
