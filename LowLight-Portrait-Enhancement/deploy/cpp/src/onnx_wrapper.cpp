/**
 * @file onnx_wrapper.cpp
 * @brief ONNX Runtime 推理封装实现
 *
 * 提供基于 ONNX Runtime 的 RetinexFormer 模型推理功能
 * 包括模型加载、预处理、推理、后处理等完整流程
 */

#include "onnx_wrapper.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

/**
 * @brief 构造函数 - 初始化 ONNX Runtime 推理引擎
 * @param model_path ONNX 模型文件路径 (.onnx)
 * @param verbose 是否输出详细日志 (默认 true)
 * @throws std::runtime_error 如果模型加载失败
 *
 * 说明:
 * - 设置单线程推理 (SetIntraOpNumThreads(1))，因为外部使用线程池管理并发
 * - 启用所有图优化 (ORT_ENABLE_ALL) 以提升推理性能
 * - 自动获取模型的输入输出信息（名称、形状）
 */
OnnxWrapper::OnnxWrapper(const std::string& model_path, bool verbose)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RetinexFormer"), verbose_(verbose) {
    std::ifstream model_stream(model_path, std::ios::binary);
    if (!model_stream.good()) {
        throw std::runtime_error("ONNX model not found: " + model_path);
    }

    // 配置 Session 选项
    // SetIntraOpNumThreads(1): 单线程推理，避免与外部线程池冲突
    session_options_.SetIntraOpNumThreads(1);
    // 启用所有图优化（常量折叠、算子融合等）
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 加载 ONNX 模型
    if (verbose_) {
        std::cout << "Loading ONNX model: " << model_path << std::endl;
    }

    // Windows 平台需要使用宽字符路径
#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    // 获取模型输入信息
    // 包括输入层名称和张量形状 (例如: [1, 3, 512, 512])
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = input_tensor_info.GetShape();

    Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_ptr.get();

    // 获取模型输出信息
    // 包括输出层名称和张量形状 (例如: [1, 3, 512, 512])
    Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = output_tensor_info.GetShape();

    Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name_ptr.get();

    if (verbose_) {
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
}

/**
 * @brief 预处理输入图像
 * @param bgr 输入的 BGR 格式图像 (OpenCV 默认格式)
 * @return 预处理后的浮点数张量，格式为 CHW (Channel-Height-Width)，值域 [0,1]
 *
 * 处理步骤:
 * 1. BGR → RGB 颜色空间转换 (RetinexFormer 模型期望 RGB 输入)
 * 2. 归一化: [0,255] → [0,1]
 * 3. 转换布局: HWC (Height-Width-Channel) → CHW (Channel-Height-Width)
 *
 * 注意: ONNX 模型通常使用 CHW 布局，而 OpenCV 使用 HWC 布局
 */
std::vector<float> OnnxWrapper::preprocess(const cv::Mat& bgr) {
    // 步骤1: BGR → RGB 颜色空间转换
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // 步骤2: 归一化到 [0, 1]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 步骤3: HWC → CHW 布局转换
    // OpenCV 的 split 将 HWC 格式的三通道图像分离为三个单通道图像
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    // 将三个通道按 CHW 顺序拼接成一维数组
    // 顺序: [R_channel, G_channel, B_channel]
    std::vector<float> input_tensor;
    input_tensor.reserve(3 * bgr.rows * bgr.cols);

    for (int c = 0; c < 3; ++c) {
        float* ptr = (float*)channels[c].data;
        input_tensor.insert(input_tensor.end(), ptr, ptr + bgr.rows * bgr.cols);
    }

    return input_tensor;
}

/**
 * @brief 后处理模型输出
 * @param output ONNX 模型输出，格式为 CHW (Channel-Height-Width)，值域 [0,1]
 * @param h 输出图像高度
 * @param w 输出图像宽度
 * @return 后处理后的 BGR 图像 (OpenCV 格式)，值域 [0,255]
 *
 * 处理步骤:
 * 1. 转换布局: CHW → HWC
 * 2. RGB → BGR 颜色空间转换
 * 3. 反归一化: [0,1] → [0,255]
 * 4. 转换为 uint8 类型
 *
 * 优化说明:
 * - 使用 cv::Mat 包装 + cv::merge 替代逐像素循环
 * - 性能提升约 2-5 倍
 */
cv::Mat OnnxWrapper::postprocess(const std::vector<float>& output, int h, int w) {
    // CHW → HWC + RGB → BGR (使用 OpenCV 矩阵运算优化)
    // 将 CHW 格式的输出转换为三个独立的通道
    std::vector<cv::Mat> channels(3);

    // 注意：模型输出是 RGB 顺序，需要转换为 OpenCV 的 BGR 顺序
    // output[0*h*w : 1*h*w] = R channel → BGR 的 channels[2] (B 位置)
    // output[1*h*w : 2*h*w] = G channel → BGR 的 channels[1] (G 位置)
    // output[2*h*w : 3*h*w] = B channel → BGR 的 channels[0] (R 位置)
    channels[2] = cv::Mat(h, w, CV_32FC1, const_cast<float*>(output.data() + 0 * h * w)).clone(); // R → B
    channels[1] = cv::Mat(h, w, CV_32FC1, const_cast<float*>(output.data() + 1 * h * w)).clone(); // G → G
    channels[0] = cv::Mat(h, w, CV_32FC1, const_cast<float*>(output.data() + 2 * h * w)).clone(); // B → R

    // 合并三个通道为 BGR 图像
    cv::Mat result;
    cv::merge(channels, result);

    // 反归一化: [0,1] → [0,255]
    result *= 255.0f;
    result.convertTo(result, CV_8UC3);

    return result;
}

/**
 * @brief 执行单个 tile 的推理
 * @param input_tile 输入的 tile 图像 (512x512, BGR 格式)
 * @return 增强后的 tile 图像 (512x512, BGR 格式)
 *
 * 流程:
 * 1. 预处理: BGR → RGB, [0,255] → [0,1], HWC → CHW
 * 2. 创建 ONNX Runtime 张量
 * 3. 执行推理
 * 4. 后处理: CHW → HWC, RGB → BGR, [0,1] → [0,255]
 *
 * 注意:
 * - 此函数不是线程安全的，需要外部加锁或使用 Session Pool
 * - 输入输出都是 BGR 格式，与 OpenCV 保持一致
 */
cv::Mat OnnxWrapper::inference(const cv::Mat& input_tile) {
    if (input_tile.empty()) {
        throw std::invalid_argument("input tile is empty");
    }
    if (input_tile.channels() != 3) {
        throw std::invalid_argument("input tile must have 3 channels");
    }

    // 步骤1: 预处理
    std::vector<float> input_tensor = preprocess(input_tile);

    // 步骤2: 准备输入张量
    // 创建形状为 [1, 3, H, W] 的 ONNX 张量
    std::vector<int64_t> input_dims = {1, 3, input_tile.rows, input_tile.cols};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_dims.data(), input_dims.size()
    );

    // 步骤3: 执行推理
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor_ort, 1,
        output_names, 1
    );

    // 步骤4: 提取输出数据
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_dims = output_info.GetShape();

    int out_h = output_dims[2];
    int out_w = output_dims[3];
    size_t output_size = 3 * out_h * out_w;

    std::vector<float> output_vec(output_data, output_data + output_size);

    // 步骤5: 后处理
    return postprocess(output_vec, out_h, out_w);
}
