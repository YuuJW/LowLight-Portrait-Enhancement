#include "ncnn_wrapper.h"
#include <iostream>

/**
 * @brief 构造函数 - 初始化 NCNN 推理引擎
 * @param param_path NCNN 模型参数文件路径 (.param)
 * @param bin_path NCNN 模型权重文件路径 (.bin)
 * @throws std::runtime_error 如果模型加载失败
 *
 * 说明:
 * - 使用 CPU 模式 (use_vulkan_compute = false)
 * - 单线程推理 (num_threads = 1)，因为外部使用线程池管理并发
 * - 如果需要 GPU 加速，可以设置 use_vulkan_compute = true
 */
NCNNWrapper::NCNNWrapper(const std::string& param_path, const std::string& bin_path) {
    // 设置 NCNN 运行选项
    net_.opt.use_vulkan_compute = false;  // 使用 CPU 模式
    net_.opt.num_threads = 1;  // 单线程推理（外部使用线程池管理并发）

    // 加载模型参数文件 (.param)
    // 返回值: 0 表示成功，非 0 表示失败
    if (net_.load_param(param_path.c_str())) {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        throw std::runtime_error("Failed to load NCNN param");
    }

    // 加载模型权重文件 (.bin)
    if (net_.load_model(bin_path.c_str())) {
        std::cerr << "Failed to load model: " << bin_path << std::endl;
        throw std::runtime_error("Failed to load NCNN model");
    }

    std::cout << "NCNN model loaded successfully" << std::endl;
}

/**
 * @brief 预处理输入图像
 * @param bgr 输入的 BGR 格式图像 (OpenCV 默认格式)
 * @return 预处理后的 ncnn::Mat，格式为 RGB，值域 [0,1]
 *
 * 处理步骤:
 * 1. BGR → RGB 颜色空间转换
 * 2. 归一化: [0,255] → [0,1]
 * 3. 转换为 ncnn::Mat 格式 (CHW layout)
 *
 * 注意: RetinexFormer 模型期望输入为 RGB 格式，值域 [0,1]
 */
ncnn::Mat NCNNWrapper::preprocess(const cv::Mat& bgr) {
    // BGR → RGB 并转换为 ncnn::Mat
    // NCNN 使用 CHW (Channel-Height-Width) 布局
    ncnn::Mat in = ncnn::Mat::from_pixels(
        bgr.data,
        ncnn::Mat::PIXEL_BGR2RGB,  // 自动进行 BGR → RGB 转换
        bgr.cols,
        bgr.rows
    );

    // 归一化到 [0, 1]
    // substract_mean_normalize(mean_vals, norm_vals)
    // 公式: output = (input - mean) * norm
    // 这里 mean = 0, norm = 1/255，即 output = input / 255
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    return in;
}

/**
 * @brief 后处理模型输出
 * @param output NCNN 模型输出，格式为 RGB，值域 [0,1]
 * @return 后处理后的 BGR 图像 (OpenCV 格式)，值域 [0,255]
 *
 * 处理步骤:
 * 1. 从 ncnn::Mat (CHW layout) 转换为 cv::Mat (HWC layout)
 * 2. RGB → BGR 颜色空间转换
 * 3. 反归一化: [0,1] → [0,255]
 * 4. 转换为 uint8 类型
 *
 * 注意:
 * - ncnn::Mat 使用 CHW 布局 (Channel-Height-Width)
 * - cv::Mat 使用 HWC 布局 (Height-Width-Channel)
 */
cv::Mat NCNNWrapper::postprocess(const ncnn::Mat& output) {
    // 从 ncnn::Mat (CHW) 转换为 cv::Mat (HWC) + RGB → BGR (使用 OpenCV 矩阵运算优化)
    std::vector<cv::Mat> channels(3);

    // 提取三个通道并进行 RGB → BGR 转换
    // ncnn::Mat 的 channel(c) 返回第 c 个通道的数据指针
    // c=0 (R) → channels[2] (B), c=1 (G) → channels[1] (G), c=2 (B) → channels[0] (R)
    for (int c = 0; c < 3; c++) {
        const float* ptr = output.channel(c);
        channels[2 - c] = cv::Mat(output.h, output.w, CV_32FC1, const_cast<float*>(ptr)).clone();
    }

    // 合并通道
    cv::Mat result;
    cv::merge(channels, result);

    // 反归一化: [0,1] → [0,255]
    result *= 255.0f;

    // 转换为 uint8 类型
    result.convertTo(result, CV_8UC3);

    return result;
}

/**
 * @brief 执行单个 tile 的推理
 * @param input_tile 输入的 tile 图像 (512x512, BGR 格式)
 * @return 增强后的 tile 图像 (512x512, BGR 格式)
 *
 * 流程:
 * 1. 预处理: BGR → RGB, [0,255] → [0,1]
 * 2. 创建 NCNN Extractor 并执行推理
 * 3. 后处理: RGB → BGR, [0,1] → [0,255]
 *
 * 线程安全性:
 * - NCNN 的 Net 对象可以在多线程间共享
 * - 每次推理创建独立的 Extractor，因此是线程安全的
 * - 在 Session Pool 中使用时，每个实例独立，无需互斥锁
 *
 * 注意:
 * - 输入输出层名称为 "input" 和 "output" (与 ONNX 导出时一致)
 */
cv::Mat NCNNWrapper::inference(const cv::Mat& input_tile) {
    // 步骤1: 预处理
    ncnn::Mat in = preprocess(input_tile);

    // 步骤2: 创建提取器
    // Extractor 是 NCNN 的推理接口，每次创建独立的实例
    // 这使得多个线程可以同时使用同一个 Net 对象进行推理
    ncnn::Extractor ex = net_.create_extractor();
    ex.set_num_threads(1);  // 单线程推理（外部使用线程池管理并发）

    // 步骤3: 输入数据
    // "input" 是模型的输入层名称 (与 ONNX 导出时定义的一致)
    ex.input("input", in);

    // 步骤4: 提取输出
    // "output" 是模型的输出层名称
    ncnn::Mat out;
    ex.extract("output", out);

    // 步骤5: 后处理
    return postprocess(out);
}
