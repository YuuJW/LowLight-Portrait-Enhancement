/**
 * @file onnx_wrapper.h
 * @brief ONNX Runtime 推理封装
 *
 * 提供基于 ONNX Runtime 的 RetinexFormer 模型推理功能
 * 支持单 tile 推理，需要外部管理并发（使用线程池 + Session Pool）
 */

#ifndef ONNX_WRAPPER_H
#define ONNX_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @brief ONNX Runtime 推理封装类
 *
 * 功能:
 * 1. 加载 ONNX 模型并初始化推理会话
 * 2. 提供预处理、推理、后处理的完整流程
 * 3. 支持 512x512 tile 的推理
 *
 * 线程安全性:
 * - 此类不是线程安全的，多线程环境下需要：
 *   a. 使用互斥锁保护 (性能差，串行执行)
 *   b. 使用 Session Pool，每个线程独立的实例 (推荐)
 *
 * 使用示例:
 * @code
 * OnnxWrapper wrapper("model.onnx");
 * cv::Mat input = cv::imread("input.png");
 * cv::Mat output = wrapper.inference(input);
 * @endcode
 */
class OnnxWrapper {
public:
    /**
     * @brief 构造函数 - 加载 ONNX 模型
     * @param model_path ONNX 模型文件路径 (.onnx)
     * @param verbose 是否输出详细日志（默认 true）
     * @throws std::runtime_error 如果模型加载失败
     *
     * 说明:
     * - 自动配置推理选项（单线程、图优化）
     * - 自动获取模型的输入输出信息
     * - Windows 平台自动处理宽字符路径
     */
    OnnxWrapper(const std::string& model_path, bool verbose = true);

    /**
     * @brief 执行单个 tile 的推理
     * @param input_tile 输入 tile 图像 (512x512, BGR 格式, [0,255])
     * @return 增强后的 tile 图像 (512x512, BGR 格式, [0,255])
     *
     * 流程:
     * 1. 预处理: BGR → RGB, [0,255] → [0,1], HWC → CHW
     * 2. 创建 ONNX 张量并执行推理
     * 3. 后处理: CHW → HWC, RGB → BGR, [0,1] → [0,255]
     *
     * 注意:
     * - 此函数不是线程安全的
     * - 输入输出都是 BGR 格式，与 OpenCV 保持一致
     */
    cv::Mat inference(const cv::Mat& input_tile);

private:
    /**
     * @brief 预处理输入图像
     * @param bgr BGR 格式图像, 值域 [0,255]
     * @return CHW 格式浮点数张量, 值域 [0,1]
     *
     * 处理步骤:
     * 1. BGR → RGB 颜色空间转换
     * 2. 归一化: [0,255] → [0,1]
     * 3. 布局转换: HWC → CHW
     */
    std::vector<float> preprocess(const cv::Mat& bgr);

    /**
     * @brief 后处理模型输出
     * @param output CHW 格式浮点数张量, 值域 [0,1]
     * @param h 输出图像高度
     * @param w 输出图像宽度
     * @return BGR 格式图像, 值域 [0,255]
     *
     * 处理步骤:
     * 1. 布局转换: CHW → HWC
     * 2. RGB → BGR 颜色空间转换
     * 3. 反归一化: [0,1] → [0,255]
     * 4. 转换为 uint8 类型
     */
    cv::Mat postprocess(const std::vector<float>& output, int h, int w);

    // ===== ONNX Runtime 对象 =====
    Ort::Env env_;                              ///< ONNX Runtime 环境
    Ort::SessionOptions session_options_;       ///< 推理会话选项
    std::unique_ptr<Ort::Session> session_;     ///< 推理会话（核心对象）
    Ort::AllocatorWithDefaultOptions allocator_; ///< 内存分配器

    // ===== 模型信息 =====
    std::string input_name_;                    ///< 输入层名称 (例如: "input")
    std::string output_name_;                   ///< 输出层名称 (例如: "output")
    std::vector<int64_t> input_shape_;          ///< 输入张量形状 (例如: [1, 3, 512, 512])
    std::vector<int64_t> output_shape_;         ///< 输出张量形状 (例如: [1, 3, 512, 512])
    bool verbose_;                              ///< 是否输出详细日志
};

#endif // ONNX_WRAPPER_H
