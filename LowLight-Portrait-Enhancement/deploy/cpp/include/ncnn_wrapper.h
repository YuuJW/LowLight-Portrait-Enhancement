/**
 * @file ncnn_wrapper.h
 * @brief NCNN 推理封装
 *
 * 提供基于 NCNN 的 RetinexFormer 模型推理功能
 * 适用于 ARM CPU（移动端部署）
 */

#ifndef NCNN_WRAPPER_H
#define NCNN_WRAPPER_H

#include <string>
#include <opencv2/opencv.hpp>
#include "net.h"

/**
 * @brief NCNN 推理封装类
 *
 * 功能:
 * 1. 加载 NCNN 模型并初始化推理引擎
 * 2. 提供预处理、推理、后处理的完整流程
 * 3. 支持 512x512 tile 的推理
 *
 * 线程安全性:
 * - NCNN 的 Net 对象可以在多线程间共享
 * - 每次推理创建独立的 Extractor，因此是线程安全的
 * - 在 Session Pool 中使用时，每个实例独立，无需互斥锁
 *
 * 使用示例:
 * @code
 * NCNNWrapper wrapper("model.param", "model.bin");
 * cv::Mat input = cv::imread("input.png");
 * cv::Mat output = wrapper.inference(input);
 * @endcode
 */
class NCNNWrapper {
public:
    /**
     * @brief 构造函数 - 加载 NCNN 模型
     * @param param_path NCNN 参数文件路径 (.param)
     * @param bin_path NCNN 权重文件路径 (.bin)
     * @throws std::runtime_error 如果模型加载失败
     *
     * 说明:
     * - 使用 CPU 模式（可选 Vulkan GPU 加速）
     * - 单线程推理（外部使用线程池管理并发）
     * - 自动加载模型参数和权重
     */
    NCNNWrapper(const std::string& param_path, const std::string& bin_path);

    /**
     * @brief 默认析构函数
     */
    ~NCNNWrapper() = default;

    /**
     * @brief 执行单个 tile 的推理
     * @param input_tile 输入 tile 图像 (512x512, BGR 格式, [0,255])
     * @return 增强后的 tile 图像 (512x512, BGR 格式, [0,255])
     *
     * 流程:
     * 1. 预处理: BGR → RGB, [0,255] → [0,1]
     * 2. 创建 NCNN Extractor 并执行推理
     * 3. 后处理: RGB → BGR, [0,1] → [0,255]
     *
     * 线程安全性:
     * - 每次调用创建独立的 Extractor
     * - 多个线程可以同时调用此函数（使用不同的实例）
     * - 不需要外部加锁
     */
    cv::Mat inference(const cv::Mat& input_tile);

private:
    /**
     * @brief 预处理输入图像
     * @param bgr BGR 格式图像, 值域 [0,255]
     * @return NCNN Mat 格式, RGB, 值域 [0,1]
     *
     * 处理步骤:
     * 1. BGR → RGB 颜色空间转换
     * 2. 归一化: [0,255] → [0,1]
     * 3. 转换为 ncnn::Mat 格式 (CHW layout)
     */
    ncnn::Mat preprocess(const cv::Mat& bgr);

    /**
     * @brief 后处理模型输出
     * @param output NCNN Mat 格式, RGB, 值域 [0,1]
     * @return BGR 格式图像, 值域 [0,255]
     *
     * 处理步骤:
     * 1. 布局转换: CHW → HWC
     * 2. RGB → BGR 颜色空间转换
     * 3. 反归一化: [0,1] → [0,255]
     * 4. 转换为 uint8 类型
     */
    cv::Mat postprocess(const ncnn::Mat& output);

    // ===== NCNN 对象 =====
    ncnn::Net net_;  ///< NCNN 网络对象（可在多线程间共享）
};

#endif
