/**
 * @file retinexformer_engine.h
 * @brief RetinexFormer 推理引擎
 *
 * 协调推理后端、TilingManager 和 ThreadPool，提供简单的图像增强接口
 * 使用 ONNX Runtime 作为推理后端
 */

#ifndef RETINEXFORMER_ENGINE_H
#define RETINEXFORMER_ENGINE_H

#include <memory>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "session_pool.h"
#include "tiling_manager.h"
#include "thread_pool.h"

/**
 * @brief 性能统计结构体
 *
 * 记录推理过程各阶段的耗时，用于性能分析和优化
 */
struct PerformanceStats {
    double tiling_time_ms = 0.0;      ///< 分块耗时（毫秒）
    double inference_time_ms = 0.0;   ///< 推理耗时（毫秒）
    double merging_time_ms = 0.0;     ///< 融合耗时（毫秒）
    double total_time_ms = 0.0;       ///< 总耗时（毫秒）
    int num_tiles = 0;                ///< tile 数量

    /**
     * @brief 打印性能统计信息
     */
    void print() const {
        std::cout << "\n=== Performance Statistics ===" << std::endl;
        std::cout << "Tiles: " << num_tiles << std::endl;
        std::cout << "Tiling:    " << tiling_time_ms << " ms" << std::endl;
        std::cout << "Inference: " << inference_time_ms << " ms" << std::endl;
        std::cout << "Merging:   " << merging_time_ms << " ms" << std::endl;
        std::cout << "Total:     " << total_time_ms << " ms" << std::endl;
        std::cout << "Avg per tile: " << (num_tiles > 0 ? inference_time_ms / num_tiles : 0) << " ms" << std::endl;
        std::cout << "===============================" << std::endl;
    }
};

/**
 * @brief RetinexFormer 推理引擎
 *
 * 功能:
 * 1. 协调 SessionPool、TilingManager、ThreadPool
 * 2. 提供简单的 enhance() 接口
 * 3. 处理任意尺寸的图像（自动分割为 tiles）
 *
 * 架构设计:
 * - TilingManager: 负责大图分割和融合
 * - ThreadPool: 负责并行处理多个 tiles
 * - SessionPool: 管理多个推理会话，实现真正的并行推理
 *
 * 性能优化:
 * - 使用 tiling 技术处理大图（避免显存不足）
 * - 使用线程池并行处理多个 tiles（提升速度）
 * - 使用 Session Pool 消除互斥锁瓶颈（4核CPU 速度提升 3-4 倍）
 * - 重叠区域线性融合（消除边界伪影）
 *
 * 使用示例:
 * @code
 * RetinexFormerEngine engine("model.onnx", 4);
 * cv::Mat input = cv::imread("low_light.png");
 * cv::Mat output = engine.enhance(input);
 * cv::imwrite("enhanced.png", output);
 * @endcode
 */
class RetinexFormerEngine {
public:
    /**
     * @brief 构造函数（使用配置对象）
     * @param model_path ONNX 模型文件路径 (.onnx)
     * @param config 引擎配置（tile大小、线程数等）
     *
     * 说明:
     * - 使用 EngineConfig 统一管理所有配置参数
     * - 自动加载 ONNX 模型并创建会话池
     * - 创建 TilingManager 和 ThreadPool
     */
    RetinexFormerEngine(
        const std::string& model_path,
        const EngineConfig& config = EngineConfig()
    );

    /**
     * @brief 构造函数（兼容旧接口）
     * @param model_path ONNX 模型文件路径 (.onnx)
     * @param num_threads 线程池大小（默认 4）
     * @param session_pool_size 会话池大小（默认等于 num_threads）
     *
     * 说明:
     * - 保持向后兼容
     * - 内部转换为 EngineConfig
     */
    RetinexFormerEngine(
        const std::string& model_path,
        int num_threads,
        int session_pool_size = -1
    );

    /**
     * @brief 增强图像（主接口）
     * @param input 输入的暗光图像 (BGR 格式)
     * @param stats 可选的性能统计输出参数
     * @return 增强后的图像 (BGR 格式)
     *
     * 处理流程:
     * 1. 分割: 将大图分割为 tiles（带重叠）
     * 2. 推理: 并行处理所有 tiles（使用线程池 + Session Pool）
     * 3. 融合: 将 tiles 融合回完整图像（重叠区域线性融合）
     *
     * 性能说明:
     * - 使用 Session Pool 实现真正的并行推理
     * - 4核CPU 处理 2048x2048 图像约 2 秒
     * - CPU 利用率接近 100%
     */
    cv::Mat enhance(const cv::Mat& input, PerformanceStats* stats = nullptr);

private:
    /**
     * @brief 条件日志输出辅助方法
     * @param msg 日志消息
     *
     * 根据 config_.verbose 决定是否输出日志
     */
    void log(const std::string& msg);

    // ===== 配置 =====
    EngineConfig config_;  ///< 引擎配置

    // ===== 推理后端 =====
    std::unique_ptr<SessionPool> session_pool_;  ///< 推理会话池（管理多个推理实例）

    // ===== 辅助模块 =====
    std::unique_ptr<TilingManager> tiling_manager_;   ///< Tile 分割和融合管理器
    std::unique_ptr<ThreadPool> thread_pool_;         ///< 线程池（并行处理 tiles）
};

#endif
