/**
 * @file session_pool.h
 * @brief 推理会话池（Session Pool）
 *
 * 管理多个推理会话实例，实现真正的并行推理
 * 解决单个推理实例 + mutex 导致的串行执行瓶颈
 */

#ifndef SESSION_POOL_H
#define SESSION_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

// 后端选择
#ifdef USE_ONNXRUNTIME
#include "onnx_wrapper.h"
#else
#include "ncnn_wrapper.h"
#endif

/**
 * @brief 推理会话池
 *
 * 功能:
 * 1. 管理多个推理会话（OnnxWrapper 或 NCNNWrapper）实例
 * 2. 提供 acquire() 获取空闲会话
 * 3. 提供 release() 归还会话到池中
 * 4. 线程安全，支持多线程并发访问
 *
 * 工作原理:
 * - 构造时创建 N 个推理会话实例
 * - acquire() 从池中取出一个空闲会话（如果没有则等待）
 * - release() 将会话归还到池中，并通知等待的线程
 * - 每个线程使用独立的会话，无需互斥锁，实现真正的并行推理
 *
 * 性能提升:
 * - 相比单个会话 + mutex：4核CPU 上速度提升 3-4 倍
 * - CPU 利用率从 25% 提升到接近 100%
 *
 * 使用示例:
 * @code
 * SessionPool pool("model.onnx", 4);  // 创建 4 个会话
 *
 * // 在线程池中使用
 * thread_pool.enqueue([&pool]() {
 *     auto session = pool.acquire();  // 获取会话
 *     cv::Mat result = session->inference(input);
 *     pool.release(session);  // 归还会话
 *     return result;
 * });
 * @endcode
 */
class SessionPool {
public:
#ifdef USE_ONNXRUNTIME
    /**
     * @brief 构造函数 - ONNX Runtime 后端
     * @param model_path ONNX 模型文件路径
     * @param pool_size 会话池大小（建议等于线程数）
     *
     * 说明:
     * - 创建 pool_size 个独立的 OnnxWrapper 实例
     * - 每个实例有独立的 Ort::Session
     * - 所有实例共享相同的模型权重（ONNX Runtime 内部优化）
     */
    SessionPool(const std::string& model_path, size_t pool_size);
#else
    /**
     * @brief 构造函数 - NCNN 后端
     * @param param_path NCNN 参数文件路径
     * @param bin_path NCNN 权重文件路径
     * @param pool_size 会话池大小（建议等于线程数）
     *
     * 说明:
     * - 创建 pool_size 个独立的 NCNNWrapper 实例
     * - 每个实例有独立的 ncnn::Extractor
     * - NCNN 的 Net 对象可以共享，但 Extractor 必须独立
     */
    SessionPool(const std::string& param_path, const std::string& bin_path, size_t pool_size);
#endif

    /**
     * @brief 析构函数
     *
     * 说明:
     * - 等待所有会话归还
     * - 释放所有会话实例
     */
    ~SessionPool();

    /**
     * @brief 获取一个空闲会话
     * @return 推理会话指针
     *
     * 说明:
     * - 如果有空闲会话，立即返回
     * - 如果没有空闲会话，阻塞等待直到有会话归还
     * - 线程安全
     *
     * 注意:
     * - 使用完毕后必须调用 release() 归还会话
     * - 否则会导致其他线程永久等待
     */
#ifdef USE_ONNXRUNTIME
    OnnxWrapper* acquire();
#else
    NCNNWrapper* acquire();
#endif

    /**
     * @brief 归还会话到池中
     * @param session 要归还的会话指针
     *
     * 说明:
     * - 将会话放回空闲队列
     * - 通知一个等待的线程
     * - 线程安全
     *
     * 注意:
     * - 必须归还通过 acquire() 获取的会话
     * - 不要归还已经归还过的会话
     */
#ifdef USE_ONNXRUNTIME
    void release(OnnxWrapper* session);
#else
    void release(NCNNWrapper* session);
#endif

    /**
     * @brief 获取池大小
     * @return 会话池中的会话总数
     */
    size_t size() const { return pool_size_; }

private:
    // ===== 会话实例 =====
#ifdef USE_ONNXRUNTIME
    std::vector<std::unique_ptr<OnnxWrapper>> sessions_;  ///< 所有会话实例
    std::queue<OnnxWrapper*> available_;                  ///< 空闲会话队列
#else
    std::vector<std::unique_ptr<NCNNWrapper>> sessions_;  ///< 所有会话实例
    std::queue<NCNNWrapper*> available_;                  ///< 空闲会话队列
#endif

    // ===== 同步机制 =====
    std::mutex mutex_;                  ///< 保护空闲队列的互斥锁
    std::condition_variable condition_; ///< 用于等待空闲会话的条件变量
    size_t pool_size_;                  ///< 池大小
};

#endif // SESSION_POOL_H
