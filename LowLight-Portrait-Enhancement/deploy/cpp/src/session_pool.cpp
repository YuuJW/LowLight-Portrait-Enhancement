/**
 * @file session_pool.cpp
 * @brief 推理会话池实现
 *
 * 管理多个 ONNX Runtime 推理会话实例，实现真正的并行推理
 */

#include "session_pool.h"
#include <iostream>

/**
 * @brief 构造函数
 *
 * 创建多个独立的 OnnxWrapper 实例
 * 每个实例有独立的 Ort::Session，可以并行执行推理
 */
SessionPool::SessionPool(const std::string& model_path, size_t pool_size, bool verbose)
    : pool_size_(pool_size), verbose_(verbose) {

    if (verbose_) {
        std::cout << "Initializing SessionPool with " << pool_size << " sessions..." << std::endl;
    }

    // 创建 pool_size 个推理会话
    for (size_t i = 0; i < pool_size; ++i) {
        sessions_.push_back(std::make_unique<OnnxWrapper>(model_path, verbose));
        available_.push(sessions_.back().get());
    }

    if (verbose_) {
        std::cout << "SessionPool initialized successfully" << std::endl;
    }
}

/**
 * @brief 获取一个空闲会话
 *
 * 如果没有空闲会话，阻塞等待直到有会话归还或超时
 * @param timeout_ms 超时时间（毫秒），0 表示无限等待
 * @return 会话指针，超时返回 nullptr
 */
OnnxWrapper* SessionPool::acquire(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (timeout_ms > 0) {
        // 带超时等待
        bool got = condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
            return !available_.empty();
        });
        if (!got) {
            return nullptr;  // 超时
        }
    } else {
        // 无限等待（向后兼容）
        condition_.wait(lock, [this]() {
            return !available_.empty();
        });
    }

    // 从队列取出一个会话
    OnnxWrapper* session = available_.front();
    available_.pop();

    return session;
}

/**
 * @brief 归还会话到池中
 *
 * 将会话放回空闲队列，并通知一个等待的线程
 */
void SessionPool::release(OnnxWrapper* session) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push(session);
    }

    // 通知一个等待的线程
    condition_.notify_one();
}

/**
 * @brief 析构函数
 *
 * 等待所有会话归还（带超时保护），然后释放资源
 * 超时后强制销毁，避免死锁
 */
SessionPool::~SessionPool() {
    if (verbose_) {
        std::cout << "Destroying SessionPool..." << std::endl;
    }

    // 等待所有会话归还，最多等待 5 秒
    std::unique_lock<std::mutex> lock(mutex_);
    bool all_returned = condition_.wait_for(lock, std::chrono::seconds(5), [this]() {
        return available_.size() == pool_size_;
    });

    if (!all_returned) {
        std::cerr << "Warning: SessionPool destroyed with "
                  << (pool_size_ - available_.size())
                  << " sessions still in use (timeout after 5s)" << std::endl;
    }

    if (verbose_) {
        std::cout << "SessionPool destroyed" << std::endl;
    }
}
