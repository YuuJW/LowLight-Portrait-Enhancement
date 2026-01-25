/**
 * @file thread_pool.h
 * @brief C++11 线程池实现
 *
 * 提供固定大小的线程池，用于并行执行任务
 * 基于标准库实现，无外部依赖
 *
 * 参考: https://github.com/progschj/ThreadPool
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

/**
 * @brief 线程池类
 *
 * 功能:
 * 1. 创建固定数量的工作线程
 * 2. 提供 enqueue() 接口提交任务
 * 3. 自动分配任务给空闲线程
 * 4. 返回 std::future 用于获取任务结果
 *
 * 工作原理:
 * - 构造时创建 N 个工作线程，每个线程循环等待任务
 * - enqueue() 将任务放入队列，并通知一个工作线程
 * - 工作线程从队列取出任务并执行
 * - 析构时等待所有任务完成，然后停止所有线程
 *
 * 线程安全:
 * - 使用 mutex 保护任务队列
 * - 使用 condition_variable 实现线程间通信
 *
 * 使用示例:
 * @code
 * ThreadPool pool(4);  // 创建 4 个工作线程
 *
 * // 提交任务
 * auto result = pool.enqueue([](int x) {
 *     return x * x;
 * }, 42);
 *
 * // 获取结果
 * std::cout << result.get() << std::endl;  // 输出: 1764
 * @endcode
 */
class ThreadPool {
public:
    /**
     * @brief 构造函数 - 创建线程池
     * @param threads 工作线程数量（建议设置为 CPU 核心数）
     *
     * 说明:
     * - 立即创建指定数量的工作线程
     * - 每个线程进入等待状态，直到有任务提交
     */
    ThreadPool(size_t threads);

    /**
     * @brief 提交任务到线程池
     * @tparam F 函数类型
     * @tparam Args 参数类型
     * @param f 要执行的函数
     * @param args 函数参数
     * @return std::future 用于获取任务结果
     * @throws std::runtime_error 如果线程池已停止
     *
     * 说明:
     * - 任务会被放入队列，由空闲线程执行
     * - 返回的 future 可以用于等待任务完成并获取结果
     * - 支持任意返回类型和参数类型
     *
     * 使用示例:
     * @code
     * // 无返回值的任务
     * pool.enqueue([]() { std::cout << "Hello\n"; });
     *
     * // 有返回值的任务
     * auto result = pool.enqueue([](int a, int b) { return a + b; }, 1, 2);
     * std::cout << result.get() << std::endl;  // 输出: 3
     * @endcode
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

    /**
     * @brief 析构函数 - 停止线程池
     *
     * 说明:
     * - 等待所有已提交的任务完成
     * - 停止所有工作线程
     * - 不会接受新任务
     */
    ~ThreadPool();

private:
    // ===== 工作线程 =====
    std::vector<std::thread> workers;  ///< 工作线程向量

    // ===== 任务队列 =====
    std::queue<std::function<void()>> tasks;  ///< 待执行的任务队列

    // ===== 同步机制 =====
    std::mutex queue_mutex;              ///< 保护任务队列的互斥锁
    std::condition_variable condition;   ///< 用于线程间通信的条件变量
    bool stop;                           ///< 停止标志
};

// ===== 实现部分 =====

/**
 * @brief 构造函数实现 - 创建工作线程
 *
 * 每个工作线程的执行逻辑:
 * 1. 等待任务队列非空或停止信号
 * 2. 如果收到停止信号且队列为空，退出线程
 * 3. 从队列取出一个任务
 * 4. 执行任务
 * 5. 回到步骤 1
 */
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
{
    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        // 等待任务或停止信号
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });

                        // 如果收到停止信号且队列为空，退出线程
                        if(this->stop && this->tasks.empty())
                            return;

                        // 从队列取出任务
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    // 执行任务
                    task();
                }
            }
        );
}

/**
 * @brief 提交任务实现
 *
 * 步骤:
 * 1. 将函数和参数打包为 std::packaged_task
 * 2. 获取 future 用于返回结果
 * 3. 将任务放入队列
 * 4. 通知一个工作线程
 */
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    // 打包任务
    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 不允许在停止后提交任务
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        // 将任务放入队列
        tasks.emplace([task](){ (*task)(); });
    }

    // 通知一个工作线程
    condition.notify_one();
    return res;
}

/**
 * @brief 析构函数实现 - 停止所有线程
 *
 * 步骤:
 * 1. 设置停止标志
 * 2. 通知所有工作线程
 * 3. 等待所有线程完成当前任务并退出
 */
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

#endif
