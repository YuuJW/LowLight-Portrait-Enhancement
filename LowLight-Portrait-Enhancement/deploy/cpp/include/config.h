#ifndef CONFIG_H
#define CONFIG_H

/**
 * @file config.h
 * @brief RetinexFormer 推理引擎配置结构体
 *
 * 统一管理引擎的各项配置参数，避免硬编码常量散落在代码各处
 */

/**
 * @brief 推理引擎配置结构体
 *
 * 包含 tile 分割、线程池、会话池等所有可配置参数
 */
struct EngineConfig {
    // ===== Tiling 配置 =====

    /**
     * @brief Tile 尺寸（正方形边长）
     *
     * 大图会被分割成 tile_size x tile_size 的小块进行推理
     * 默认 512，与模型训练尺寸一致
     */
    int tile_size = 512;

    /**
     * @brief Tile 重叠像素数
     *
     * 相邻 tile 之间的重叠区域，用于消除边界伪影
     * 重叠区域会进行线性融合
     * 默认 32 像素
     */
    int overlap = 32;

    // ===== 并行配置 =====

    /**
     * @brief 线程池大小
     *
     * 用于并行处理多个 tile 的线程数量
     * 建议设置为 CPU 核心数
     * 默认 4
     */
    int num_threads = 4;

    /**
     * @brief 推理会话池大小
     *
     * 同时存在的推理会话（Session）数量
     * 每个线程需要独立的会话才能真正并行
     * 建议设置为 num_threads，避免线程等待
     * 默认 4
     */
    int session_pool_size = 4;

    // ===== 日志配置 =====

    /**
     * @brief 是否输出详细日志
     *
     * true: 输出 tile 分割、推理进度等详细信息
     * false: 仅输出关键信息
     * 默认 true
     */
    bool verbose = true;

    // ===== 构造函数 =====

    /**
     * @brief 默认构造函数
     *
     * 使用默认配置：512x512 tile, 32px overlap, 4 threads
     */
    EngineConfig() = default;

    /**
     * @brief 自定义构造函数
     *
     * @param tile_sz Tile 尺寸
     * @param ovlp 重叠像素数
     * @param threads 线程数
     * @param sessions 会话池大小（默认等于线程数）
     * @param verb 是否输出详细日志
     */
    EngineConfig(int tile_sz, int ovlp, int threads, int sessions = -1, bool verb = true)
        : tile_size(tile_sz)
        , overlap(ovlp)
        , num_threads(threads)
        , session_pool_size(sessions > 0 ? sessions : threads)  // 默认等于线程数
        , verbose(verb)
    {}
};

#endif // CONFIG_H
