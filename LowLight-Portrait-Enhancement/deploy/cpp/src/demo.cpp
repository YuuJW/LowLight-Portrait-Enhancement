/**
 * @file demo.cpp
 * @brief RetinexFormer 推理引擎演示程序
 *
 * 展示如何使用 EngineConfig 配置引擎参数，并收集性能统计信息
 * 支持命令行参数自定义配置
 */

#include "retinexformer_engine.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

/**
 * @brief 打印使用帮助
 */
void print_usage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " <model_path> <input_image> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --tile-size <size>      Tile size (default: 512)\n";
    std::cout << "  --overlap <pixels>      Overlap pixels (default: 32)\n";
    std::cout << "  --threads <num>         Number of threads (default: 4)\n";
    std::cout << "  --sessions <num>        Session pool size (default: same as threads)\n";
    std::cout << "  --quiet                 Disable verbose logging\n";
    std::cout << "  --help                  Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " model.onnx input.png\n";
    std::cout << "  " << program_name << " model.onnx input.png --tile-size 256 --threads 8 --quiet\n";
    std::cout << std::endl;
}

/**
 * @brief 解析命令行参数
 */
bool parse_args(int argc, char* argv[],
                std::string& model_path,
                std::string& input_path,
                EngineConfig& config) {

    if (argc < 3) {
        return false;
    }

    model_path = argv[1];
    input_path = argv[2];

    // 解析可选参数
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            return false;
        }
        else if (arg == "--quiet") {
            config.verbose = false;
        }
        else if (arg == "--tile-size" && i + 1 < argc) {
            config.tile_size = std::stoi(argv[++i]);
        }
        else if (arg == "--overlap" && i + 1 < argc) {
            config.overlap = std::stoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
            // 如果未显式设置 sessions，自动更新为 threads 数量
            if (config.session_pool_size == 4) {  // 默认值
                config.session_pool_size = config.num_threads;
            }
        }
        else if (arg == "--sessions" && i + 1 < argc) {
            config.session_pool_size = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string input_path;
    EngineConfig config;  // 使用默认配置

    // 解析命令行参数
    if (!parse_args(argc, argv, model_path, input_path, config)) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "\n=== RetinexFormer Demo ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Input: " << input_path << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Tile size: " << config.tile_size << "x" << config.tile_size << std::endl;
    std::cout << "  Overlap: " << config.overlap << " pixels" << std::endl;
    std::cout << "  Threads: " << config.num_threads << std::endl;
    std::cout << "  Sessions: " << config.session_pool_size << std::endl;
    std::cout << "  Verbose: " << (config.verbose ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    try {
        // 1. 加载输入图像
        cv::Mat input = cv::imread(input_path);
        if (input.empty()) {
            std::cerr << "Error: Failed to load image: " << input_path << std::endl;
            return 1;
        }

        // 2. 初始化推理引擎
        RetinexFormerEngine engine(model_path, config);

        // 3. 执行增强并收集性能统计
        PerformanceStats stats;
        cv::Mat output = engine.enhance(input, &stats);

        // 4. 保存输出图像
        std::string output_path = "output_enhanced.png";
        cv::imwrite(output_path, output);
        std::cout << "\nOutput saved to: " << output_path << std::endl;

        // 5. 打印性能统计
        stats.print();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
