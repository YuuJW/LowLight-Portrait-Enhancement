#include <iostream>
#include <chrono>
#include "retinexformer_engine.h"

/**
 * @brief 测试程序主函数
 *
 * 用法: test_engine <param_path> <bin_path> <image_path>
 *
 * 示例:
 * ./test_engine.exe \
 *     ../../models/retinexformer_opt.param \
 *     ../../models/retinexformer_opt.bin \
 *     ../../../data/LOL/lol_dataset/eval15/low/1.png
 */
int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <param> <bin> <image>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " model.param model.bin input.png" << std::endl;
        return -1;
    }

    std::string param_path = argv[1];
    std::string bin_path = argv[2];
    std::string image_path = argv[3];

    std::cout << "=== RetinexFormer C++ Engine Test ===" << std::endl;
    std::cout << "Param: " << param_path << std::endl;
    std::cout << "Model: " << bin_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << std::endl;

    // 步骤1: 加载图像
    std::cout << "Loading image..." << std::endl;
    cv::Mat input = cv::imread(image_path);
    if (input.empty()) {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image loaded: " << input.cols << "x" << input.rows << std::endl;
    std::cout << std::endl;

    // 步骤2: 创建引擎
    std::cout << "Creating RetinexFormer engine..." << std::endl;
    try {
        RetinexFormerEngine engine(param_path, bin_path, 4);
        std::cout << std::endl;

        // 步骤3: 执行推理并计时
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat output = engine.enhance(input);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "\n=== Performance ===" << std::endl;
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;

        // 步骤4: 保存结果
        std::string output_path = "output_enhanced.png";
        std::cout << "Saving result..." << std::endl;
        if (cv::imwrite(output_path, output)) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Error: Failed to save result" << std::endl;
            return -1;
        }

        std::cout << "\n=== Test completed successfully ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
