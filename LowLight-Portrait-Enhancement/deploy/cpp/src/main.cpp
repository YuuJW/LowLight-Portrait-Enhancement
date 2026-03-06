#include <iostream>
#include <chrono>
#include "retinexformer_engine.h"

/**
 * @brief Test program for RetinexFormer C++ Engine
 *
 * Usage: test_engine <onnx_path> <image_path>
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <image>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " model.onnx input.png" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    std::cout << "=== RetinexFormer C++ Engine Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << std::endl;

    // Step 1: Load image
    std::cout << "Loading image..." << std::endl;
    cv::Mat input = cv::imread(image_path);
    if (input.empty()) {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image loaded: " << input.cols << "x" << input.rows << std::endl;
    std::cout << std::endl;

    // Step 2: Create engine
    std::cout << "Creating RetinexFormer engine..." << std::endl;
    try {
        RetinexFormerEngine engine(model_path, 4);
        std::cout << std::endl;

        // Step 3: Run inference
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat output = engine.enhance(input);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "\n=== Performance ===" << std::endl;
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;

        // Step 4: Save result
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
