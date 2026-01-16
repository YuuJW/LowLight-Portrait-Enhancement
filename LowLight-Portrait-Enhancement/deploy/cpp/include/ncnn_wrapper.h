#ifndef NCNN_WRAPPER_H
#define NCNN_WRAPPER_H

#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "net.h"

class NCNNWrapper {
public:
    NCNNWrapper(const std::string& param_path, const std::string& bin_path);
    ~NCNNWrapper() = default;

    cv::Mat inference(const cv::Mat& input_tile);

private:
    ncnn::Net net_;
    std::mutex mutex_;

    // BGR → RGB, [0,255] → [0,1]
    ncnn::Mat preprocess(const cv::Mat& bgr);

    // [0,1] → [0,255], RGB → BGR
    cv::Mat postprocess(const ncnn::Mat& output);
};

#endif
