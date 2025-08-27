#include "image_processing.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdexcept>

std::vector<float> Image_Processing::processImage(const std::string& path, bool invert) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Failed to load: " + path);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);

    if (invert) cv::bitwise_not(resized, resized);

    resized.convertTo(resized, CV_32F, 1.0/255.0);

    cv::Mat flat = resized.reshape(1, 1);    // 1 x 784, type CV_32F
    if (!flat.isContinuous()) flat = flat.clone();

    std::vector<float> vec(28*28);
    std::memcpy(vec.data(), flat.ptr<float>(0), 28*28*sizeof(float));
    return vec;
}


