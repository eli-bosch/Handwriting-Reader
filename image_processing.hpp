#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <vector>
#include <string>

class Image_Processing {
public:
    std::vector<float> processImage(const std::string& path, bool invert = false);

    static inline const std::string raster = ".png";
};

#endif
