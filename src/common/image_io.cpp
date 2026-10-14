#include "pixelforge/image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace pixelforge {

bool loadImage(const std::string& filepath, Image& image, std::string& errorMsg, int desired_channels) {
    int w = 0;
    int h = 0;
    int actual_channels = 0;

    stbi_uc* data = stbi_load(filepath.c_str(), &w, &h, &actual_channels, desired_channels);
    if (!data) {
        const char* reason = stbi_failure_reason();
        errorMsg = "Failed to load image '" + filepath + "': " + (reason ? reason : "Unknown error");
        return false;
    }

    int final_channels = (desired_channels > 0) ? desired_channels : actual_channels;
    if (final_channels != 1 && final_channels != 3 && final_channels != 4) {
        stbi_image_free(data);
        errorMsg = "Unsupported channel count: " + std::to_string(final_channels) + " (must be 1, 3, or 4)";
        return false;
    }

    image.width = w;
    image.height = h;
    image.channels = final_channels;
    const std::size_t total_bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(final_channels);
    image.pixels.assign(data, data + total_bytes);

    stbi_image_free(data);

    if (!image.valid()) {
        errorMsg = "Loaded image data failed structure validation";
        return false;
    }

    return true;
}

bool saveImagePNG(const std::string& filepath, const Image& image, std::string& errorMsg) {
    if (!image.valid()) {
        errorMsg = "Cannot save image: Image struct is not valid";
        return false;
    }

    const int stride_bytes = image.width * image.channels;
    const int result = stbi_write_png(filepath.c_str(), image.width, image.height, image.channels, image.pixels.data(), stride_bytes);
    if (result == 0) {
        errorMsg = "Failed to write PNG file to '" + filepath + "'";
        return false;
    }

    return true;
}

} // namespace pixelforge
