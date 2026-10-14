#pragma once

#include "pixelforge/image.hpp"
#include <string>

namespace pixelforge {

// Loads an image from a file (PNG, JPG, BMP, etc.).
// desired_channels: 0 = preserve native channels (1, 3, or 4), or specify 1/3/4 to force conversion.
bool loadImage(const std::string& filepath, Image& image, std::string& errorMsg, int desired_channels = 0);

// Saves an Image struct to a PNG file.
bool saveImagePNG(const std::string& filepath, const Image& image, std::string& errorMsg);

} // namespace pixelforge
