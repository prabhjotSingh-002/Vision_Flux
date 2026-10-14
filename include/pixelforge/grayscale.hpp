#pragma once

#include "pixelforge/image.hpp"
#include <string>

namespace pixelforge {

// Future Module: Grayscale Conversion (Collaborator Module)
// Converts an RGB/RGBA Image into single-channel Grayscale

// Reference CPU implementation
Image rgbToGrayscaleCPU(const Image& input);

// Custom CUDA Kernel implementation
bool rgbToGrayscaleCUDA(
    const Image& input,
    Image& output,
    std::string& error
);

} // namespace pixelforge
