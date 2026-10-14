#pragma once

#include "pixelforge/image.hpp"
#include <string>

namespace pixelforge {

// Performs bilinear downscaling on CPU (reference implementation)
// scale: integer reduction factor (2 = 1/2 resolution, 4 = 1/4 resolution)
// Output dimensions: out_w = in_w / scale, out_h = in_h / scale
Image downscaleBilinearCPU(const Image& input, int scale);

// Performs bilinear downscaling on NVIDIA GPU using CUDA
// Returns true on success, false on error with diagnostic message
bool downscaleBilinearCUDA(
    const Image& input,
    int scale,
    Image& output,
    std::string& error
);

} // namespace pixelforge
