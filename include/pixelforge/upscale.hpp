#pragma once

#include "pixelforge/image.hpp"
#include <string>

namespace pixelforge {

// Performs bilinear upscaling on CPU (reference implementation)
// Supported scale factors: 2, 4
// Returns valid Image on success, or throws std::invalid_argument / returns empty Image on error
Image upscaleBilinearCPU(const Image& input, int scale);

// Performs bilinear upscaling on NVIDIA GPU using CUDA
// Supported scale factors: 2, 4
// Returns true on success, false on error with diagnostic message in error
bool upscaleBilinearCUDA(
    const Image& input,
    int scale,
    Image& output,
    std::string& error
);

// Timing measurements structure for benchmarking
struct UpscaleTiming {
    double cpu_compute_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double gpu_end_to_end_ms = 0.0;
    double speedup = 0.0;
};

// Benchmark runner comparing CPU reference and GPU kernel
bool benchmarkUpscale(
    const Image& input,
    int scale,
    int warmup_runs,
    int measured_runs,
    UpscaleTiming& timing,
    std::string& error
);

} // namespace pixelforge
