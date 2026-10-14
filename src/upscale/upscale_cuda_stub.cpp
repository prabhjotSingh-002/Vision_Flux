#include "pixelforge/upscale.hpp"

namespace pixelforge {

bool upscaleBilinearCUDA(
    const Image& /*input*/,
    int /*scale*/,
    Image& /*output*/,
    std::string& error
) {
    error = "CUDA backend is unavailable in this build. Rebuild with -DPIXELFORGE_ENABLE_CUDA=ON on a machine with CUDA Toolkit and a supported NVIDIA GPU (e.g. Google Colab).";
    return false;
}

bool benchmarkUpscale(
    const Image& /*input*/,
    int /*scale*/,
    int /*warmup_runs*/,
    int /*measured_runs*/,
    UpscaleTiming& /*timing*/,
    std::string& error
) {
    error = "CUDA backend is unavailable in this build. Benchmarking GPU requires a build with -DPIXELFORGE_ENABLE_CUDA=ON.";
    return false;
}

} // namespace pixelforge
