#include "pixelforge/downscale.hpp"

namespace pixelforge {

bool downscaleBilinearCUDA(
    const Image& /*input*/,
    int /*scale*/,
    Image& /*output*/,
    std::string& error
) {
    error = "CUDA backend is unavailable in this build. Rebuild with -DPIXELFORGE_ENABLE_CUDA=ON on a machine with CUDA Toolkit and a supported NVIDIA GPU (e.g. Google Colab).";
    return false;
}

} // namespace pixelforge
