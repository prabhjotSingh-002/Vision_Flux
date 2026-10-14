#include "pixelforge/downscale.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace pixelforge {

// CUDA Kernel: Each thread computes one destination downscaled pixel (xd, yd)
__global__ void bilinearDownscaleKernel(
    const std::uint8_t* __restrict__ d_in,
    std::uint8_t* __restrict__ d_out,
    int in_w,
    int in_h,
    int out_w,
    int out_h,
    int channels,
    int scale
) {
    const int xd = blockIdx.x * blockDim.x + threadIdx.x;
    const int yd = blockIdx.y * blockDim.y + threadIdx.y;

    if (xd >= out_w || yd >= out_h) {
        return;
    }

    const float scale_f = static_cast<float>(scale);
    const float max_x = static_cast<float>(in_w - 1);
    const float max_y = static_cast<float>(in_h - 1);

    // Map destination downscaled coordinate back to source coordinates
    float xs = (static_cast<float>(xd) + 0.5f) * scale_f - 0.5f;
    if (xs < 0.0f) xs = 0.0f;
    if (xs > max_x) xs = max_x;

    float ys = (static_cast<float>(yd) + 0.5f) * scale_f - 0.5f;
    if (ys < 0.0f) ys = 0.0f;
    if (ys > max_y) ys = max_y;

    const int x0 = static_cast<int>(floorf(xs));
    const int x1 = (x0 + 1 < in_w) ? (x0 + 1) : (in_w - 1);
    const float alpha = xs - static_cast<float>(x0);

    const int y0 = static_cast<int>(floorf(ys));
    const int y1 = (y0 + 1 < in_h) ? (y0 + 1) : (in_h - 1);
    const float beta = ys - static_cast<float>(y0);

    const float w00 = (1.0f - alpha) * (1.0f - beta);
    const float w10 = alpha * (1.0f - beta);
    const float w01 = (1.0f - alpha) * beta;
    const float w11 = alpha * beta;

    const std::size_t idx00 = (static_cast<std::size_t>(y0) * in_w + x0) * channels;
    const std::size_t idx10 = (static_cast<std::size_t>(y0) * in_w + x1) * channels;
    const std::size_t idx01 = (static_cast<std::size_t>(y1) * in_w + x0) * channels;
    const std::size_t idx11 = (static_cast<std::size_t>(y1) * in_w + x1) * channels;

    const std::size_t out_idx = (static_cast<std::size_t>(yd) * out_w + xd) * channels;

    for (int c = 0; c < channels; ++c) {
        const float p00 = static_cast<float>(d_in[idx00 + c]);
        const float p10 = static_cast<float>(d_in[idx10 + c]);
        const float p01 = static_cast<float>(d_in[idx01 + c]);
        const float p11 = static_cast<float>(d_in[idx11 + c]);

        const float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
        const float rounded = roundf(val);
        float clamped = rounded;
        if (clamped < 0.0f) clamped = 0.0f;
        if (clamped > 255.0f) clamped = 255.0f;

        d_out[out_idx + c] = static_cast<std::uint8_t>(clamped);
    }
}

static inline bool checkCuda(cudaError_t result, const char* msg, std::string& error) {
    if (result != cudaSuccess) {
        error = std::string(msg) + ": " + cudaGetErrorString(result);
        return false;
    }
    return true;
}

bool downscaleBilinearCUDA(
    const Image& input,
    int scale,
    Image& output,
    std::string& error
) {
    if (!input.valid()) {
        error = "Input image is invalid or empty.";
        return false;
    }
    if (scale != 2 && scale != 4) {
        error = "Unsupported downscale factor: " + std::to_string(scale) + ". Only 2x and 4x are supported.";
        return false;
    }

    const int in_w = input.width;
    const int in_h = input.height;
    const int channels = input.channels;
    const int out_w = in_w / scale;
    const int out_h = in_h / scale;

    if (out_w < 1 || out_h < 1) {
        error = "Input image dimensions (" + std::to_string(in_w) + "x" + std::to_string(in_h) +
                ") are too small for " + std::to_string(scale) + "x downscaling.";
        return false;
    }

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        error = "No CUDA-capable GPU detected on system.";
        return false;
    }

    const std::size_t in_bytes = static_cast<std::size_t>(in_w) * in_h * channels * sizeof(std::uint8_t);
    const std::size_t out_bytes = static_cast<std::size_t>(out_w) * out_h * channels * sizeof(std::uint8_t);

    output.width = out_w;
    output.height = out_h;
    output.channels = channels;
    output.pixels.resize(out_bytes);

    std::uint8_t* d_in = nullptr;
    std::uint8_t* d_out = nullptr;

    if (!checkCuda(cudaMalloc(&d_in, in_bytes), "cudaMalloc d_in failed", error)) {
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc d_out failed", error)) {
        cudaFree(d_in);
        return false;
    }

    if (!checkCuda(cudaMemcpy(d_in, input.pixels.data(), in_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D failed", error)) {
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y
    );

    bilinearDownscaleKernel<<<grid, block>>>(
        d_in, d_out, in_w, in_h, out_w, out_h, channels, scale
    );

    cudaError_t launchErr = cudaGetLastError();
    if (!checkCuda(launchErr, "Downscale kernel launch failed", error)) {
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (!checkCuda(syncErr, "cudaDeviceSynchronize failed", error)) {
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    if (!checkCuda(cudaMemcpy(output.pixels.data(), d_out, out_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H failed", error)) {
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return true;
}

} // namespace pixelforge
