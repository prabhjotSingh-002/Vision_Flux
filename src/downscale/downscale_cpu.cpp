#include "pixelforge/downscale.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pixelforge {

Image downscaleBilinearCPU(const Image& input, int scale) {
    if (!input.valid()) {
        throw std::invalid_argument("Input image is invalid or empty.");
    }
    if (scale != 2 && scale != 4) {
        throw std::invalid_argument("Unsupported downscale factor: " + std::to_string(scale) +
                                    ". Only 2x (1/2 size) and 4x (1/4 size) are supported.");
    }

    const int in_w = input.width;
    const int in_h = input.height;
    const int channels = input.channels;

    const int out_w = in_w / scale;
    const int out_h = in_h / scale;

    if (out_w < 1 || out_h < 1) {
        throw std::invalid_argument("Input image dimensions (" + std::to_string(in_w) + "x" +
                                    std::to_string(in_h) + ") are too small for " +
                                    std::to_string(scale) + "x downscaling.");
    }

    Image output;
    output.width = out_w;
    output.height = out_h;
    output.channels = channels;
    output.pixels.resize(static_cast<std::size_t>(out_w) * out_h * channels);

    const float scale_f = static_cast<float>(scale);
    const float max_x = static_cast<float>(in_w - 1);
    const float max_y = static_cast<float>(in_h - 1);

    for (int yd = 0; yd < out_h; ++yd) {
        float ys = (static_cast<float>(yd) + 0.5f) * scale_f - 0.5f;
        ys = std::clamp(ys, 0.0f, max_y);

        const int y0 = static_cast<int>(std::floor(ys));
        const int y1 = std::min(y0 + 1, in_h - 1);
        const float beta = ys - static_cast<float>(y0);

        for (int xd = 0; xd < out_w; ++xd) {
            float xs = (static_cast<float>(xd) + 0.5f) * scale_f - 0.5f;
            xs = std::clamp(xs, 0.0f, max_x);

            const int x0 = static_cast<int>(std::floor(xs));
            const int x1 = std::min(x0 + 1, in_w - 1);
            const float alpha = xs - static_cast<float>(x0);

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
                const float p00 = static_cast<float>(input.pixels[idx00 + c]);
                const float p10 = static_cast<float>(input.pixels[idx10 + c]);
                const float p01 = static_cast<float>(input.pixels[idx01 + c]);
                const float p11 = static_cast<float>(input.pixels[idx11 + c]);

                const float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
                const float rounded = std::round(val);
                output.pixels[out_idx + c] = static_cast<std::uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
            }
        }
    }

    return output;
}

} // namespace pixelforge
