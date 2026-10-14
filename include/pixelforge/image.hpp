#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pixelforge {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0; // 1 = grayscale, 3 = RGB, 4 = RGBA
    std::vector<std::uint8_t> pixels;

    bool valid() const {
        return width > 0 && height > 0 &&
               (channels == 1 || channels == 3 || channels == 4) &&
               pixels.size() == static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height) *
                                static_cast<std::size_t>(channels);
    }
};

} // namespace pixelforge
