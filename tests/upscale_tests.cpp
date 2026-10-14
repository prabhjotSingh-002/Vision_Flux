#include "pixelforge/image.hpp"
#include "pixelforge/upscale.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static int g_failed = 0;
static int g_passed = 0;

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << #cond << ") at line " << __LINE__ << "\n"; \
            g_failed++; \
        } else { \
            g_passed++; \
        } \
    } while(0)

#define ASSERT_THROWS(expr, msg) \
    do { \
        bool threw = false; \
        try { \
            expr; \
        } catch (...) { \
            threw = true; \
        } \
        if (!threw) { \
            std::cerr << "  [FAIL] Expected exception for " << msg << " at line " << __LINE__ << "\n"; \
            g_failed++; \
        } else { \
            g_passed++; \
        } \
    } while(0)

// Helper: create synthetic test image
static pixelforge::Image makeTestImage(int w, int h, int channels) {
    pixelforge::Image img;
    img.width = w;
    img.height = h;
    img.channels = channels;
    img.pixels.resize(static_cast<std::size_t>(w) * h * channels);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::size_t idx = (static_cast<std::size_t>(y) * w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                img.pixels[idx + c] = static_cast<std::uint8_t>((x * 37 + y * 59 + c * 71) % 256);
            }
        }
    }
    return img;
}

void testDimensionsAndScaling() {
    std::cout << "[Test] Dimension and scaling test (2x and 4x)...\n";
    pixelforge::Image img512 = makeTestImage(512, 512, 3);

    pixelforge::Image out2x = pixelforge::upscaleBilinearCPU(img512, 2);
    ASSERT_TRUE(out2x.width == 1024, "512x512 at 2x width == 1024");
    ASSERT_TRUE(out2x.height == 1024, "512x512 at 2x height == 1024");
    ASSERT_TRUE(out2x.channels == 3, "Channels preserved at 2x");
    ASSERT_TRUE(out2x.valid(), "Output 2x image is valid");

    pixelforge::Image out4x = pixelforge::upscaleBilinearCPU(img512, 4);
    ASSERT_TRUE(out4x.width == 2048, "512x512 at 4x width == 2048");
    ASSERT_TRUE(out4x.height == 2048, "512x512 at 4x height == 2048");
    ASSERT_TRUE(out4x.channels == 3, "Channels preserved at 4x");
    ASSERT_TRUE(out4x.valid(), "Output 4x image is valid");
}

void testChannelPreservation() {
    std::cout << "[Test] Channel count preservation (1, 3, 4 channels)...\n";
    for (int ch : {1, 3, 4}) {
        pixelforge::Image img = makeTestImage(64, 64, ch);
        pixelforge::Image out = pixelforge::upscaleBilinearCPU(img, 2);
        ASSERT_TRUE(out.channels == ch, "Channel count match");
        ASSERT_TRUE(out.valid(), "Valid multi-channel output");
    }
}

void testSinglePixel() {
    std::cout << "[Test] 1x1 image handling...\n";
    pixelforge::Image img1x1;
    img1x1.width = 1;
    img1x1.height = 1;
    img1x1.channels = 3;
    img1x1.pixels = {120, 180, 240};

    pixelforge::Image out = pixelforge::upscaleBilinearCPU(img1x1, 2);
    ASSERT_TRUE(out.width == 2 && out.height == 2, "1x1 at 2x -> 2x2");
    for (std::size_t i = 0; i < out.pixels.size(); i += 3) {
        ASSERT_TRUE(out.pixels[i + 0] == 120, "1x1 pixel Red matches");
        ASSERT_TRUE(out.pixels[i + 1] == 180, "1x1 pixel Green matches");
        ASSERT_TRUE(out.pixels[i + 2] == 240, "1x1 pixel Blue matches");
    }
}

void testNonMultipleDimensions() {
    std::cout << "[Test] Non-multiple image dimensions...\n";
    pixelforge::Image odd = makeTestImage(17, 23, 3);
    pixelforge::Image out2x = pixelforge::upscaleBilinearCPU(odd, 2);
    ASSERT_TRUE(out2x.width == 34 && out2x.height == 46, "17x23 at 2x -> 34x46");
    ASSERT_TRUE(out2x.valid(), "Odd dimensions valid");

    pixelforge::Image out4x = pixelforge::upscaleBilinearCPU(odd, 4);
    ASSERT_TRUE(out4x.width == 68 && out4x.height == 92, "17x23 at 4x -> 68x92");
    ASSERT_TRUE(out4x.valid(), "Odd dimensions at 4x valid");
}

void testInvalidScale() {
    std::cout << "[Test] Invalid scale error handling...\n";
    pixelforge::Image img = makeTestImage(32, 32, 3);
    ASSERT_THROWS(pixelforge::upscaleBilinearCPU(img, 3), "scale 3x rejected");
    ASSERT_THROWS(pixelforge::upscaleBilinearCPU(img, 1), "scale 1x rejected");
    ASSERT_THROWS(pixelforge::upscaleBilinearCPU(img, -2), "negative scale rejected");
}

void testCpuVsCudaIfAvailable() {
    std::cout << "[Test] CPU vs CUDA correctness validation (if CUDA available)...\n";
    pixelforge::Image img = makeTestImage(128, 128, 3);
    pixelforge::Image cpu_out = pixelforge::upscaleBilinearCPU(img, 2);

    pixelforge::Image cuda_out;
    std::string err;
    bool ok = pixelforge::upscaleBilinearCUDA(img, 2, cuda_out, err);

    if (!ok) {
        std::cout << "  [Notice] Skipping GPU comparison: " << err << "\n";
        return;
    }

    ASSERT_TRUE(cuda_out.width == cpu_out.width, "Width matches");
    ASSERT_TRUE(cuda_out.height == cpu_out.height, "Height matches");
    ASSERT_TRUE(cuda_out.channels == cpu_out.channels, "Channels match");
    ASSERT_TRUE(cuda_out.pixels.size() == cpu_out.pixels.size(), "Buffer size matches");

    int max_diff = 0;
    for (std::size_t i = 0; i < cpu_out.pixels.size(); ++i) {
        int diff = std::abs(static_cast<int>(cpu_out.pixels[i]) - static_cast<int>(cuda_out.pixels[i]));
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "  Max absolute difference between CPU and CUDA: " << max_diff << "\n";
    ASSERT_TRUE(max_diff <= 1, "CPU and CUDA max difference <= 1 intensity level");
}

int main() {
    std::cout << "========================================================\n"
              << " PixelForge Upscale Test Suite\n"
              << "========================================================\n";

    testDimensionsAndScaling();
    testChannelPreservation();
    testSinglePixel();
    testNonMultipleDimensions();
    testInvalidScale();
    testCpuVsCudaIfAvailable();

    std::cout << "========================================================\n";
    std::cout << "Results: " << g_passed << " passed, " << g_failed << " failed.\n";
    std::cout << "========================================================\n";

    return (g_failed == 0) ? 0 : 1;
}
