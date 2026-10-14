#include "pixelforge/image.hpp"
#include "pixelforge/image_io.hpp"
#include "pixelforge/upscale.hpp"
#include "pixelforge/downscale.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static void printUsage(const char* prog) {
    std::cout << "========================================================\n"
              << " PixelForge — High-Performance Image Processing CLI\n"
              << "========================================================\n"
              << "Usage:\n"
              << "  " << prog << " upscale <input_path> <output_path> [options]\n"
              << "  " << prog << " downscale <input_path> <output_path> [options]\n"
              << "  " << prog << " benchmark [input_path] [options]\n"
              << "  " << prog << " --help\n\n"
              << "Scaling Options (upscale & downscale):\n"
              << "  --scale <2|4>          Scaling factor (default: 2)\n"
              << "  --backend <cpu|cuda>   Processing backend (default: cpu)\n\n"
              << "Benchmark Options:\n"
              << "  --scale <2|4>          Scaling factor (default: 2)\n"
              << "  --warmup <N>           Warm-up runs (default: 5)\n"
              << "  --runs <N>             Measured runs (default: 20)\n"
              << "  --csv <file>           Append results to CSV file\n\n"
              << "Examples:\n"
              << "  " << prog << " upscale samples/demo.png outputs/demo_2x.png --scale 2 --backend cpu\n"
              << "  " << prog << " upscale samples/demo.png outputs/demo_cuda_2x.png --scale 2 --backend cuda\n"
              << "  " << prog << " downscale samples/demo.png outputs/demo_half.png --scale 2 --backend cuda\n"
              << "  " << prog << " benchmark samples/demo.png --scale 2 --csv benchmarks/results.csv\n"
              << "========================================================\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string command = argv[1];

    if (command == "--help" || command == "-h" || command == "help") {
        printUsage(argv[0]);
        return 0;
    }

    if (command == "upscale") {
        if (argc < 4) {
            std::cerr << "Error: 'upscale' command requires <input_path> and <output_path>.\n";
            printUsage(argv[0]);
            return 1;
        }

        const std::string input_path = argv[2];
        const std::string output_path = argv[3];
        int scale = 2;
        std::string backend = "cpu";

        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--scale" && i + 1 < argc) {
                scale = std::stoi(argv[++i]);
            } else if (arg == "--backend" && i + 1 < argc) {
                backend = argv[++i];
            } else {
                std::cerr << "Warning: Unknown option '" << arg << "' ignored.\n";
            }
        }

        if (scale != 2 && scale != 4) {
            std::cerr << "Error: Invalid scale '" << scale << "'. Only 2 and 4 are supported.\n";
            return 1;
        }
        if (backend != "cpu" && backend != "cuda") {
            std::cerr << "Error: Invalid backend '" << backend << "'. Must be 'cpu' or 'cuda'.\n";
            return 1;
        }

        std::cout << "[PixelForge] Loading: " << input_path << " ...\n";
        pixelforge::Image in_img;
        std::string err;
        if (!pixelforge::loadImage(input_path, in_img, err)) {
            std::cerr << "[PixelForge Error] " << err << "\n";
            return 1;
        }

        std::cout << "[PixelForge] Image: " << in_img.width << "x" << in_img.height
                  << " (" << in_img.channels << " channels)\n"
                  << "[PixelForge] Upscaling " << scale << "x using [" << backend << "] backend...\n";

        pixelforge::Image out_img;
        auto start_t = std::chrono::high_resolution_clock::now();

        if (backend == "cpu") {
            try {
                out_img = pixelforge::upscaleBilinearCPU(in_img, scale);
            } catch (const std::exception& e) {
                std::cerr << "[PixelForge Error] CPU Upscaling failed: " << e.what() << "\n";
                return 1;
            }
        } else {
            if (!pixelforge::upscaleBilinearCUDA(in_img, scale, out_img, err)) {
                std::cerr << "[PixelForge Error] CUDA Upscaling failed: " << err << "\n";
                return 1;
            }
        }

        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

        std::cout << "[PixelForge] Done in " << std::fixed << std::setprecision(2)
                  << elapsed_ms << " ms. Output size: "
                  << out_img.width << "x" << out_img.height << "\n";

        std::cout << "[PixelForge] Saving: " << output_path << " ...\n";
        if (!pixelforge::saveImagePNG(output_path, out_img, err)) {
            std::cerr << "[PixelForge Error] " << err << "\n";
            return 1;
        }

        std::cout << "[PixelForge] Success! Output saved to: " << output_path << "\n";
        return 0;
    }

    if (command == "downscale") {
        if (argc < 4) {
            std::cerr << "Error: 'downscale' command requires <input_path> and <output_path>.\n";
            printUsage(argv[0]);
            return 1;
        }

        const std::string input_path = argv[2];
        const std::string output_path = argv[3];
        int scale = 2;
        std::string backend = "cpu";

        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--scale" && i + 1 < argc) {
                scale = std::stoi(argv[++i]);
            } else if (arg == "--backend" && i + 1 < argc) {
                backend = argv[++i];
            } else {
                std::cerr << "Warning: Unknown option '" << arg << "' ignored.\n";
            }
        }

        if (scale != 2 && scale != 4) {
            std::cerr << "Error: Invalid scale '" << scale << "'. Only 2 and 4 are supported.\n";
            return 1;
        }
        if (backend != "cpu" && backend != "cuda") {
            std::cerr << "Error: Invalid backend '" << backend << "'. Must be 'cpu' or 'cuda'.\n";
            return 1;
        }

        std::cout << "[PixelForge] Loading: " << input_path << " ...\n";
        pixelforge::Image in_img;
        std::string err;
        if (!pixelforge::loadImage(input_path, in_img, err)) {
            std::cerr << "[PixelForge Error] " << err << "\n";
            return 1;
        }

        std::cout << "[PixelForge] Image: " << in_img.width << "x" << in_img.height
                  << " (" << in_img.channels << " channels)\n"
                  << "[PixelForge] Downscaling " << scale << "x using [" << backend << "] backend...\n";

        pixelforge::Image out_img;
        auto start_t = std::chrono::high_resolution_clock::now();

        if (backend == "cpu") {
            try {
                out_img = pixelforge::downscaleBilinearCPU(in_img, scale);
            } catch (const std::exception& e) {
                std::cerr << "[PixelForge Error] CPU Downscaling failed: " << e.what() << "\n";
                return 1;
            }
        } else {
            if (!pixelforge::downscaleBilinearCUDA(in_img, scale, out_img, err)) {
                std::cerr << "[PixelForge Error] CUDA Downscaling failed: " << err << "\n";
                return 1;
            }
        }

        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

        std::cout << "[PixelForge] Done in " << std::fixed << std::setprecision(2)
                  << elapsed_ms << " ms. Output size: "
                  << out_img.width << "x" << out_img.height << "\n";

        std::cout << "[PixelForge] Saving: " << output_path << " ...\n";
        if (!pixelforge::saveImagePNG(output_path, out_img, err)) {
            std::cerr << "[PixelForge Error] " << err << "\n";
            return 1;
        }

        std::cout << "[PixelForge] Success! Output saved to: " << output_path << "\n";
        return 0;
    }

    if (command == "benchmark") {
        std::string input_path;
        int scale = 2;
        int warmup = 5;
        int runs = 20;
        std::string csv_file = "benchmarks/results.csv";

        int arg_idx = 2;
        if (arg_idx < argc && argv[arg_idx][0] != '-') {
            input_path = argv[arg_idx++];
        }

        while (arg_idx < argc) {
            std::string arg = argv[arg_idx++];
            if (arg == "--scale" && arg_idx < argc) {
                scale = std::stoi(argv[arg_idx++]);
            } else if (arg == "--warmup" && arg_idx < argc) {
                warmup = std::stoi(argv[arg_idx++]);
            } else if (arg == "--runs" && arg_idx < argc) {
                runs = std::stoi(argv[arg_idx++]);
            } else if (arg == "--csv" && arg_idx < argc) {
                csv_file = argv[arg_idx++];
            }
        }

        std::cout << "========================================================\n"
                  << " PixelForge Benchmark Suite\n"
                  << " Scale: " << scale << "x | Warmup: " << warmup << " | Measured Runs: " << runs << "\n"
                  << " CSV Target: " << csv_file << "\n"
                  << "========================================================\n";

        // Check if CSV exists to write header if new
        std::ifstream check_csv(csv_file);
        bool write_header = !check_csv.good();
        check_csv.close();

        std::ofstream csv_out(csv_file, std::ios::app);
        if (write_header && csv_out.is_open()) {
            csv_out << "input_width,input_height,scale,cpu_compute_ms,gpu_kernel_ms,gpu_end_to_end_ms,speedup\n";
        }

        auto runSingleBench = [&](const pixelforge::Image& img, const std::string& label) {
            std::cout << "Testing " << label << " (" << img.width << "x" << img.height << ", "
                      << img.channels << "ch) -> " << (img.width * scale) << "x" << (img.height * scale) << "...\n";

            pixelforge::UpscaleTiming timing;
            std::string benchErr;
            bool gpu_ok = pixelforge::benchmarkUpscale(img, scale, warmup, runs, timing, benchErr);

            if (!gpu_ok) {
                // If CUDA is unavailable, measure CPU alone
                std::cout << "  [Notice] " << benchErr << "\n";
                std::vector<double> cpu_times;
                for (int w = 0; w < warmup; ++w) {
                    volatile auto tmp = pixelforge::upscaleBilinearCPU(img, scale);
                    (void)tmp;
                }
                for (int r = 0; r < runs; ++r) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    volatile auto tmp = pixelforge::upscaleBilinearCPU(img, scale);
                    (void)tmp;
                    auto t1 = std::chrono::high_resolution_clock::now();
                    cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
                std::sort(cpu_times.begin(), cpu_times.end());
                timing.cpu_compute_ms = cpu_times[cpu_times.size() / 2];
                timing.gpu_kernel_ms = 0.0;
                timing.gpu_end_to_end_ms = 0.0;
                timing.speedup = 0.0;
            }

            std::cout << "  CPU compute time:        " << std::fixed << std::setprecision(3) << timing.cpu_compute_ms << " ms\n";
            if (gpu_ok) {
                std::cout << "  GPU kernel-only time:    " << timing.gpu_kernel_ms << " ms\n";
                std::cout << "  GPU end-to-end time:     " << timing.gpu_end_to_end_ms << " ms\n";
                std::cout << "  Speedup (CPU / GPU E2E): " << std::setprecision(2) << timing.speedup << "x\n";
            }

            if (csv_out.is_open()) {
                csv_out << img.width << "," << img.height << "," << scale << ","
                        << std::fixed << std::setprecision(3) << timing.cpu_compute_ms << ","
                        << timing.gpu_kernel_ms << ","
                        << timing.gpu_end_to_end_ms << ","
                        << std::setprecision(2) << timing.speedup << "\n";
            }
        };

        if (!input_path.empty()) {
            pixelforge::Image user_img;
            std::string err;
            if (!pixelforge::loadImage(input_path, user_img, err)) {
                std::cerr << "Failed to load '" << input_path << "': " << err << "\n";
                return 1;
            }
            runSingleBench(user_img, input_path);
        } else {
            // Standard Benchmark Matrix: 256x256, 512x512, 1024x1024 synthetic images
            const std::vector<int> sizes = {256, 512, 1024};
            for (int s : sizes) {
                pixelforge::Image synth;
                synth.width = s;
                synth.height = s;
                synth.channels = 3;
                synth.pixels.resize(static_cast<std::size_t>(s) * s * 3);
                for (int y = 0; y < s; ++y) {
                    for (int x = 0; x < s; ++x) {
                        std::size_t idx = (static_cast<std::size_t>(y) * s + x) * 3;
                        synth.pixels[idx + 0] = static_cast<std::uint8_t>((x * 255) / s);
                        synth.pixels[idx + 1] = static_cast<std::uint8_t>((y * 255) / s);
                        synth.pixels[idx + 2] = static_cast<std::uint8_t>(((x + y) * 128) / s);
                    }
                }
                runSingleBench(synth, "Synthetic " + std::to_string(s) + "x" + std::to_string(s));
            }
        }

        std::cout << "\n[PixelForge] Benchmark complete. Results written to: " << csv_file << "\n";
        return 0;
    }

    std::cerr << "Error: Unknown command '" << command << "'.\n";
    printUsage(argv[0]);
    return 1;
}
