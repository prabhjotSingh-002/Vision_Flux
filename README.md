# PixelForge — High-Performance Image Processing with CUDA & C++17

PixelForge is a modular, high-performance image processing engine built using **C++17** and **NVIDIA CUDA C++**. It provides reference CPU implementations alongside custom massively parallel GPU kernels for image transformations.

The initial release focuses on **Bilinear Image Upscaling (2× and 4×)**, designed with an isolated architecture so collaborators can develop additional modules (e.g., Grayscale conversion, Denoising) independently without merge conflicts.

---

## 🌟 Key Features

- **Decoupled Architecture**: Common `Image` representation in `include/pixelforge/image.hpp` enables teammates to work simultaneously on upscaling and grayscale modules.
- **Dual-Backend Support**:
  - **CPU Reference**: High-precision reference implementation with boundary clamping and rounding.
  - **CUDA GPU Kernel**: High-throughput 2D grid kernel mapping destination pixels directly to CUDA thread blocks.
  - **Zero-Dependency CPU Build**: Includes a stub backend (`upscale_cuda_stub.cpp`) allowing compilation and testing on weak or non-GPU local development PCs without requiring CUDA Toolkit.
- **Lightweight Dependencies**: Uses `stb_image` and `stb_image_write` single-header libraries (no heavy OpenCV build requirements).
- **Correctness First**: Automated test suite (`tests/upscale_tests.cpp`) verifying dimension scaling, odd boundaries, 1×1 images, channel preservation, and CPU vs GPU tolerance ($\le 1$ intensity level).
- **Comprehensive Benchmarks**: Host and CUDA event timers reporting CPU time, GPU kernel time, GPU end-to-end time, and speedup ratios with CSV and Matplotlib visualizations.

---

## 📁 Repository Structure

```
PixelForge/
├── CMakeLists.txt                # CMake build configuration with CUDA toggle
├── README.md                     # Project documentation
├── .gitignore                    # Build and output ignore rules
├── PixelForge_Colab_Benchmark.ipynb # Ready-to-run Google Colab GPU notebook
│
├── include/
│   └── pixelforge/
│       ├── image.hpp             # Shared image data structure
│       ├── image_io.hpp          # PNG/JPG loading and saving interfaces
│       ├── upscale.hpp           # Upscaling public API (CPU, CUDA, Benchmark)
│       └── grayscale.hpp         # Future module header (collaborator isolation)
│
├── src/
│   ├── app/
│   │   └── main.cpp              # CLI application (upscale & benchmark commands)
│   ├── common/
│   │   └── image_io.cpp          # stb_image / stb_image_write implementation
│   ├── upscale/
│   │   ├── upscale_cpu.cpp       # CPU bilinear interpolation reference
│   │   ├── upscale_cuda.cu       # NVIDIA CUDA 2D bilinear upscaling kernel
│   │   └── upscale_cuda_stub.cpp # Fallback stub when CUDA is disabled
│   └── grayscale/                # Future collaborator module layout
│
├── tests/
│   └── upscale_tests.cpp         # Correctness & unit test suite
│
├── scripts/
│   └── plot_benchmarks.py        # Python benchmark plotting tool
│
├── samples/
│   └── demo.png                  # Sample test image (256x256 RGB)
├── outputs/                      # Generated images destination
└── benchmarks/                   # Benchmark CSV results and plots
```

---

## 📐 Bilinear Interpolation Algorithm

For destination pixel coordinates $(x_d, y_d)$ and scale factor $s \in \{2, 4\}$:

1. **Normalized Centered Coordinate Mapping**:
   $$x_s = \frac{x_d + 0.5}{s} - 0.5, \quad y_s = \frac{y_d + 0.5}{s} - 0.5$$

2. **Boundary Clamping**:
   $$x_s = \max(0, \min(x_s, W - 1)), \quad y_s = \max(0, \min(y_s, H - 1))$$

3. **Neighbor Sampling & Fractional Weights**:
   $$x_0 = \lfloor x_s \rfloor, \quad x_1 = \min(x_0 + 1, W - 1), \quad \alpha = x_s - x_0$$
   $$y_0 = \lfloor y_s \rfloor, \quad y_1 = \min(y_0 + 1, H - 1), \quad \beta = y_s - y_0$$

4. **Per-Channel Value Interpolation**:
   $$v = (1 - \alpha)(1 - \beta) p_{00} + \alpha (1 - \beta) p_{10} + (1 - \alpha) \beta p_{01} + \alpha \beta p_{11}$$
   Output pixel is rounded and clamped to $[0, 255]$:
   $$p_{\text{out}} = \text{clamp}(\text{round}(v), 0, 255)$$

Both CPU and CUDA implementations share this exact mapping convention to guarantee matching outputs.

---

## 💻 Local PC Setup & Build (CPU-Only Mode)

No NVIDIA GPU or CUDA Toolkit is required on your local development machine!

### Prerequisites
- C++17 compiler (`g++`, `clang++`, or `MSVC`)
- CMake 3.18+
- Python 3 (optional, for benchmark plotting)

### Build Commands
```bash
# 1. Configure for CPU-only build
cmake -S . -B build -DPIXELFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release

# 2. Build pixelforge and test suite
cmake --build build --parallel 2

# 3. Run correctness test suite
./build/upscale_tests
```

### Run CLI Commands
```bash
# Upscale sample image 2x on CPU
./build/pixelforge upscale samples/demo.png outputs/demo_2x.png --scale 2 --backend cpu

# Upscale sample image 4x on CPU
./build/pixelforge upscale samples/demo.png outputs/demo_4x.png --scale 4 --backend cpu

# Run benchmark suite on CPU
./build/pixelforge benchmark samples/demo.png --scale 2 --csv benchmarks/results.csv
```

---

## ⚡ Google Colab Setup (Remote CUDA Testing)

Test the custom CUDA kernel on a remote NVIDIA GPU without installing anything locally:

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** → **Change runtime type** → **Hardware accelerator: GPU** (e.g. T4 or A100).
3. Upload `PixelForge_Colab_Benchmark.ipynb` or run the following commands in code cells:

```bash
# Step 0: Check GPU availability and compilers
!nvidia-smi
!nvcc --version
!cmake --version

# Step 1: Clone repository
!git clone YOUR_REPOSITORY_URL PixelForge
%cd PixelForge
!git checkout feature/cuda-upscale

# Step 2: Build with CUDA enabled
!cmake -S . -B build -DPIXELFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
!cmake --build build --parallel 2

# Step 3: Run correctness tests (validates CPU vs CUDA outputs)
!./build/upscale_tests

# Step 4: Upscale with CUDA kernel
!./build/pixelforge upscale samples/demo.png outputs/demo_cuda_2x.png --scale 2 --backend cuda

# Step 5: Run full benchmark matrix
!./build/pixelforge benchmark --scale 2 --warmup 5 --runs 25 --csv benchmarks/results.csv
!./build/pixelforge benchmark --scale 4 --warmup 5 --runs 25 --csv benchmarks/results.csv

# Step 6: Generate performance graph
!python scripts/plot_benchmarks.py benchmarks/results.csv benchmarks/benchmark_plot.png
```

---

## 📊 Benchmark Metrics Reported

- **`cpu_compute_ms`**: Median execution time of the CPU reference implementation.
- **`gpu_kernel_ms`**: Time spent strictly executing `bilinearUpscaleKernel` on device, measured with `cudaEvent_t`.
- **`gpu_end_to_end_ms`**: Total latency including `cudaMalloc`, host-to-device memory copy (`cudaMemcpy`), kernel execution, device-to-host memory copy, and cleanup.
- **`speedup`**: $\frac{\text{cpu\_compute\_ms}}{\text{gpu\_end\_to\_end\_ms}}$.

> **Note on Performance Overhead**: On smaller images (e.g. $256 \times 256$), PCIe transfer overhead may dominate execution time. As image size increases ($1024 \times 1024$ and above), GPU parallelism achieves significant end-to-end acceleration.

---

## 🤝 Collaborator Workflow (Zero Merge Conflicts)

To work seamlessly with your teammate:

| Role | Branch | Focus Area |
|------|--------|------------|
| **Lead / Upscale Owner** | `feature/cuda-upscale` | `src/upscale/`, `tests/upscale_tests.cpp`, `CMakeLists.txt` |
| **Friend / Grayscale Owner** | `feature/grayscale` | `src/grayscale/`, `tests/grayscale_tests.cpp` |
| **Integrated Production** | `main` | Reviewed and merged pull requests |

1. Both members rely on the shared `include/pixelforge/image.hpp` and `include/pixelforge/image_io.hpp`.
2. Do not modify each other's module folders (`src/upscale` vs `src/grayscale`).
3. Merge via GitHub Pull Requests after verifying tests on both CPU and Colab GPU.

---

## 📜 License
PixelForge is open-source under the MIT License. Third-party headers `stb_image.h` and `stb_image_write.h` are public domain / MIT (Sean Barrett).
