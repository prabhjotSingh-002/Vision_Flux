cmake_minimum_required(VERSION 3.18)
project(PixelForge LANGUAGES CXX)

# Compilation Options
option(PIXELFORGE_ENABLE_CUDA "Build with CUDA acceleration" OFF)
option(PIXELFORGE_BUILD_TESTS "Build test suite" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Core source files shared across builds
set(CORE_SOURCES
    src/common/image_io.cpp
    src/upscale/upscale_cpu.cpp
)

if(PIXELFORGE_ENABLE_CUDA)
    message(STATUS "[PixelForge] Configuring with CUDA Backend ENABLED")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    list(APPEND CORE_SOURCES src/upscale/upscale_cuda.cu)

    # Support modern NVIDIA GPU architectures (Turing, Ampere, Ada Lovelace, Hopper)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
else()
    message(STATUS "[PixelForge] Configuring with CPU-Only Backend (CUDA disabled/stubbed)")
    list(APPEND CORE_SOURCES src/upscale/upscale_cuda_stub.cpp)
endif()

# Main CLI Application Target
add_executable(pixelforge
    src/app/main.cpp
    ${CORE_SOURCES}
)

target_include_directories(pixelforge PRIVATE
    include
    third_party
)

if(PIXELFORGE_ENABLE_CUDA)
    target_link_libraries(pixelforge PRIVATE CUDA::cudart)
    set_target_properties(pixelforge PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# Unit & Correctness Tests Target
if(PIXELFORGE_BUILD_TESTS)
    add_executable(upscale_tests
        tests/upscale_tests.cpp
        ${CORE_SOURCES}
    )

    target_include_directories(upscale_tests PRIVATE
        include
        third_party
    )

    if(PIXELFORGE_ENABLE_CUDA)
        target_link_libraries(upscale_tests PRIVATE CUDA::cudart)
        set_target_properties(upscale_tests PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
    endif()

    enable_testing()
    add_test(NAME UpscaleCorrectnessTests COMMAND upscale_tests)
endif()
