cmake_minimum_required(VERSION 3.13)
project(stable_diffusion_native LANGUAGES C CXX)

set(SD_NATIVE_UPSTREAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp")
if(NOT EXISTS "${SD_NATIVE_UPSTREAM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing submodule: third_party/stable-diffusion.cpp. "
        "Run: git submodule update --init --recursive")
endif()

# Windows builds run close to the 260-character path limit. Keep the upstream
# build directory short, and move ggml's nested Vulkan shader generator build
# from ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix to ep.
define_property(DIRECTORY PROPERTY EP_PREFIX INHERITED
    BRIEF_DOCS "ExternalProject prefix"
    FULL_DOCS "ExternalProject prefix, inherited by the upstream directories")
set_property(DIRECTORY PROPERTY EP_PREFIX "${CMAKE_BINARY_DIR}/ep")
add_subdirectory("${SD_NATIVE_UPSTREAM_DIR}" sd)

# Wrapper exports are compiled into the upstream library target, so they share
# its SD_API export macro and reach upstream without patching it.
target_sources(stable-diffusion PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_device.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_exit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_wrapper.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_wrapper.h")

# sd_dart_gpu_device_count() tells a build without a GPU backend from one
# whose backend found no device, which ggml's registry cannot.
set(SD_DART_GPU_BACKEND 0)
foreach(backend CUDA HIPBLAS METAL VULKAN OPENCL SYCL MUSA)
    if(SD_${backend})
        set(SD_DART_GPU_BACKEND 1)
    endif()
endforeach()
target_compile_definitions(stable-diffusion PRIVATE
    SD_DART_GPU_BACKEND=${SD_DART_GPU_BACKEND})
