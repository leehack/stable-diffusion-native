cmake_minimum_required(VERSION 3.13)
project(stable_diffusion_native LANGUAGES C CXX)

set(SD_NATIVE_UPSTREAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp")
if(NOT EXISTS "${SD_NATIVE_UPSTREAM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing submodule: third_party/stable-diffusion.cpp. "
        "Run: git submodule update --init --recursive")
endif()

add_subdirectory("${SD_NATIVE_UPSTREAM_DIR}")

# Wrapper exports are compiled into the upstream library target, so they share
# its SD_API export macro and reach upstream without patching it.
target_sources(stable-diffusion PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_wrapper.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_wrapper.h")
