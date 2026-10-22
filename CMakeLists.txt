cmake_minimum_required(VERSION 3.18)
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

# Fixes the pinned upstream lacks (patches/README.md). Each patch is applied to
# build-tree copies of the upstream sources it changes, and the library
# compiles the copies, so the submodule stays pristine. Configuring fails when
# a patch no longer applies or changes nothing.
find_package(Git REQUIRED)
function(sd_native_carry_patch patch)
    get_filename_component(name "${patch}" NAME_WE)
    set(patched_root "${CMAKE_BINARY_DIR}/patched")
    set(patched_dir "${patched_root}/${name}")
    file(REMOVE_RECURSE "${patched_dir}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${patch}")
    get_target_property(sources stable-diffusion SOURCES)
    foreach(path IN LISTS ARGN)
        set(upstream_file "${SD_NATIVE_UPSTREAM_DIR}/${path}")
        if(NOT "${upstream_file}" IN_LIST sources)
            message(FATAL_ERROR "patches/${patch}: ${path} is not a stable-diffusion source")
        endif()
        file(READ "${upstream_file}" text)
        # A Windows checkout can convert upstream's line endings; patches use LF.
        string(REPLACE "\r\n" "\n" text "${text}")
        file(WRITE "${patched_dir}/${path}" "${text}")
    endforeach()

    # The build tree can sit inside this repository's checkout, where git apply
    # would skip every path outside it and still succeed.
    get_filename_component(ceiling "${patched_root}" REALPATH)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${ceiling}"
                "${GIT_EXECUTABLE}" apply --verbose "${CMAKE_CURRENT_SOURCE_DIR}/patches/${patch}"
        WORKING_DIRECTORY "${patched_dir}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE output)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "patches/${patch} does not apply to the pinned upstream:\n${output}")
    endif()

    foreach(path IN LISTS ARGN)
        set(upstream_file "${SD_NATIVE_UPSTREAM_DIR}/${path}")
        set(patched_file "${patched_dir}/${path}")
        file(READ "${upstream_file}" original)
        string(REPLACE "\r\n" "\n" original "${original}")
        file(READ "${patched_file}" patched)
        if(original STREQUAL patched)
            message(FATAL_ERROR "patches/${patch} left ${path} unchanged:\n${output}")
        endif()
        list(REMOVE_ITEM sources "${upstream_file}")
        list(APPEND sources "${patched_file}")
        # Quoted includes in the copy must still find the upstream files next to it.
        get_filename_component(upstream_file_dir "${upstream_file}" DIRECTORY)
        set_source_files_properties("${patched_file}" TARGET_DIRECTORY stable-diffusion
            PROPERTIES INCLUDE_DIRECTORIES "${upstream_file_dir}")
    endforeach()
    set_property(TARGET stable-diffusion PROPERTY SOURCES ${sources})
    message(STATUS "Carrying patches/${patch}")
endfunction()

sd_native_carry_patch(pickle_line_span.patch src/model_io/pickle_io.cpp)

# Wrapper exports are compiled into the upstream library target, so they share
# its SD_API export macro and reach upstream without patching it.
target_sources(stable-diffusion PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_device.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_exit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_internal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_log.cpp"
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
