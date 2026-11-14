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
# a patch no longer applies or changes nothing. A patch's sources belong to
# one target: stable-diffusion unless TARGET names another, such as ggml-base.
find_package(Git REQUIRED)
function(sd_native_carry_patch patch)
    cmake_parse_arguments(PARSE_ARGV 1 carry "" "TARGET" "")
    if(NOT carry_TARGET)
        set(carry_TARGET stable-diffusion)
    endif()
    set(carry_paths ${carry_UNPARSED_ARGUMENTS})
    get_filename_component(name "${patch}" NAME_WE)
    set(patched_root "${CMAKE_BINARY_DIR}/patched")
    set(patched_dir "${patched_root}/${name}")
    file(REMOVE_RECURSE "${patched_dir}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${patch}")
    get_target_property(target_dir ${carry_TARGET} SOURCE_DIR)
    get_target_property(target_sources ${carry_TARGET} SOURCES)
    set(sources)
    foreach(source IN LISTS target_sources)
        if(NOT source MATCHES "^\\$<")
            get_filename_component(source "${source}" ABSOLUTE BASE_DIR "${target_dir}")
        endif()
        list(APPEND sources "${source}")
    endforeach()
    foreach(path IN LISTS carry_paths)
        set(upstream_file "${SD_NATIVE_UPSTREAM_DIR}/${path}")
        string(MD5 source_key "${upstream_file}")
        get_property(current_source GLOBAL PROPERTY "SD_NATIVE_PATCHED_${source_key}")
        if(NOT current_source)
            set(current_source "${upstream_file}")
        endif()
        if(NOT "${current_source}" IN_LIST sources)
            message(FATAL_ERROR "patches/${patch}: ${path} is not a ${carry_TARGET} source")
        endif()
        file(READ "${current_source}" text)
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

    foreach(path IN LISTS carry_paths)
        set(upstream_file "${SD_NATIVE_UPSTREAM_DIR}/${path}")
        string(MD5 source_key "${upstream_file}")
        get_property(current_source GLOBAL PROPERTY "SD_NATIVE_PATCHED_${source_key}")
        if(NOT current_source)
            set(current_source "${upstream_file}")
        endif()
        set(patched_file "${patched_dir}/${path}")
        file(READ "${current_source}" original)
        string(REPLACE "\r\n" "\n" original "${original}")
        file(READ "${patched_file}" patched)
        if(original STREQUAL patched)
            message(FATAL_ERROR "patches/${patch} left ${path} unchanged:\n${output}")
        endif()
        list(REMOVE_ITEM sources "${current_source}")
        list(APPEND sources "${patched_file}")
        set_property(GLOBAL PROPERTY "SD_NATIVE_PATCHED_${source_key}" "${patched_file}")
        # Quoted includes in the copy must still find the upstream files next to it.
        get_filename_component(upstream_file_dir "${upstream_file}" DIRECTORY)
        set_source_files_properties("${patched_file}" TARGET_DIRECTORY ${carry_TARGET}
            PROPERTIES INCLUDE_DIRECTORIES "${upstream_file_dir}")
    endforeach()
    set_property(TARGET ${carry_TARGET} PROPERTY SOURCES ${sources})
    message(STATUS "Carrying patches/${patch}")
endfunction()

sd_native_carry_patch(pickle_line_span.patch src/model_io/pickle_io.cpp)
sd_native_carry_patch(torch_zip_entry_lookup.patch src/model_io/torch_zip_io.cpp)
sd_native_carry_patch(fp8_metal_f16_load.patch src/model_manager.cpp src/core/backend_fit.cpp)
sd_native_carry_patch(lora_immediate_fallback.patch src/pipeline/diffusion_engine.cpp)
sd_native_carry_patch(intrinsic_tiny_decoder_policy.patch src/stable-diffusion.cpp src/pipeline/diffusion_engine.cpp)
sd_native_carry_patch(ggml_apple_vm_allocate.patch TARGET ggml-base ggml/src/ggml.c)
if(TARGET ggml-metal)
    sd_native_carry_patch(ggml_metal_apple_vm_allocate.patch TARGET ggml-metal ggml/src/ggml-metal/ggml-metal-device.m)
endif()

# Wrapper exports are compiled into the upstream library target, so they share
# its SD_API export macro and reach upstream without patching it.
target_sources(stable-diffusion PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_decoder_policy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_device.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_exit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd_dart_graph_split.cpp"
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
if(SD_METAL)
    set(SD_DART_METAL 1)
else()
    set(SD_DART_METAL 0)
endif()
target_compile_definitions(stable-diffusion PRIVATE
    SD_DART_GPU_BACKEND=${SD_DART_GPU_BACKEND}
    SD_DART_METAL=${SD_DART_METAL})

# tools/build.py checks these: a missing definition falls back to 0 in the
# sources and would drop the Metal guard without a build error.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sd_dart_definitions-$<CONFIG>.txt"
    CONTENT "$<JOIN:$<TARGET_PROPERTY:stable-diffusion,COMPILE_DEFINITIONS>,\n>\n")
