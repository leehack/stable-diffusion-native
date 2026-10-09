#include "sd_dart_internal.h"
#include "sd_dart_wrapper.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

// Set by the top-level CMakeLists.txt: whether a GPU backend is compiled in.
// A backend that finds no device registers nothing, so the registry cannot
// tell.
#ifndef SD_DART_GPU_BACKEND
#define SD_DART_GPU_BACKEND 0
#endif

namespace {

bool is_gpu(ggml_backend_dev_t device) {
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

// Registers the backends the way upstream does before it lists or picks a
// device, dynamically loaded ones included.
void register_backends() {
    sd_list_devices(nullptr, 0);
}

ggml_backend_dev_t find_device(int32_t index) {
    if (index == SD_DART_GPU_DEFAULT_DEVICE) {
        // Upstream's choice for a context without a backend name.
        ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        return device != nullptr ? device : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (is_gpu(device) && index-- == 0) {
            return device;
        }
    }
    return nullptr;
}

void copy_text(char* destination, size_t capacity, const char* text) {
    const size_t length = text != nullptr ? std::min(std::strlen(text), capacity - 1) : 0;
    if (length > 0) {
        std::memcpy(destination, text, length);
    }
    std::memset(destination + length, 0, capacity - length);
}

}  // namespace

int32_t sd_dart_gpu_device_count(void) {
    if (!SD_DART_GPU_BACKEND) {
        return SD_DART_GPU_NO_BACKEND;
    }
    // ggml's device registry is a static that exit() destroys.
    SdDartStaticsCall call;
#if SD_DART_EXIT_ON_LINUX
    if (call.refused()) {
        return SD_DART_GPU_UNAVAILABLE;
    }
#endif
    try {
        register_backends();
        int32_t devices    = 0;
        const size_t count = ggml_backend_dev_count();
        for (size_t i = 0; i < count; ++i) {
            devices += is_gpu(ggml_backend_dev_get(i)) ? 1 : 0;
        }
        return devices;
    } catch (...) {
        return SD_DART_GPU_UNAVAILABLE;
    }
}

int32_t sd_dart_gpu_device_memory(int32_t device_index, sd_dart_gpu_device_memory_t* out) {
    if (out == nullptr || device_index < SD_DART_GPU_DEFAULT_DEVICE) {
        return SD_DART_GPU_INVALID_ARGUMENT;
    }
    if (!SD_DART_GPU_BACKEND) {
        return SD_DART_GPU_NO_BACKEND;
    }
    // ggml's device registry is a static that exit() destroys.
    SdDartStaticsCall call;
#if SD_DART_EXIT_ON_LINUX
    if (call.refused()) {
        return SD_DART_GPU_UNAVAILABLE;
    }
#endif
    try {
        register_backends();
        ggml_backend_dev_t device = find_device(device_index);
        if (device == nullptr) {
            return SD_DART_GPU_NO_DEVICE;
        }
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        // ggml-vulkan asks the driver here, through calls that throw.
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        if (total_bytes == 0) {
            return SD_DART_GPU_UNAVAILABLE;
        }
        // ggml-vulkan adds up budget less use per heap in unsigned
        // arithmetic, so use above the budget arrives as a value near 2^64.
        if (free_bytes > total_bytes) {
            free_bytes = free_bytes > SIZE_MAX / 2 ? 0 : total_bytes;
        }
        sd_dart_gpu_device_memory_t memory = {};
        memory.total_bytes = total_bytes;
        memory.free_bytes  = free_bytes;
        memory.type        = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU
                                 ? SD_DART_GPU_DEVICE_INTEGRATED
                                 : SD_DART_GPU_DEVICE_DISCRETE;
        copy_text(memory.name, sizeof(memory.name), ggml_backend_dev_name(device));
        copy_text(memory.description, sizeof(memory.description), ggml_backend_dev_description(device));
        *out = memory;
        return SD_DART_GPU_OK;
    } catch (...) {
        return SD_DART_GPU_UNAVAILABLE;
    }
}
