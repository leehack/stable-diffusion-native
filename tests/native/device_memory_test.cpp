// Exercises src/sd_dart_device.cpp against a stand-in for ggml's device
// registry, so it needs no GPU and runs on every host.
//
// Usage: device_memory_test backend|no-backend
// for a build with SD_DART_GPU_BACKEND set to 1 or to 0.

#include "sd_dart_internal.h"
#include "sd_dart_wrapper.h"

#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

// The stand-in for a ggml device.
struct ggml_backend_device {
    std::string name;
    std::string description;
    enum ggml_backend_dev_type type;
    size_t free;
    size_t total;
    bool fails = false;
};

namespace {

std::vector<ggml_backend_device> devices;
int registrations  = 0;
int calls_begun    = 0;
int calls_ended    = 0;
int memory_queries = 0;

const uint64_t kGiB = 1ull << 30;

sd_dart_gpu_device_memory_t untouched() {
    sd_dart_gpu_device_memory_t memory;
    std::memset(&memory, 0x5a, sizeof(memory));
    return memory;
}

bool is_untouched(const sd_dart_gpu_device_memory_t& memory) {
    const sd_dart_gpu_device_memory_t expected = untouched();
    return std::memcmp(&memory, &expected, sizeof(memory)) == 0;
}

// The status for `index`, and that a call which is not OK writes nothing.
int32_t status_of(int32_t index) {
    sd_dart_gpu_device_memory_t memory = untouched();
    const int32_t status               = sd_dart_gpu_device_memory(index, &memory);
    CHECK((status == SD_DART_GPU_OK) != is_untouched(memory));
    return status;
}

sd_dart_gpu_device_memory_t memory_of(int32_t index) {
    sd_dart_gpu_device_memory_t memory = untouched();
    CHECK(sd_dart_gpu_device_memory(index, &memory) == SD_DART_GPU_OK);
    return memory;
}

int test_no_backend() {
    // Devices that a build without a GPU backend would not have: the status
    // comes from the build, and the registry is not asked.
    devices = {{"Vulkan0", "a GPU", GGML_BACKEND_DEVICE_TYPE_GPU, kGiB, 2 * kGiB}};
    CHECK(sd_dart_gpu_device_count() == SD_DART_GPU_NO_BACKEND);
    CHECK(status_of(SD_DART_GPU_DEFAULT_DEVICE) == SD_DART_GPU_NO_BACKEND);
    CHECK(status_of(0) == SD_DART_GPU_NO_BACKEND);
    CHECK(sd_dart_gpu_device_memory(0, nullptr) == SD_DART_GPU_INVALID_ARGUMENT);
    CHECK(status_of(-2) == SD_DART_GPU_INVALID_ARGUMENT);
    CHECK(registrations == 0 && memory_queries == 0 && calls_begun == 0);
    return 0;
}

int test_backend() {
    // A backend that found no device: only the CPU is registered.
    devices = {{"CPU", "the CPU", GGML_BACKEND_DEVICE_TYPE_CPU, 8 * kGiB, 8 * kGiB}};
    CHECK(sd_dart_gpu_device_count() == 0);
    CHECK(registrations == 1);
    CHECK(status_of(SD_DART_GPU_DEFAULT_DEVICE) == SD_DART_GPU_NO_DEVICE);
    CHECK(status_of(0) == SD_DART_GPU_NO_DEVICE);
    CHECK(sd_dart_gpu_device_memory(0, nullptr) == SD_DART_GPU_INVALID_ARGUMENT);
    CHECK(status_of(-2) == SD_DART_GPU_INVALID_ARGUMENT);
    CHECK(status_of(INT32_MIN) == SD_DART_GPU_INVALID_ARGUMENT);
    CHECK(memory_queries == 0);

    // GPU devices are numbered in registry order, past the devices of other
    // types, and the default is upstream's: the first discrete GPU.
    devices = {
        {"CPU", "the CPU", GGML_BACKEND_DEVICE_TYPE_CPU, 8 * kGiB, 8 * kGiB},
        {"Vulkan0", "integrated", GGML_BACKEND_DEVICE_TYPE_IGPU, 3 * kGiB, 16 * kGiB},
        {"BLAS", "an accelerator", GGML_BACKEND_DEVICE_TYPE_ACCEL, 0, 0},
        {"Vulkan1", "discrete", GGML_BACKEND_DEVICE_TYPE_GPU, 5 * kGiB, 24 * kGiB},
        {"Meta", "several devices", GGML_BACKEND_DEVICE_TYPE_META, kGiB, kGiB},
    };
    CHECK(sd_dart_gpu_device_count() == 2);
    sd_dart_gpu_device_memory_t memory = memory_of(0);
    CHECK(std::strcmp(memory.name, "Vulkan0") == 0 && std::strcmp(memory.description, "integrated") == 0);
    CHECK(memory.type == SD_DART_GPU_DEVICE_INTEGRATED);
    CHECK(memory.free_bytes == 3 * kGiB && memory.total_bytes == 16 * kGiB);
    memory = memory_of(1);
    CHECK(std::strcmp(memory.name, "Vulkan1") == 0 && std::strcmp(memory.description, "discrete") == 0);
    CHECK(memory.type == SD_DART_GPU_DEVICE_DISCRETE);
    CHECK(memory.free_bytes == 5 * kGiB && memory.total_bytes == 24 * kGiB);
    CHECK(std::strcmp(memory_of(SD_DART_GPU_DEFAULT_DEVICE).name, "Vulkan1") == 0);
    CHECK(status_of(2) == SD_DART_GPU_NO_DEVICE);
    CHECK(status_of(INT32_MAX) == SD_DART_GPU_NO_DEVICE);

    // Without a discrete GPU the default is the first integrated one.
    devices.erase(devices.begin() + 3);
    CHECK(sd_dart_gpu_device_count() == 1);
    CHECK(std::strcmp(memory_of(SD_DART_GPU_DEFAULT_DEVICE).name, "Vulkan0") == 0);

    // Each call asks the device again.
    devices[1].free = 2 * kGiB;
    CHECK(memory_of(0).free_bytes == 2 * kGiB);

    // Use above the budget, which ggml-vulkan reports as a value that wrapped
    // around, is no free memory; free memory is never more than the total.
    devices[1].free = static_cast<size_t>(0) - kGiB;
    CHECK(memory_of(0).free_bytes == 0);
    devices[1].free = 17 * kGiB;
    CHECK(memory_of(0).free_bytes == 16 * kGiB);
    devices[1].free = 0;
    CHECK(memory_of(0).free_bytes == 0 && memory_of(0).total_bytes == 16 * kGiB);

    // A device that reports no memory, or fails to, is unavailable.
    devices[1].total = 0;
    CHECK(status_of(0) == SD_DART_GPU_UNAVAILABLE);
    CHECK(sd_dart_gpu_device_count() == 1);
    devices[1].total = 16 * kGiB;
    devices[1].fails = true;
    CHECK(status_of(0) == SD_DART_GPU_UNAVAILABLE);
    devices[1].fails = false;
    CHECK(status_of(0) == SD_DART_GPU_OK);

    // Names are cut to fit, and what follows them is cleared.
    devices[1].name        = std::string(200, 'n');
    devices[1].description = std::string(600, 'd');
    memory                 = memory_of(0);
    CHECK(std::string(memory.name) == std::string(sizeof(memory.name) - 1, 'n'));
    CHECK(std::string(memory.description) == std::string(sizeof(memory.description) - 1, 'd'));
    devices[1].name        = "V";
    devices[1].description = "";
    memory                 = memory_of(0);
    for (size_t i = 1; i < sizeof(memory.name); ++i) {
        CHECK(memory.name[i] == '\0');
    }
    for (size_t i = 0; i < sizeof(memory.description); ++i) {
        CHECK(memory.description[i] == '\0');
    }

    // Every call that reached the registry was one that exit teardown waits
    // for, also the one whose device failed.
    CHECK(calls_begun > 0 && calls_begun == calls_ended);
    return 0;
}

}  // namespace

// The stand-ins for ggml and upstream.

size_t ggml_backend_dev_count(void) {
    CHECK(calls_begun == calls_ended + 1);
    return devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    CHECK(index < devices.size());
    return &devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    CHECK(calls_begun == calls_ended + 1);
    for (ggml_backend_device& device : devices) {
        if (device.type == type) {
            return &device;
        }
    }
    return nullptr;
}

const char* ggml_backend_dev_name(ggml_backend_dev_t device) {
    return device->name.c_str();
}

const char* ggml_backend_dev_description(ggml_backend_dev_t device) {
    return device->description.empty() ? nullptr : device->description.c_str();
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    return device->type;
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t* free, size_t* total) {
    ++memory_queries;
    if (device->fails) {
        throw std::runtime_error("the driver failed");
    }
    *free  = device->free;
    *total = device->total;
}

size_t sd_list_devices(char* buffer, size_t buffer_size) {
    CHECK(buffer == nullptr && buffer_size == 0);
    ++registrations;
    return 0;
}

SdDartStaticsCall::SdDartStaticsCall()
    : counted_(true) {
    ++calls_begun;
}

SdDartStaticsCall::~SdDartStaticsCall() {
    calls_ended += counted_ ? 1 : 0;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "backend") {
        return test_backend();
    }
    if (mode == "no-backend") {
        return test_no_backend();
    }
    std::fprintf(stderr, "usage: %s backend|no-backend\n", argv[0]);
    return 2;
}
