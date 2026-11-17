#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

static bool from_library(void* caller) {
    Dl_info info;
    return dladdr(caller, &info) != 0 && info.dli_fname != nullptr &&
           std::strstr(info.dli_fname, "libstable-diffusion") != nullptr;
}

// Test-only dyld interposition: observe actual worker scratch addresses, not
// allocator totals. Fixed storage avoids allocating from inside an allocator.
// The immediate VM-region query proves virtual unmapping; physical footprint
// and jetsam behavior still need the original iOS controls.
struct ScratchMapping {
    std::atomic<vm_address_t> address{0};
    std::atomic<vm_size_t> size{0};
};
static ScratchMapping mappings[2048];
static std::atomic<size_t> allocations{0}, releases{0}, unmapped{0};

static kern_return_t observed_vm_allocate(vm_map_t task, vm_address_t* address,
                                          vm_size_t size, int flags) {
    const auto result = vm_allocate(task, address, size, flags);
    if (result == KERN_SUCCESS && pthread_main_np() == 0 &&
        from_library(__builtin_return_address(0))) {
        const size_t index = allocations.fetch_add(1);
        if (index >= 2048) std::abort();
        mappings[index].size.store(size);
        mappings[index].address.store(*address);
    }
    return result;
}

static kern_return_t observed_vm_deallocate(vm_map_t task, vm_address_t address, vm_size_t size) {
    // A freed address can be immediately reused by another loader worker;
    // claim its record before releasing it to the kernel.
    bool tracked = false;
    for (size_t i = 0; i < allocations.load() && i < 2048; ++i) {
        vm_address_t expected = address;
        if (mappings[i].size.load() == size && mappings[i].address.compare_exchange_strong(expected, 0)) {
            tracked = true;
            break;
        }
    }
    const auto result = vm_deallocate(task, address, size);
    if (tracked && result == KERN_SUCCESS) {
        releases.fetch_add(1);
        // The fixtures select one loader thread, avoiding other scratch
        // allocations racing this immediate query.
        mach_vm_address_t region = address;
        mach_vm_size_t region_size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        const auto status = mach_vm_region(task, &region, &region_size, VM_REGION_BASIC_INFO_64,
                                           reinterpret_cast<vm_region_info_t>(&info), &count, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        if (status == KERN_INVALID_ADDRESS || (status == KERN_SUCCESS && region > address)) {
            unmapped.fetch_add(1);
        }
    }
    return result;
}

__attribute__((used, section("__DATA,__interpose")))
static const struct { const void* replacement; const void* original; } interpose[] = {
    {reinterpret_cast<const void*>(observed_vm_allocate), reinterpret_cast<const void*>(vm_allocate)},
    {reinterpret_cast<const void*>(observed_vm_deallocate), reinterpret_cast<const void*>(vm_deallocate)},
};

extern "C" void sd_test_vm_report() {
    size_t live = 0;
    for (size_t i = 0; i < allocations.load() && i < 2048; ++i) {
        if (mappings[i].address.load() != 0) ++live;
    }
    std::printf("VM_SCRATCH %zu %zu %zu %zu\n", allocations.load(), releases.load(), unmapped.load(), live);
}
