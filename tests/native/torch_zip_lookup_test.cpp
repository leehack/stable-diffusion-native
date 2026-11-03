// Real upstream ZIP/pickle readers with observed API calls; no tensor backend.
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model_io/pickle_io.h"
#include "zip.h"
#include "json.hpp"

extern "C" size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_I32: return 4;
        case GGML_TYPE_F16: case GGML_TYPE_BF16: return 2;
        default: throw std::runtime_error("unexpected metadata scalar type");
    }
}
extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    (void)ggml_type_size(type);
    return 1;
}
std::string sd_format(const char* format, ...) {
    char buffer[64];
    va_list args;
    va_start(args, format);
    const int count = std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (count < 0 || static_cast<size_t>(count) >= sizeof(buffer))
        throw std::runtime_error("error format overflow");
    return buffer;
}

namespace {
size_t opens = 0, names = 0, archives_opened = 0, archives_closed = 0;
size_t metadata_reads = 0, payload_reads = 0;
std::unordered_set<void*> pickle_allocations;
std::string fault;
zip_t* observed_zip_open(const char* path, int level, char mode) {
    zip_t* zip = zip_open(path, level, mode);
    if (zip) ++archives_opened;
    return zip;
}
void observed_zip_close(zip_t* zip) {
    if (zip) ++archives_closed;
    zip_close(zip);
}
ssize_t observed_entry_count(zip_t* zip) {
    if (fault == "negative-count") return -1;
    if (fault == "large-count") return static_cast<ssize_t>(INT_MAX) + 1;
    return zip_entries_total(zip);
}
int observed_entry_open(zip_t* zip, size_t index) {
    ++opens;
    // A failed open may already have allocated entry state. Exercise cleanup
    // after the real call, rather than replacing the ZIP implementation.
    const int result = zip_entry_openbyindex(zip, index);
    if ((fault == "outer-open" && opens == 1) ||
        (fault == "lookup-open" && opens == 2)) return ZIP_ENOENT;
    return result;
}
const char* observed_entry_name(zip_t* zip) {
    ++names;
    if ((fault == "outer-name" && names == 1) ||
        (fault == "lookup-name" && names == 2)) return nullptr;
    return zip_entry_name(zip);
}
ssize_t observed_entry_read(zip_t* zip, void** buffer, size_t* size) {
    const std::string name = zip_entry_name(zip);
    if (name.find("data.pkl") == std::string::npos) {
        ++payload_reads;
        throw std::runtime_error("tensor payload extraction refused");
    }
    ++metadata_reads;
    const ssize_t result = zip_entry_read(zip, buffer, size);
    if (*buffer) pickle_allocations.insert(*buffer);
    return result;
}
void observed_free(void* ptr) {
    pickle_allocations.erase(ptr);
    std::free(ptr);
}
}  // namespace

#define zip_open observed_zip_open
#define zip_close observed_zip_close
#define zip_entries_total observed_entry_count
#define zip_entry_openbyindex observed_entry_open
#define zip_entry_name observed_entry_name
#define zip_entry_read observed_entry_read
#define free observed_free
#include "model_io/torch_zip_io.cpp"
#undef free
#undef zip_entry_read
#undef zip_entry_name
#undef zip_entry_openbyindex
#undef zip_entries_total
#undef zip_close
#undef zip_open

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    fault = argv[1];
    for (int i = 2; i < argc; ++i) {
        opens = names = archives_opened = archives_closed = metadata_reads = payload_reads = 0;
        if (!pickle_allocations.empty()) return 4;
        std::string error;
        std::vector<TensorStorage> tensors;
        const bool ok = read_torch_zip_file(argv[i], tensors, &error);
        nlohmann::json inventory = nlohmann::json::array();
        for (const auto& t : tensors) {
            inventory.push_back({{"name", t.name}, {"type", t.type},
                {"dims", std::vector<int64_t>(t.ne, t.ne + SD_MAX_DIMS)}, {"n_dims", t.n_dims},
                {"index", t.index_in_zip}, {"offset", t.offset}, {"bytes", t.nbytes_to_read()},
                {"storage_key", t.storage_key}, {"f64", t.is_f64}, {"i64", t.is_i64}});
        }
        std::cout << nlohmann::json{{"ok", ok}, {"error", error}, {"inventory", inventory},
            {"entry_opens", opens}, {"entry_names", names}, {"archives_opened", archives_opened},
            {"archives_closed", archives_closed}, {"live_pickle_buffers", pickle_allocations.size()},
            {"metadata_reads", metadata_reads}, {"payload_reads", payload_reads}}.dump() << '\n';
    }
}
