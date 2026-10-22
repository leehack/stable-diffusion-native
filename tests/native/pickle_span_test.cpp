// Runs upstream's pickle scanner, compiled as the library compiles it, on the
// spans read_torch_legacy_file gives it. That reader passes the rest of the
// whole checkpoint, tensor storage included, so a small pickle can sit in a
// span longer than INT_MAX.
//
// Modes:
//   bounded    each pickle in a buffer of exactly its size
//   large      malformed pickles at the start of spans above INT_MAX
//   large-ok   valid pickles at the start of spans above INT_MAX
//
// Large spans are mapped read-only, so they cost no memory; only the first
// page is writable and holds the pickle.

#include "model_io/pickle_io.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

// Stands in for upstream's util.cpp; pickle_io.cpp calls it only to format errors.
std::string sd_format(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return buffer;
}

namespace {

struct Pickle {
    const char* name;
    std::string bytes;
};

// Keeps embedded NULs; drops the literal's terminator.
template <size_t N>
std::string bytes(const char (&data)[N]) {
    return std::string(data, N - 1);
}

// Every opcode whose operand skip_pickle_object scans to a newline, plus a
// length-prefixed string, which does not depend on the span.
std::vector<Pickle> valid_pickles() {
    return {
        {"GLOBAL", bytes("\x80\x02" "ccollections\nOrderedDict\n)R.")},
        {"INST", bytes("\x80\x02(icollections\nOrderedDict\n.")},
        {"PERSID", bytes("\x80\x02P0\n.")},
        {"UNICODE", bytes("\x80\x02Vweight\n.")},
        {"BINUNICODE", bytes("\x80\x02X\x06\x00\x00\x00weight.")},
    };
}

std::vector<Pickle> malformed_pickles() {
    return {
        {"GLOBAL without a name line", bytes("\x80\x02" "ccollections\nOrderedDict")},
        {"GLOBAL without a module line", bytes("\x80\x02" "ccollections")},
        {"PERSID without a newline", bytes("\x80\x02P0")},
        {"BINUNICODE past the end", bytes("\x80\x02X\xff\x00\x00\x00weight.")},
        {"unknown opcode", bytes("\x80\x02\x00.")},
        {"no STOP", bytes("\x80\x02)")},
    };
}

bool skip(const uint8_t* data, size_t size, size_t* consumed) {
    *consumed = 0;
    return skip_pickle_object(data, size, consumed);
}

void check_bounded() {
    for (const Pickle& pickle : valid_pickles()) {
        // An exact-size heap copy lets a sanitizer catch reads past the end.
        std::vector<uint8_t> data(pickle.bytes.begin(), pickle.bytes.end());
        size_t consumed = 0;
        std::printf("bounded %s\n", pickle.name);
        CHECK(skip(data.data(), data.size(), &consumed));
        CHECK(consumed == data.size());
    }
    for (const Pickle& pickle : malformed_pickles()) {
        std::vector<uint8_t> data(pickle.bytes.begin(), pickle.bytes.end());
        size_t consumed = 0;
        std::printf("bounded %s\n", pickle.name);
        CHECK(!skip(data.data(), data.size(), &consumed));
    }
}

// Above INT_MAX, and above UINT32_MAX, where narrowing to int keeps only the
// low 32 bits.
const size_t kLargeSpans[] = {(size_t(1) << 31) + 4096, (size_t(1) << 32) + 16};

uint8_t* map_span(size_t span, const std::string& pickle) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void* mapped      = mmap(nullptr, span, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mapped != MAP_FAILED);
    CHECK(mprotect(mapped, page, PROT_READ | PROT_WRITE) == 0);
    CHECK(pickle.size() <= page);
    std::memcpy(mapped, pickle.data(), pickle.size());
    return static_cast<uint8_t*>(mapped);
}

void check_large(const std::vector<Pickle>& pickles, bool valid) {
    for (size_t span : kLargeSpans) {
        for (const Pickle& pickle : pickles) {
            uint8_t* data   = map_span(span, pickle.bytes);
            size_t consumed = 0;
            std::printf("span %zu %s\n", span, pickle.name);
            std::fflush(stdout);
            const bool skipped = skip(data, span, &consumed);
            CHECK(munmap(data, span) == 0);
            if (valid) {
                CHECK(skipped);
                CHECK(consumed == pickle.bytes.size());
            } else {
                CHECK(!skipped);
            }
        }
    }
}

// The rest of a malformed pickle in a large span is zeros, which the scanner
// rejects as an unknown opcode; a scan for a missing newline would read up to
// INT_MAX bytes of them, so those cases stay bounded only.
std::vector<Pickle> malformed_large_pickles() {
    return {
        {"no STOP", bytes("\x80\x02)")},
        {"BINUNICODE past the pickle", bytes("\x80\x02X\x00\x00\x00\x80")},
    };
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "bounded") {
        check_bounded();
    } else if (mode == "large") {
        check_large(malformed_large_pickles(), false);
    } else if (mode == "large-ok") {
        check_large(valid_pickles(), true);
    } else {
        std::fprintf(stderr, "usage: %s bounded|large|large-ok\n", argv[0]);
        return 2;
    }
    return 0;
}
