#include "sd_dart_internal.h"
#include "sd_dart_wrapper.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <type_traits>

namespace {

// A power of two. Messages are stored one after the other, each behind a
// Header, and the oldest make room for a new one.
constexpr size_t kBufferBytes = 256 * 1024;
constexpr size_t kTextBytes   = SD_DART_LOG_TEXT_SIZE - 1;
// The error messages kept for sd_dart_last_error(), and their length.
constexpr uint64_t kErrors     = 32;
constexpr size_t kErrorBytes = 511;
// How long a thread waits for a flag before it gives up. A flag is held for
// one copy, which takes less than a microsecond, so the wait only ends for a
// holder that the scheduler keeps off its processor that long, or that died
// holding the flag, as ExitProcess() lets one. A logging thread then loses
// its message rather than hang the exit, and the ones after it do not wait
// at all.
constexpr std::chrono::milliseconds kWait{100};
// The clock is read once per this many attempts.
constexpr int kAttemptsPerClockRead = 64;

enum Registration : int { kUnregistered, kRegistering, kRegistered };

struct Header {
    uint32_t length;
    int32_t level;
};

struct Error {
    uint32_t length;
    char text[kErrorBytes];
};

struct Flag {
    std::atomic_flag busy;
    // A logging thread gave up waiting for the flag and none has had it
    // since.
    std::atomic<bool> stuck;
};

// All zeros to begin with, so it takes no space in the library file.
struct Log {
    std::atomic<int> registration;
    // The lowest level recorded, as the distance from SD_LOG_INFO.
    std::atomic<int32_t> level_above_info;
    std::atomic<uint64_t> dropped;
    // The number of the newest error message. Message n is in
    // errors[n % kErrors] until message n + kErrors replaces it.
    std::atomic<uint64_t> errors_recorded;

    // The errors have a flag of their own, so that the reason of a failed
    // call never waits behind the other messages or their reader.
    Flag errors_flag;
    Error errors[kErrors];

    // The rest belongs to the thread that set `messages_flag`.
    Flag messages_flag;
    // The sequence of the newest message, how many messages the buffer holds
    // and the highest sequence a read has returned.
    uint64_t newest;
    uint64_t count;
    uint64_t returned;
    // Where the oldest message begins, and the bytes in use from there.
    size_t head;
    size_t used;
    // Where message `cursor_sequence` begins, or will: the one after the
    // message last read. 0 before the first read.
    uint64_t cursor_sequence;
    size_t cursor;
    char buffer[kBufferBytes];
};

// Never destroyed: threads still log, and callers still read, after exit()
// has run static destructors.
static_assert(std::is_trivially_destructible<Log>::value, "Log must outlive exit()");
static_assert((kBufferBytes & (kBufferBytes - 1)) == 0, "offsets wrap with a mask");
Log state;

// The error messages of the calling thread's last call: those numbered above
// `window_begin`, up to `window_end`.
thread_local uint64_t window_begin = 0;
thread_local uint64_t window_end   = 0;

bool try_acquire(Flag& flag) {
    return !flag.busy.test_and_set(std::memory_order_acquire);
}

// For a thread that reads: waits for the flag, at most kWait.
bool acquire(Flag& flag) {
    if (try_acquire(flag)) {
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() + kWait;
    do {
        for (int i = 0; i < kAttemptsPerClockRead; ++i) {
            std::this_thread::yield();
            if (try_acquire(flag)) {
                return true;
            }
        }
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

// For a thread that logs: does not wait again for a flag that another
// logging thread gave up on.
bool acquire_to_log(Flag& flag) {
    const bool stuck = flag.stuck.load(std::memory_order_relaxed);
    if (stuck ? try_acquire(flag) : acquire(flag)) {
        if (stuck) {
            flag.stuck.store(false, std::memory_order_relaxed);
        }
        return true;
    }
    flag.stuck.store(true, std::memory_order_relaxed);
    return false;
}

void release(Flag& flag) {
    flag.busy.clear(std::memory_order_release);
}

void put(size_t offset, const void* source, size_t length) {
    offset &= kBufferBytes - 1;
    const size_t first = std::min(length, kBufferBytes - offset);
    std::memcpy(state.buffer + offset, source, first);
    std::memcpy(state.buffer, static_cast<const char*>(source) + first, length - first);
}

void get(size_t offset, void* destination, size_t length) {
    offset &= kBufferBytes - 1;
    const size_t first = std::min(length, kBufferBytes - offset);
    std::memcpy(destination, state.buffer + offset, first);
    std::memcpy(static_cast<char*>(destination) + first, state.buffer, length - first);
}

// The longest length up to `end` at which `text` ends between two UTF-8
// sequences, given the byte that follows `end`.
size_t whole_sequences(const char* text, size_t end, char next) {
    for (int i = 0; i < 3 && end > 0 && (static_cast<unsigned char>(next) & 0xC0) == 0x80; ++i) {
        next = text[--end];
    }
    return end;
}

// `length`, or the longest length up to `limit` that does not split a UTF-8
// sequence.
size_t cut(const char* text, size_t length, size_t limit) {
    return length <= limit ? length : whole_sequences(text, limit, text[limit]);
}

void store(int32_t level, const char* origin, size_t origin_length, const char* text, size_t length) {
    length            = cut(text, length, kTextBytes - origin_length);
    const size_t need = sizeof(Header) + origin_length + length;
    while (kBufferBytes - state.used < need) {
        Header oldest;
        get(state.head, &oldest, sizeof(oldest));
        const size_t size = sizeof(oldest) + oldest.length;
        if (state.newest - state.count + 1 > state.returned) {
            state.dropped.fetch_add(1, std::memory_order_relaxed);
        }
        state.head = (state.head + size) & (kBufferBytes - 1);
        state.used -= size;
        --state.count;
    }
    const Header header = {static_cast<uint32_t>(origin_length + length), level};
    const size_t tail   = state.head + state.used;
    put(tail, &header, sizeof(header));
    put(tail + sizeof(header), origin, origin_length);
    put(tail + sizeof(header) + origin_length, text, length);
    state.used += need;
    ++state.count;
    ++state.newest;
}

void store_error(const char* origin, size_t origin_length, const char* text, size_t length) {
    length                = cut(text, length, kErrorBytes - origin_length);
    const uint64_t number = state.errors_recorded.load(std::memory_order_relaxed) + 1;
    Error& error          = state.errors[number % kErrors];
    error.length          = static_cast<uint32_t>(origin_length + length);
    std::memcpy(error.text, origin, origin_length);
    std::memcpy(error.text + origin_length, text, length);
    state.errors_recorded.store(number);
}

// Runs on whichever thread logs, for as long as the library is loaded.
// Allocates nothing and waits for nothing but the copy of another thread.
void record(int32_t level, const char* origin, const char* text) {
    if (text == nullptr) {
        return;
    }
    const bool wanted   = level >= SD_LOG_INFO + state.level_above_info.load(std::memory_order_relaxed);
    const bool is_error = level >= SD_LOG_ERROR;
    if (!wanted && !is_error) {
        return;
    }
    size_t length = std::strlen(text);
    while (length > 0 && (text[length - 1] == '\n' || text[length - 1] == '\r')) {
        --length;
    }
    if (length == 0) {
        return;
    }
    const size_t origin_length = std::strlen(origin);
    if (is_error && acquire_to_log(state.errors_flag)) {
        store_error(origin, origin_length, text, length);
        release(state.errors_flag);
    }
    if (!wanted) {
        return;
    }
    if (!acquire_to_log(state.messages_flag)) {
        state.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    store(level, origin, origin_length, text, length);
    release(state.messages_flag);
}

void record_upstream(enum sd_log_level_t level, const char* text, void*) {
    record(level, "", text);
}

// ggml's messages until the first context exists. From then on upstream
// routes them to its own callback, in this form and with these levels.
void record_ggml(enum ggml_log_level level, const char* text, void*) {
    int32_t sd_level = SD_LOG_VERBOSE;
    if (level == GGML_LOG_LEVEL_INFO) {
        sd_level = SD_LOG_INFO;
    } else if (level == GGML_LOG_LEVEL_WARN) {
        sd_level = SD_LOG_WARN;
    } else if (level == GGML_LOG_LEVEL_ERROR) {
        sd_level = SD_LOG_ERROR;
    }
    record(sd_level, "ggml - ", text);
}

// Copies the `length` bytes at `offset`, or as many whole UTF-8 sequences of
// them as fit, and terminates them.
void copy_out(char* text, size_t capacity, size_t offset, size_t length) {
    if (text == nullptr || capacity == 0) {
        return;
    }
    size_t copied = std::min(length, capacity - 1);
    get(offset, text, copied);
    if (copied < length) {
        char next;
        get(offset + copied, &next, 1);
        copied = whole_sequences(text, copied, next);
    }
    text[copied] = '\0';
}

}  // namespace

void sd_dart_log_call_begin() {
    window_begin = state.errors_recorded.load();
    window_end   = window_begin;
}

void sd_dart_log_call_end() {
    window_end = state.errors_recorded.load();
}

void sd_dart_log_enable(void) {
    // Upstream and ggml keep their callbacks in unsynchronized globals, so
    // they are written once rather than on every call.
    int expected = kUnregistered;
    if (state.registration.compare_exchange_strong(expected, kRegistering)) {
        ggml_log_set(record_ggml, nullptr);
        sd_set_log_callback(record_upstream, nullptr);
        state.registration.store(kRegistered);
        return;
    }
    while (state.registration.load() != kRegistered) {
        std::this_thread::yield();
    }
}

void sd_dart_log_set_level(int32_t level) {
    state.level_above_info.store(std::clamp<int32_t>(level, SD_LOG_DEBUG, SD_LOG_ERROR + 1) - SD_LOG_INFO);
}

uint64_t sd_dart_log_read(uint64_t after, char* text, size_t capacity, int32_t* level, size_t* length) {
    if (!acquire(state.messages_flag)) {
        return 0;
    }
    if (after >= state.newest) {
        release(state.messages_flag);
        return 0;
    }
    const uint64_t oldest = state.newest - state.count + 1;
    const uint64_t wanted = std::max(after + 1, oldest);
    uint64_t sequence     = oldest;
    size_t offset         = state.head;
    if (state.cursor_sequence >= oldest && state.cursor_sequence <= wanted) {
        sequence = state.cursor_sequence;
        offset   = state.cursor;
    }
    Header header;
    for (;;) {
        get(offset, &header, sizeof(header));
        if (sequence == wanted) {
            break;
        }
        offset += sizeof(header) + header.length;
        ++sequence;
    }
    copy_out(text, capacity, offset + sizeof(header), header.length);
    if (level != nullptr) {
        *level = header.level;
    }
    if (length != nullptr) {
        *length = header.length;
    }
    state.cursor_sequence = wanted + 1;
    state.cursor          = (offset + sizeof(header) + header.length) & (kBufferBytes - 1);
    state.returned        = std::max(state.returned, wanted);
    release(state.messages_flag);
    return wanted;
}

uint64_t sd_dart_log_dropped(void) {
    return state.dropped.load();
}

size_t sd_dart_last_error(char* text, size_t capacity) {
    const size_t limit = text != nullptr && capacity > 0 ? capacity - 1 : 0;
    size_t length      = 0;
    // The first byte that did not fit.
    char next = '\0';
    if (window_end > window_begin && acquire(state.errors_flag)) {
        const uint64_t recorded = state.errors_recorded.load();
        const uint64_t kept     = recorded > kErrors ? recorded - kErrors : 0;
        for (uint64_t number = std::max(window_begin, kept) + 1; number <= window_end; ++number) {
            const Error& error = state.errors[number % kErrors];
            // One message per line.
            for (size_t i = 0; i < error.length + (number < window_end ? 1u : 0u); ++i, ++length) {
                const char byte = i < error.length ? error.text[i] : '\n';
                if (length < limit) {
                    text[length] = byte;
                } else if (length == limit) {
                    next = byte;
                }
            }
        }
        release(state.errors_flag);
    }
    if (text != nullptr && capacity > 0) {
        text[length <= limit ? length : whole_sequences(text, limit, next)] = '\0';
    }
    return length;
}
