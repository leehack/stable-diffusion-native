// Exercises src/sd_dart_log.cpp against stand-ins for upstream's and ggml's
// log callbacks, so it needs no model and runs under sanitizers.
//
// With the argument `exit` it only checks that logging and reading stay safe
// while exit() runs.

#include "sd_dart_internal.h"
#include "sd_dart_wrapper.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

// The bytes the header says the library keeps.
const size_t kBufferBytes = 256 * 1024;
const size_t kHeaderBytes = 8;
const size_t kErrorsKept  = 32;
const size_t kErrorBytes  = 511;

std::atomic<sd_log_cb_t> upstream_callback{nullptr};
std::atomic<ggml_log_callback> ggml_callback{nullptr};
std::atomic<int> upstream_registrations{0};
std::atomic<int> ggml_registrations{0};

// What upstream does for each message: the text ends in a line break and is
// gone when the callback returns.
void log(enum sd_log_level_t level, const std::string& text) {
    std::string passed = text + "\n";
    upstream_callback.load()(level, passed.c_str(), nullptr);
    std::fill(passed.begin(), passed.end(), '#');
}

struct Message {
    uint64_t sequence = 0;
    int32_t level     = -1;
    std::string text;
};

// One read into a buffer that starts out as garbage. A sequence of 0 means
// there was no message.
Message read(uint64_t after, size_t capacity = SD_DART_LOG_TEXT_SIZE) {
    std::vector<char> text(capacity + 1, '\x7f');
    Message message;
    size_t length    = ~static_cast<size_t>(0);
    message.sequence = sd_dart_log_read(after, text.data(), capacity, &message.level, &length);
    if (message.sequence == 0) {
        CHECK(length == ~static_cast<size_t>(0) && message.level == -1);
        CHECK(capacity == 0 || text[0] == '\x7f');
        return message;
    }
    CHECK(message.sequence > after);
    CHECK(length < SD_DART_LOG_TEXT_SIZE);
    // Nothing is written past the capacity, and the text is terminated.
    CHECK(text[capacity] == '\x7f');
    if (capacity > 0) {
        const size_t copied = length < capacity - 1 ? length : capacity - 1;
        CHECK(std::strlen(text.data()) == copied);
        message.text.assign(text.data(), copied);
    }
    if (capacity > length) {
        CHECK(message.text.size() == length);
    }
    return message;
}

std::vector<Message> read_all(uint64_t after) {
    std::vector<Message> messages;
    for (Message message = read(after); message.sequence != 0; message = read(after)) {
        after = message.sequence;
        messages.push_back(message);
    }
    return messages;
}

uint64_t newest_sequence() {
    const std::vector<Message> messages = read_all(0);
    return messages.empty() ? 0 : messages.back().sequence;
}

std::string last_error(size_t capacity = 1 << 16) {
    std::vector<char> text(capacity + 1, '\x7f');
    const size_t length = sd_dart_last_error(text.data(), capacity);
    CHECK(text[capacity] == '\x7f');
    if (capacity == 0) {
        return std::string(length, '?');
    }
    CHECK(std::strlen(text.data()) == (length < capacity - 1 ? length : capacity - 1));
    CHECK(sd_dart_last_error(nullptr, 0) == length);
    return std::string(text.data());
}

// A text that follows from `n`, of a length that varies, so that a reader can
// tell a whole message from a mix of two and from a shifted copy.
std::string numbered(uint64_t n, size_t bytes) {
    std::string text = "message " + std::to_string(n) + " ";
    while (text.size() < bytes) {
        text += static_cast<char>('a' + (n + text.size()) % 26);
    }
    return text;
}

void test_before_enable() {
    CHECK(read(0).sequence == 0);
    CHECK(sd_dart_log_read(0, nullptr, 0, nullptr, nullptr) == 0);
    CHECK(sd_dart_log_dropped() == 0);
    CHECK(last_error().empty());
    sd_dart_log_call_begin();
    sd_dart_log_call_end();
    CHECK(last_error().empty());
    CHECK(upstream_registrations.load() == 0 && ggml_registrations.load() == 0);
}

void test_enable_registers_once() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++) {
        threads.emplace_back([] {
            sd_dart_log_enable();
            // Registered when the call returns, whichever thread did it.
            CHECK(upstream_callback.load() != nullptr && ggml_callback.load() != nullptr);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    sd_dart_log_enable();
    CHECK(upstream_registrations.load() == 1 && ggml_registrations.load() == 1);
}

void test_messages_keep_order_level_and_text() {
    CHECK(read(0).sequence == 0);
    const sd_log_level_t levels[] = {SD_LOG_INFO, SD_LOG_WARN, SD_LOG_ERROR};
    std::thread([&] {
        for (uint64_t n = 1; n <= 300; n++) {
            log(levels[n % 3], numbered(n, 20 + n % 200));
        }
    }).join();
    const std::vector<Message> messages = read_all(0);
    CHECK(messages.size() == 300);
    for (uint64_t n = 1; n <= 300; n++) {
        const Message& message = messages[n - 1];
        CHECK(message.sequence == n);
        CHECK(message.level == levels[n % 3]);
        CHECK(message.text == numbered(n, 20 + n % 200));
    }
    // Reading removes nothing, and any message can be asked for.
    CHECK(read(0).sequence == 1 && read(0).text == numbered(1, 21));
    CHECK(read(150).sequence == 151 && read(299).sequence == 300);
    CHECK(read(300).sequence == 0 && read(~0ull).sequence == 0);

    // A buffer that is too small gets the start of the text.
    const std::string whole = numbered(7, 27);
    CHECK(read(6, 5).text == whole.substr(0, 4));
    CHECK(read(6, 1).text.empty());
    CHECK(read(6, 0).sequence == 7);
    size_t length = 0;
    int32_t level = -1;
    CHECK(sd_dart_log_read(6, nullptr, 0, &level, &length) == 7);
    CHECK(length == whole.size() && level == levels[7 % 3]);
    CHECK(sd_dart_log_read(6, nullptr, 0, nullptr, nullptr) == 7);
}

void test_line_breaks_and_empty_messages() {
    const uint64_t before = newest_sequence();
    sd_log_cb_t callback  = upstream_callback.load();
    callback(SD_LOG_INFO, "no line break", nullptr);
    callback(SD_LOG_INFO, "windows\r\n", nullptr);
    callback(SD_LOG_INFO, "two\nlines\n", nullptr);
    // Nothing to record.
    callback(SD_LOG_INFO, "\n", nullptr);
    callback(SD_LOG_INFO, "", nullptr);
    callback(SD_LOG_INFO, nullptr, nullptr);
    const std::vector<Message> messages = read_all(before);
    CHECK(messages.size() == 3);
    CHECK(messages[0].text == "no line break" && messages[1].text == "windows");
    CHECK(messages[2].text == "two\nlines");
}

void test_level() {
    uint64_t before = newest_sequence();
    // The default leaves out what is below SD_LOG_INFO, without a sequence.
    log(SD_LOG_DEBUG, "debug");
    log(SD_LOG_VERBOSE, "verbose");
    log(SD_LOG_INFO, "info");
    std::vector<Message> messages = read_all(before);
    CHECK(messages.size() == 1 && messages[0].sequence == before + 1 && messages[0].text == "info");

    sd_dart_log_set_level(SD_LOG_DEBUG);
    log(SD_LOG_DEBUG, "debug");
    sd_dart_log_set_level(-100);
    log(SD_LOG_VERBOSE, "verbose");
    messages = read_all(before + 1);
    CHECK(messages.size() == 2 && messages[0].level == SD_LOG_DEBUG && messages[1].level == SD_LOG_VERBOSE);

    sd_dart_log_set_level(SD_LOG_ERROR);
    log(SD_LOG_WARN, "warning");
    log(SD_LOG_ERROR, "error");
    messages = read_all(before + 3);
    CHECK(messages.size() == 1 && messages[0].text == "error");

    // Above every level nothing is recorded, but errors are still kept.
    for (int32_t level : {static_cast<int32_t>(SD_LOG_ERROR) + 1, 100}) {
        sd_dart_log_set_level(level);
        sd_dart_log_call_begin();
        log(SD_LOG_ERROR, "an unrecorded error");
        sd_dart_log_call_end();
        CHECK(read(before + 4).sequence == 0);
        CHECK(last_error() == "an unrecorded error");
    }
    sd_dart_log_set_level(SD_LOG_INFO);
}

void test_ggml_messages() {
    const uint64_t before      = newest_sequence();
    ggml_log_callback callback = ggml_callback.load();
    callback(GGML_LOG_LEVEL_INFO, "ggml_metal_device_init: GPU name: MTL0\n", nullptr);
    callback(GGML_LOG_LEVEL_WARN, "a warning\n", nullptr);
    callback(GGML_LOG_LEVEL_ERROR, "an error\n", nullptr);
    // Below SD_LOG_INFO, as upstream maps them.
    callback(GGML_LOG_LEVEL_DEBUG, "debug\n", nullptr);
    callback(GGML_LOG_LEVEL_CONT, ".", nullptr);
    callback(GGML_LOG_LEVEL_NONE, "none\n", nullptr);
    const std::vector<Message> messages = read_all(before);
    CHECK(messages.size() == 3);
    CHECK(messages[0].level == SD_LOG_INFO && messages[0].text == "ggml - ggml_metal_device_init: GPU name: MTL0");
    CHECK(messages[1].level == SD_LOG_WARN && messages[1].text == "ggml - a warning");
    CHECK(messages[2].level == SD_LOG_ERROR && messages[2].text == "ggml - an error");
}

void test_long_messages_are_cut() {
    const uint64_t before = newest_sequence();
    const size_t longest  = SD_DART_LOG_TEXT_SIZE - 1;
    log(SD_LOG_INFO, std::string(longest, 'x'));
    log(SD_LOG_INFO, std::string(longest + 1, 'y'));
    log(SD_LOG_INFO, std::string(100000, 'z'));
    // A three-byte sequence that the limit would split is left out whole.
    for (size_t lead = longest - 3; lead <= longest; lead++) {
        log(SD_LOG_INFO, std::string(lead, 'u') + "\xE2\x82\xAC" + "tail");
    }
    upstream_callback.load()(SD_LOG_INFO, (std::string(longest, 'n') + "\n").c_str(), nullptr);
    const std::vector<Message> messages = read_all(before);
    CHECK(messages.size() == 8);
    CHECK(messages[0].text == std::string(longest, 'x'));
    CHECK(messages[1].text == std::string(longest, 'y'));
    CHECK(messages[2].text == std::string(longest, 'z'));
    CHECK(messages[3].text == std::string(longest - 3, 'u') + "\xE2\x82\xAC");
    CHECK(messages[4].text == std::string(longest - 2, 'u'));
    CHECK(messages[5].text == std::string(longest - 1, 'u'));
    CHECK(messages[6].text == std::string(longest, 'u'));
    CHECK(messages[7].text == std::string(longest, 'n'));

    ggml_callback.load()(GGML_LOG_LEVEL_INFO, std::string(100000, 'g').c_str(), nullptr);
    CHECK(read(messages[7].sequence).text == "ggml - " + std::string(longest - 7, 'g'));
}

// More than the buffer holds: the oldest messages go, whole, and are counted.
void test_overflow_drops_the_oldest_and_counts_them() {
    const uint64_t before  = newest_sequence();
    const uint64_t dropped = sd_dart_log_dropped();
    // Everything so far has been read, so nothing counts as lost yet.
    const uint64_t written = 3000;
    size_t bytes           = 0;
    std::thread([&] {
        for (uint64_t n = 1; n <= written; n++) {
            const std::string text = numbered(n, 50 + (n * 37) % 400);
            bytes += kHeaderBytes + text.size();
            log(SD_LOG_INFO, text);
        }
    }).join();
    CHECK(bytes > 2 * kBufferBytes);

    const std::vector<Message> messages = read_all(before);
    CHECK(!messages.empty());
    const uint64_t first = messages.front().sequence;
    CHECK(first > before + 1);
    CHECK(messages.back().sequence == before + written);
    size_t kept = 0;
    for (size_t i = 0; i < messages.size(); i++) {
        const uint64_t n = messages[i].sequence - before;
        CHECK(messages[i].sequence == first + i);
        CHECK(messages[i].text == numbered(n, 50 + (n * 37) % 400));
        kept += kHeaderBytes + messages[i].text.size();
    }
    // The buffer was full to within one message.
    CHECK(kept <= kBufferBytes && kept > kBufferBytes - 500);
    CHECK(sd_dart_log_dropped() - dropped == first - before - 1);

    // A reader sees how far it fell behind, wherever it asks from.
    CHECK(read(0).sequence == first && read(before + 5).sequence == first);
    // What a read has returned is not lost when it goes.
    const uint64_t counted = sd_dart_log_dropped();
    log(SD_LOG_INFO, std::string(4000, 'k'));
    CHECK(read(0).sequence > first);
    CHECK(sd_dart_log_dropped() == counted);
}

// Threads that log at once, and one that reads while they do. Every message
// is whole, each thread's stay in its order, and a message is either read or
// counted as dropped.
void test_concurrent_logging_and_reading() {
    const uint64_t before  = newest_sequence();
    const uint64_t dropped = sd_dart_log_dropped();
    const int kThreads     = 4;
    const uint64_t kEach   = 20000;
    std::atomic<int> running{kThreads};
    std::vector<std::thread> threads;
    for (int thread = 0; thread < kThreads; thread++) {
        threads.emplace_back([thread, kEach, &running] {
            for (uint64_t n = 1; n <= kEach; n++) {
                log(static_cast<sd_log_level_t>(SD_LOG_INFO + thread % 3), std::to_string(thread) + " " + numbered(n, 20 + n % 60));
            }
            running.fetch_sub(1);
        });
    }
    uint64_t after = before;
    uint64_t read_count = 0;
    uint64_t next[kThreads];
    std::fill(next, next + kThreads, 1);
    bool done = false;
    while (!done) {
        // The last pass begins after the last message was logged.
        done = running.load() == 0;
        for (Message message = read(after); message.sequence != 0; message = read(after)) {
            after = message.sequence;
            read_count++;
            const int thread = message.text[0] - '0';
            CHECK(thread >= 0 && thread < kThreads && message.text[1] == ' ');
            CHECK(message.level == SD_LOG_INFO + thread % 3);
            const uint64_t n = std::strtoull(message.text.c_str() + 10, nullptr, 10);
            CHECK(n >= next[thread] && n <= kEach);
            CHECK(message.text.substr(2) == numbered(n, 20 + n % 60));
            next[thread] = n + 1;
        }
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    CHECK(after == before + kThreads * kEach);
    CHECK(read_count + (sd_dart_log_dropped() - dropped) == kThreads * kEach);
    CHECK(read_count > 0);
}

// A reader that keeps up misses nothing while messages go around the buffer
// several times.
void test_a_reader_that_keeps_up_loses_nothing() {
    const uint64_t before  = newest_sequence();
    const uint64_t dropped = sd_dart_log_dropped();
    const uint64_t written = 30000;
    std::atomic<uint64_t> read_up_to{before};
    std::thread writer([&] {
        for (uint64_t n = 1; n <= written; n++) {
            // At most 500 messages of 68 bytes ahead of the reader.
            while (before + n > read_up_to.load() + 500) {
                std::this_thread::yield();
            }
            log(SD_LOG_WARN, numbered(n, 60));
        }
    });
    for (uint64_t after = before; after < before + written;) {
        const Message message = read(after);
        if (message.sequence == 0) {
            std::this_thread::yield();
            continue;
        }
        CHECK(message.sequence == after + 1);
        CHECK(message.level == SD_LOG_WARN && message.text == numbered(message.sequence - before, 60));
        after = message.sequence;
        read_up_to.store(after);
    }
    writer.join();
    CHECK(written * 68 > 5 * kBufferBytes);
    CHECK(sd_dart_log_dropped() == dropped);
}

void test_last_error() {
    // The errors another thread logs during the call, which is where the
    // model loader's workers log theirs.
    log(SD_LOG_ERROR, "before the call");
    sd_dart_log_call_begin();
    std::thread([] {
        log(SD_LOG_ERROR, "model_loader.cpp:1 - tensor 'a' not in model metadata");
        log(SD_LOG_WARN, "a warning");
        log(SD_LOG_INFO, "progress");
    }).join();
    log(SD_LOG_ERROR, "diffusion_engine.cpp:2 - load failed");
    ggml_callback.load()(GGML_LOG_LEVEL_ERROR, "out of memory\n", nullptr);
    sd_dart_log_call_end();
    log(SD_LOG_ERROR, "after the call");
    const std::string expected =
        "model_loader.cpp:1 - tensor 'a' not in model metadata\n"
        "diffusion_engine.cpp:2 - load failed\n"
        "ggml - out of memory";
    CHECK(last_error() == expected);
    // It stays until the thread's next call, and is the thread's own.
    CHECK(last_error() == expected);
    std::thread([] { CHECK(last_error().empty()); }).join();

    // A buffer that is too small gets the start, and the whole length back.
    CHECK(sd_dart_last_error(nullptr, 0) == expected.size());
    CHECK(last_error(11) == expected.substr(0, 10));
    CHECK(last_error(1).empty());
    char one = 'x';
    CHECK(sd_dart_last_error(&one, 1) == expected.size() && one == '\0');

    // A call that logs no error has none, whatever the one before it logged.
    sd_dart_log_call_begin();
    log(SD_LOG_WARN, "only a warning");
    sd_dart_log_call_end();
    CHECK(last_error().empty());
    char cleared[4] = {'x', 'x', 'x', 'x'};
    CHECK(sd_dart_last_error(cleared, sizeof(cleared)) == 0 && cleared[0] == '\0');

    // Of more errors than are kept, the most recent ones remain.
    sd_dart_log_call_begin();
    for (size_t n = 1; n <= kErrorsKept + 8; n++) {
        log(SD_LOG_ERROR, "error " + std::to_string(n));
    }
    sd_dart_log_call_end();
    std::string recent;
    for (size_t n = 9; n <= kErrorsKept + 8; n++) {
        recent += (n == 9 ? "error " : "\nerror ") + std::to_string(n);
    }
    CHECK(last_error() == recent);
    // Later errors replace them, one by one.
    log(SD_LOG_ERROR, "later");
    CHECK(last_error() == recent.substr(recent.find('\n') + 1));
    for (size_t n = 0; n < kErrorsKept; n++) {
        log(SD_LOG_ERROR, "later");
    }
    CHECK(last_error().empty());

    // A long error is cut, between two UTF-8 sequences.
    sd_dart_log_call_begin();
    log(SD_LOG_ERROR, std::string(kErrorBytes - 1, 'e') + "\xC3\xA9" + "tail");
    log(SD_LOG_ERROR, std::string(2000, 'f'));
    sd_dart_log_call_end();
    CHECK(last_error() == std::string(kErrorBytes - 1, 'e') + "\n" + std::string(kErrorBytes, 'f'));

    // The call in progress has none yet.
    sd_dart_log_call_begin();
    log(SD_LOG_ERROR, "during");
    CHECK(last_error().empty());
    sd_dart_log_call_end();
    CHECK(last_error() == "during");
}

// The errors of a call are read while other threads log more of them: each
// one read is whole, and they are in the order they were logged.
void test_errors_are_read_whole_while_others_log() {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> logged{0};
    std::vector<std::thread> threads;
    for (int thread = 0; thread < 2; thread++) {
        threads.emplace_back([thread, &stop, &logged] {
            for (uint64_t n = 1; !stop.load(); n++) {
                log(SD_LOG_ERROR, std::to_string(thread) + " " + numbered(n, 30 + n % 400));
                logged.fetch_add(1);
                if (n % 16 == 0) {
                    std::this_thread::yield();
                }
            }
        });
    }
    uint64_t lines = 0;
    for (int call = 0; call < 2000; call++) {
        const uint64_t before = logged.load();
        sd_dart_log_call_begin();
        while (logged.load() < before + 3) {
            std::this_thread::yield();
        }
        sd_dart_log_call_end();
        // One call: later errors replace the ones of this window meanwhile.
        std::vector<char> text(1 << 15);
        const size_t length = sd_dart_last_error(text.data(), text.size());
        CHECK(length < text.size() && std::strlen(text.data()) == length);
        const std::string errors = text.data();
        uint64_t last[2]         = {0, 0};
        for (size_t start = 0; start < errors.size();) {
            const size_t end       = std::min(errors.find('\n', start), errors.size());
            const std::string line = errors.substr(start, end - start);
            const int thread       = line[0] - '0';
            CHECK(line.size() > 10 && (thread == 0 || thread == 1) && line[1] == ' ');
            const uint64_t n = std::strtoull(line.c_str() + 10, nullptr, 10);
            CHECK(n > last[thread] && line.substr(2) == numbered(n, 30 + n % 400));
            last[thread] = n;
            lines++;
            start = end + 1;
        }
    }
    stop.store(true);
    for (std::thread& thread : threads) {
        thread.join();
    }
    CHECK(lines > 0);
}

// Threads that log and read are still running when main returns: neither may
// touch anything that exit() destroys.
int test_exit() {
    sd_dart_log_enable();
    for (int i = 0; i < 4; i++) {
        std::thread([i] {
            for (uint64_t n = 0;; n++) {
                log(n % 7 == 0 ? SD_LOG_ERROR : SD_LOG_INFO, numbered(n, 30 + (n + i) % 300));
            }
        }).detach();
    }
    for (int i = 0; i < 2; i++) {
        std::thread([] {
            uint64_t after = 0;
            for (;;) {
                const Message message = read(after);
                after                 = message.sequence != 0 ? message.sequence : after;
                // Errors are logged meanwhile, so what the window holds
                // changes from one call to the next.
                char errors[64];
                sd_dart_log_call_begin();
                sd_dart_log_call_end();
                CHECK(sd_dart_last_error(errors, sizeof(errors)) >= std::strlen(errors));
                sd_dart_log_dropped();
            }
        }).detach();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(read(0).sequence != 0);
    return 0;
}

}  // namespace

// The stand-ins for upstream and ggml.

void sd_set_log_callback(sd_log_cb_t callback, void* data) {
    CHECK(callback != nullptr && data == nullptr);
    upstream_registrations.fetch_add(1);
    upstream_callback.store(callback);
}

void ggml_log_set(ggml_log_callback callback, void* data) {
    CHECK(callback != nullptr && data == nullptr);
    ggml_registrations.fetch_add(1);
    ggml_callback.store(callback);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "exit") == 0) {
        return test_exit();
    }
    test_before_enable();
    test_enable_registers_once();
    test_messages_keep_order_level_and_text();
    test_line_breaks_and_empty_messages();
    test_level();
    test_ggml_messages();
    test_long_messages_are_cut();
    test_overflow_drops_the_oldest_and_counts_them();
    test_concurrent_logging_and_reading();
    test_a_reader_that_keeps_up_loses_nothing();
    test_last_error();
    test_errors_are_read_whole_while_others_log();
    return 0;
}
