#ifndef SD_DART_INTERNAL_H
#define SD_DART_INTERNAL_H

// What the wrapper's sources share with each other. Not shipped and not
// exported.

// Where C exit() keeps the statics of the library alive and frees nothing.
// Android is left out: an exit during a call in flight has not been examined
// there.
#if defined(__linux__) && !defined(__ANDROID__)
#define SD_DART_EXIT_ON_LINUX 1
#else
#define SD_DART_EXIT_ON_LINUX 0
#endif

// A call that uses the statics of the library without a tracked object, as a
// query of ggml's device registry does. Exit teardown waits for it whatever
// is tracked, with the bound of a load, and it blocks once teardown has
// begun.
class SdDartStaticsCall {
public:
    SdDartStaticsCall();
    ~SdDartStaticsCall();
    SdDartStaticsCall(const SdDartStaticsCall&)            = delete;
    SdDartStaticsCall& operator=(const SdDartStaticsCall&) = delete;

#if SD_DART_EXIT_ON_LINUX
    // Whether exit() has begun, and the call must not begin: the exit
    // handlers of the driver it would ask may have run.
    bool refused() const { return refused_; }
#endif

private:
    // False on the teardown thread once teardown has begun.
    bool counted_;
#if SD_DART_EXIT_ON_LINUX
    bool refused_ = false;
#endif
};

// Mark the start and end of a call whose error messages
// sd_dart_last_error() reports to the calling thread.
void sd_dart_log_call_begin();
void sd_dart_log_call_end();

#endif  // SD_DART_INTERNAL_H
