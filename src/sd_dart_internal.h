#ifndef SD_DART_INTERNAL_H
#define SD_DART_INTERNAL_H

// What the wrapper's sources share with each other. Not shipped and not
// exported.

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

private:
    // False on the teardown thread once teardown has begun.
    bool counted_;
};

// Mark the start and end of a call whose error messages
// sd_dart_last_error() reports to the calling thread.
void sd_dart_log_call_begin();
void sd_dart_log_call_end();

#endif  // SD_DART_INTERNAL_H
