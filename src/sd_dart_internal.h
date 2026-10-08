#ifndef SD_DART_INTERNAL_H
#define SD_DART_INTERNAL_H

// What the wrapper's sources share with each other. Not shipped and not
// exported.

// Mark the start and end of a call whose error messages
// sd_dart_last_error() reports to the calling thread.
void sd_dart_log_call_begin();
void sd_dart_log_call_end();

#endif  // SD_DART_INTERNAL_H
