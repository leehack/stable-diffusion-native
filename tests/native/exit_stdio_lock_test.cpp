#define main sd_repository_main
#include "exit_teardown_test.cpp"
#undef main

static std::atomic<bool> locked_stdout{false};
static std::atomic<bool> release_stdout{false};
static std::thread* lock_worker;
static void host_shutdown() {
    release_stdout.store(true);
    lock_worker->join();
    fprintf(stderr, "HOST_HANDLER_COMPLETED\n");
}
int main() {
    // Host shutdown is registered before image runtime calls, as in the
    // project's existing host-handler test. The worker's log stream lock
    // must be released by this handler before normal exit flushes stdio.
    CHECK(at_exit(host_shutdown));
    static char buffer[4096];
    setvbuf(stdout, buffer, _IOFBF, sizeof(buffer));
    fputs("BUFFERED_HOST_OUTPUT", stdout);
    sd_ctx_params_t params{};
    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    CHECK(context != nullptr);
    lock_worker = new std::thread([] {
        sd_dart_exit_call_begin();
        flockfile(stdout);
        locked_stdout.store(true);
        while (!release_stdout.load()) sleep_ms(1);
        funlockfile(stdout);
        sd_dart_exit_call_end();
    });
    while (!locked_stdout.load()) sleep_ms(1);
    write(STDERR_FILENO, "EXIT_REACHED\n", 13);
    exit(37);
}
