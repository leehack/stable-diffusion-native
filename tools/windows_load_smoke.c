#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <wchar.h>

static const wchar_t *const required_runtime[] = {
    L"msvcp140.dll", L"msvcp140_codecvt_ids.dll", L"vcruntime140.dll", L"vcruntime140_1.dll",
};

static int is_runtime_module(const wchar_t *name) {
    static const wchar_t *const prefixes[] = {
        L"msvcp140", L"vcruntime140", L"concrt140", L"vccorlib140", L"vcomp140", L"libomp140",
    };
    for (size_t i = 0; i < ARRAYSIZE(prefixes); ++i) {
        if (_wcsnicmp(name, prefixes[i], wcslen(prefixes[i])) == 0) {
            return 1;
        }
    }
    return 0;
}

static int check_runtime_modules(const wchar_t *bundle_dir) {
    HMODULE modules[1024];
    DWORD needed = 0;
    if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed)
            || needed > sizeof(modules)) {
        fwprintf(stderr, L"Cannot enumerate all process modules.\n");
        return 0;
    }
    int seen[ARRAYSIZE(required_runtime)] = {0};
    for (size_t i = 0; i < needed / sizeof(HMODULE); ++i) {
        wchar_t path[MAX_PATH];
        DWORD length = GetModuleFileNameW(modules[i], path, MAX_PATH);
        if (length == 0 || length >= MAX_PATH) {
            fwprintf(stderr, L"Cannot resolve a loaded module path.\n");
            return 0;
        }
        const wchar_t *separator = wcsrchr(path, L'\\');
        const wchar_t *name = separator ? separator + 1 : path;
        if (!is_runtime_module(name)) {
            continue;
        }
        if (!bundle_dir || !separator || wcslen(bundle_dir) != (size_t)(separator - path)
                || _wcsnicmp(path, bundle_dir, (size_t)(separator - path)) != 0) {
            fwprintf(stderr, L"Runtime DLL loaded outside the bundle: %ls\n", path);
            return 0;
        }
        wprintf(L"bundled: %ls\n", path);
        for (size_t j = 0; j < ARRAYSIZE(required_runtime); ++j) {
            if (_wcsicmp(name, required_runtime[j]) == 0) {
                seen[j] = 1;
            }
        }
    }
    if (bundle_dir) {
        for (size_t i = 0; i < ARRAYSIZE(required_runtime); ++i) {
            if (!seen[i]) {
                fwprintf(stderr, L"%ls was not loaded from the bundle.\n", required_runtime[i]);
                return 0;
            }
        }
    }
    return 1;
}

static int probe_string(HMODULE module, const char *name) {
    typedef const char *(__cdecl *GetString)(void);
    FARPROC address = GetProcAddress(module, name);
    if (!address) {
        fwprintf(stderr, L"Missing export %hs.\n", name);
        return 0;
    }
    const char *value = ((GetString)address)();
    if (!value || !*value) {
        fwprintf(stderr, L"%hs returned no value.\n", name);
        return 0;
    }
    wprintf(L"%hs: %hs\n", name, value);
    return 1;
}

int wmain(int argc, wchar_t **argv) {
    if (argc != 3) {
        fwprintf(stderr, L"Usage: %ls <bundle-dir> <dll>\n", argv[0]);
        return 2;
    }
    wchar_t bundle_dir[MAX_PATH];
    DWORD length = GetFullPathNameW(argv[1], MAX_PATH, bundle_dir, NULL);
    if (length == 0 || length >= MAX_PATH) {
        fwprintf(stderr, L"Cannot resolve %ls.\n", argv[1]);
        return 2;
    }
    if (bundle_dir[length - 1] == L'\\') {
        bundle_dir[length - 1] = L'\0';
    }

    // /MT keeps the host from preloading the runtime and masking missing DLLs.
    if (!check_runtime_modules(NULL)) {
        return 1;
    }
    wchar_t path[MAX_PATH];
    if (swprintf(path, MAX_PATH, L"%ls\\%ls", bundle_dir, argv[2]) < 0) {
        return 2;
    }
    HMODULE module = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        fwprintf(stderr, L"LoadLibraryExW(%ls) failed: error %lu\n", path, GetLastError());
        return 1;
    }
    return probe_string(module, "sd_version") && probe_string(module, "sd_commit")
        && check_runtime_modules(bundle_dir) ? 0 : 1;
}
