import Companion
import Darwin

// RTLD_DEFAULT, as Dart's DynamicLibrary.process() searches.
let process = UnsafeMutableRawPointer(bitPattern: -2)

func symbol(_ name: String) -> UnsafeMutableRawPointer? { dlsym(process, name) }

typealias VersionFn = @convention(c) () -> UnsafePointer<CChar>
typealias ListDevicesFn = @convention(c) (UnsafeMutablePointer<CChar>?, Int) -> Int
typealias ClearProgressFn = @convention(c) (UnsafeMutableRawPointer?) -> Void

guard let versionSymbol = symbol("sd_version"), let listSymbol = symbol("sd_list_devices") else {
    print("error: SD_API symbols are not visible process-wide")
    exit(1)
}
let version = String(cString: unsafeBitCast(versionSymbol, to: VersionFn.self)())
guard version == Companion.version() else {
    print("error: dlsym and module import disagree")
    exit(1)
}
let listDevices = unsafeBitCast(listSymbol, to: ListDevicesFn.self)
var buffer = [CChar](repeating: 0, count: listDevices(nil, 0) + 1)
_ = listDevices(&buffer, buffer.count)
let devices = String(cString: buffer)
print("stable-diffusion.cpp \(version)\n\(devices)")
if !devices.contains("CPU") {
    print("error: no CPU device")
    exit(1)
}
guard symbol("sd_dart_set_progress_callback") != nil,
      let clearSymbol = symbol("sd_dart_clear_progress_callback") else {
    print("error: sd_dart_wrapper.h symbols are not visible process-wide")
    exit(1)
}
unsafeBitCast(clearSymbol, to: ClearProgressFn.self)(nil)
if symbol("ggml_init") != nil {
    print("error: ggml symbols leak into the process")
    exit(1)
}
