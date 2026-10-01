// swift-tools-version: 5.9
// Links the XCFramework the way the llamadart Flutter companion does: a
// dynamic library re-exports it, and callers resolve SD_API symbols globally.
import PackageDescription

let package = Package(
    name: "swiftpm_consumer",
    platforms: [.iOS("16.4"), .macOS("14.0")],
    products: [
        .library(name: "Companion", type: .dynamic, targets: ["Companion"]),
        .executable(name: "Probe", targets: ["Probe"]),
    ],
    targets: [
        // `apple_xcframework.py consumer` copies the archive under test here.
        .binaryTarget(name: "stable_diffusion", path: "stable_diffusion.zip"),
        .target(
            name: "Companion",
            dependencies: ["stable_diffusion"],
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-reexport_framework", "-Xlinker", "stable_diffusion"]),
            ]
        ),
        .executableTarget(name: "Probe", dependencies: ["Companion"]),
    ]
)
