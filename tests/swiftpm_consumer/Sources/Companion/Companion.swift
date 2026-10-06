import stable_diffusion

public enum Companion {
    public static func version() -> String { String(cString: sd_version()) }

    public static func progressSequence() -> UInt64 {
        var progress = sd_dart_progress_t(sequence: 1, step: 1, steps: 1, time: 1)
        var latest: UInt64 = 1
        let count = sd_dart_progress_read(0, &progress, 1, &latest)
        return latest + UInt64(count)
    }
}
