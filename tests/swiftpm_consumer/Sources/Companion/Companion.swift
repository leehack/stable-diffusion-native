import stable_diffusion

public enum Companion {
    public static func version() -> String { String(cString: sd_version()) }

    public static func progressSequence() -> UInt64 {
        var progress = sd_dart_progress_t(sequence: 1, step: 1, steps: 1, time: 1)
        sd_dart_progress_read(&progress)
        return progress.sequence
    }
}
