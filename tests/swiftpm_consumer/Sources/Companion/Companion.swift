import stable_diffusion

public enum Companion {
    public static func version() -> String { String(cString: sd_version()) }
}
