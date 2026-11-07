// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "OBSNeuralPost",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "NeuralPost", type: .dynamic, targets: ["NeuralPost"]),
        .executable(name: "neural-post-bench", targets: ["NeuralPostBench"]),
        .executable(name: "PS5NeuralAudit", targets: ["PS5NeuralAudit"])
    ],
    dependencies: [.package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3")],
    targets: [
        .target(name: "NeuralPost", dependencies: [.product(name: "MLX", package: "mlx-swift")],
                swiftSettings: [.swiftLanguageMode(.v5)],
                linkerSettings: [.linkedFramework("MetalPerformanceShaders"), .linkedFramework("AppKit")]),
        .executableTarget(name: "NeuralPostBench", dependencies: ["NeuralPost"],
                          swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "PS5NeuralAudit", dependencies: ["NeuralPost"], path: "tools", exclude: ["FullModelANEAudit.swift", "ane_full_model.py", "run_ane_audit.py", "export_stage_island.py", "try-full720-ane.sh", "live_quality", "build-live-quality.sh", "NativeQualityPipeline.swift", "resolution_sweep.py", "tight_144_probe.py", "NativeTeacherBudget.swift", "FrameGenBudget.swift", "quality_first_bench.py", "build-quality-bench.sh", "appearance_student", "CoreMLAudit.swift", "__pycache__", "build.sh", "convert_rt4ksr.py", "make_full_fixture.py", "make_training_fixture.py", "open-ps5-screen.sh", "open-stage-split-screen.sh", "open-three-mode-screen.sh", "train_appearance.py"], sources: ["CaptureAudit.swift"],
                          swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "NeuralPostTests", dependencies: ["NeuralPost"],
                    swiftSettings: [.swiftLanguageMode(.v5)])
    ])
