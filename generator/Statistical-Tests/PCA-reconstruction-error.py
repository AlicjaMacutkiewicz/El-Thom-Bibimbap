import numpy as np
from _common import load_representatives, save_heatmap, test_parser

WINDOW_SIZE = 50
COMPONENTS = 10


def windows(values: np.ndarray) -> np.ndarray:
    return np.array([
        values[index:index + WINDOW_SIZE].ravel()
        for index in range(len(values) - WINDOW_SIZE + 1)
    ])


def main() -> None:
    args = test_parser(__doc__).parse_args()
    paths, flights = load_representatives(args)
    mean = flights[0].mean(axis=0)
    scale = flights[0].std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = [(flight - mean) / scale for flight in flights]
    reference = windows(scaled[0])
    center = reference.mean(axis=0)
    _, _, basis = np.linalg.svd(reference - center, full_matrices=False)
    components = basis[:COMPONENTS]
    errors = []
    for flight in scaled:
        flight_windows = windows(flight)
        centered = flight_windows - center
        reconstruction = centered @ components.T @ components + center
        errors.append(np.percentile(np.mean((flight_windows - reconstruction) ** 2, axis=1), 95))
    matrix = np.log1p(np.abs(np.subtract.outer(errors, errors)))
    save_heatmap(matrix, paths, "PCA reconstruction error difference (log1p P95)",
                 args.output_dir / "pca_reconstruction_error.png", ".3f")


if __name__ == "__main__":
    main()
