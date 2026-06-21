import numpy as np
from _common import load_representatives, save_heatmap, standardize, test_parser
from scipy.spatial.distance import cdist


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    costs = cdist(left, right, metric="cityblock") / left.shape[1]
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for row in costs:
        current = np.full(len(right) + 1, np.inf)
        for column, cost in enumerate(row, start=1):
            current[column] = cost + min(previous[column], current[column - 1], previous[column - 1])
        previous = current
    return previous[-1] / (len(left) + len(right))


def main() -> None:
    args = test_parser(__doc__, samples=128).parse_args()
    paths, flights = load_representatives(args)
    flights = [standardize(flight) for flight in flights]
    matrix = np.zeros((len(flights), len(flights)))
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            matrix[i, j] = matrix[j, i] = dtw_distance(flights[i], flights[j])
    save_heatmap(matrix, paths, "Multivariate DTW distance",
                 args.output_dir / "dynamic_time_warping.png", ".3f")


if __name__ == "__main__":
    main()
