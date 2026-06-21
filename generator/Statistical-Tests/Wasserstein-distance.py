import numpy as np
from _common import FEATURE_COLUMNS, load_representatives, save_heatmap, standardize, test_parser
from scipy import stats


def main() -> None:
    args = test_parser(__doc__).parse_args()
    paths, flights = load_representatives(args)
    flights = [standardize(flight) for flight in flights]
    matrix = np.zeros((len(flights), len(flights)))
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            values = [stats.wasserstein_distance(flights[i][:, k], flights[j][:, k])
                      for k in range(len(FEATURE_COLUMNS))]
            matrix[i, j] = matrix[j, i] = np.mean(values)
    save_heatmap(matrix, paths, "Normalized Wasserstein distance",
                 args.output_dir / "wasserstein_distance.png", ".3f")


if __name__ == "__main__":
    main()
