import numpy as np
from _common import FEATURE_COLUMNS, load_representatives, save_heatmap, test_parser
from scipy import stats


def main() -> None:
    args = test_parser(__doc__).parse_args()
    paths, flights = load_representatives(args)
    matrix = np.zeros((len(flights), len(flights)))
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            values = [stats.ks_2samp(flights[i][:, k], flights[j][:, k]).statistic
                      for k in range(len(FEATURE_COLUMNS))]
            matrix[i, j] = matrix[j, i] = np.mean(values)
    save_heatmap(matrix, paths, "KS statistic (mean over features)",
                 args.output_dir / "kolmogorov_smirnov.png", ".3f")


if __name__ == "__main__":
    main()
