import numpy as np
from _common import FEATURE_COLUMNS, load_representatives, save_heatmap, test_parser
from scipy import stats


def main() -> None:
    args = test_parser(__doc__).parse_args()
    paths, flights = load_representatives(args)
    matrix = np.zeros((len(flights), len(flights)))
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            values = [stats.mannwhitneyu(flights[i][:, k], flights[j][:, k]).pvalue
                      for k in range(len(FEATURE_COLUMNS))]
            score = -np.log10(max(np.median(values), 1e-300))
            matrix[i, j] = matrix[j, i] = score
    save_heatmap(matrix, paths, "Mann-Whitney significance (-log10 median p)",
                 args.output_dir / "mann_whitney_u.png")


if __name__ == "__main__":
    main()
