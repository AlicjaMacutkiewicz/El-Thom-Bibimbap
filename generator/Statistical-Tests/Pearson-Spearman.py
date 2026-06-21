import numpy as np
from _common import FEATURE_COLUMNS, load_representatives, save_heatmap, standardize, test_parser
from scipy import stats


def main() -> None:
    args = test_parser(__doc__).parse_args()
    paths, flights = load_representatives(args)
    flights = [standardize(flight) for flight in flights]
    matrix = np.eye(len(flights))
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            pearson = [stats.pearsonr(flights[i][:, k], flights[j][:, k]).statistic
                       for k in range(len(FEATURE_COLUMNS))]
            spearman = [stats.spearmanr(flights[i][:, k], flights[j][:, k]).statistic
                        for k in range(len(FEATURE_COLUMNS))]
            score = np.nanmean(pearson + spearman)
            matrix[i, j] = matrix[j, i] = score
    save_heatmap(matrix, paths, "Mean Pearson/Spearman correlation",
                 args.output_dir / "pearson_spearman.png")


if __name__ == "__main__":
    main()
