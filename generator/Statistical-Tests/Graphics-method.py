import numpy as np
from _common import FEATURE_COLUMNS, load_representatives, test_parser
from matplotlib import pyplot as plt


def main() -> None:
    args = test_parser(__doc__).parse_args()
    _, flights = load_representatives(args)
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        figure, axis = plt.subplots(figsize=(10, 6))
        for flight in flights:
            axis.plot(np.linspace(0, 1, len(flight)), flight[:, feature_index], alpha=0.25)
        axis.set_title(feature)
        axis.set_xlabel("Normalized flight time")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"overlay_{feature}.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
