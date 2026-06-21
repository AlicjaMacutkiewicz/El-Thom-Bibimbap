from __future__ import annotations

import numpy as np
import pandas as pd
from _common import FEATURE_COLUMNS, discover_files, test_parser
from matplotlib import pyplot as plt

SCENARIO_COLUMNS = [
    "Scenario_Oxidizer_Fraction",
    "Scenario_Pressure_Scale",
    "Scenario_Rocket_Mass_Scale",
    "Scenario_Drag_Multiplier",
]
LEGACY_COLUMN = "Scenario_Propellant_Fraction"


def main() -> None:
    args = test_parser("Validate full batch integrity and scenario coverage.").parse_args()
    files = discover_files(args.input_dir, args.start_year, args.end_year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    invalid_files = 0
    nonfinite_values = 0

    for index, path in enumerate(files, start=1):
        try:
            frame = pd.read_parquet(path)
            required = FEATURE_COLUMNS + SCENARIO_COLUMNS + [LEGACY_COLUMN]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            nonfinite_values += int(
                (~np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float))).sum()
            )
            metadata = [column for column in frame if column.startswith("Scenario_")]
            if any(frame[column].nunique(dropna=False) != 1 for column in metadata):
                raise ValueError("Scenario metadata changes within the flight")
            if not np.allclose(frame[LEGACY_COLUMN], frame[SCENARIO_COLUMNS[0]]):
                raise ValueError("Legacy propellant fraction does not match oxidizer fraction")
            row = {column: float(frame[column].iloc[0]) for column in SCENARIO_COLUMNS}
            row["max_altitude_m"] = float(frame["Position_Z"].max())
            rows.append(row)
        except Exception as error:
            invalid_files += 1
            print(f"INVALID {path.name}: {error}")
        if index % 100 == 0:
            print(f"Checked {index}/{len(files)} files")

    results = pd.DataFrame(rows)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, column in zip(axes.flat, SCENARIO_COLUMNS, strict=False):
        axis.scatter(results[column], results["max_altitude_m"], s=9, alpha=0.4)
        rho = results[[column, "max_altitude_m"]].corr(method="spearman").iloc[0, 1]
        axis.set_title(f"Spearman r = {rho:.3f}")
        axis.set_xlabel(column.replace("Scenario_", ""))
        axis.set_ylabel("Maximum altitude [m]")
    figure.tight_layout()
    figure.savefig(args.output_dir / "scenario_vs_max_altitude.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, column in zip(axes.flat, SCENARIO_COLUMNS, strict=False):
        axis.hist(results[column], bins=25, edgecolor="black", alpha=0.75)
        axis.set_title(column.replace("Scenario_", ""))
        axis.set_ylabel("Flights")
    figure.tight_layout()
    figure.savefig(args.output_dir / "scenario_coverage.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(results["max_altitude_m"], bins=40, edgecolor="black", alpha=0.8)
    axis.axvline(results["max_altitude_m"].median(), color="red", linestyle="--")
    axis.set_title("Maximum altitude distribution")
    axis.set_xlabel("Maximum altitude [m]")
    axis.set_ylabel("Flights")
    figure.tight_layout()
    figure.savefig(args.output_dir / "maximum_altitude_distribution.png", dpi=180)
    plt.close(figure)

    print(
        f"Finished: {len(results)}/{len(files)} valid files; "
        f"{invalid_files} invalid; {nonfinite_values} non-finite feature values."
    )


if __name__ == "__main__":
    main()
