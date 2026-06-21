# "Grzesiek" - Generator

A high-performance synthetic flight data generator for suborbital rockets (tested on the R7 Orzeł model), designed to run massive simulations using the Monte Carlo method.

## Tech Stack
* **Engine:** `RocketPy` (physical simulation core).
* **Performance:** 
    * `numba`: JIT compilation for matrix and vector operations (`fastmath=True` enabled for interpolation).
    * `cupy` (CUDA): GPU acceleration for computations, featuring **automatic fallback to `numpy`** if no CUDA-compatible hardware is detected. Multi-GPU distribution is supported.
    * `pathos`: Parallel execution of simulation instances (multiprocessing).
* **Data Source:** ERA5 meteorological database (Copernicus) – dynamic fetching of historical weather data.
* **Storage:** `.parquet` (target columnar format for I/O optimization), support for `.csv` and `.out`.

## Key Features

### 1. Stochastic Motor Modeling
The generator simulates production differences between propulsion units by randomly modifying nominal parameters:
* **Grain geometry:** Propellant density, outer/inner radius, initial height.
* **Nozzle geometry:** Throat radius, exit radius.
* **Energetics:** Total motor impulse.

### 2. Environmental Conditions (ERA5)
* Integration with the ERA5 database for the Gdańsk University of Technology location.
* Generation of 10 unique scenarios for each hour within a selected time range.
* Automatic fetching of weather data on the fly.

### 3. Sensor Failure and Noise Emulation
The system generates data at a frequency of **500Hz** and applies physical error models to it:
* **Sensor Signal Dropout:** Random transmission interruptions. The probability of signal loss is a function of wind speed and g-loads.
* **Bit-switch:** Simulation of bit errors in the digital signal.
* **Sample-and-hold:** A mechanism for maintaining the last value for sensors with a sampling rate < 500Hz.

### 4. Dynamic Auto-Ranging Sensor Selection
The generator simulates multiple hardware measurement ranges simultaneously. It uses a dynamic thresholding algorithm (`get_best_acceleration`, `get_best_angular_velocity`) to automatically select the most precise unclipped sensor range for a given moment in flight:
* **Accelerometers:** Dynamically switches between 2g, 4g, 8g, and 16g data streams.
* **Gyroscopes:** Dynamically switches between 125, 250, 500, 1000, and 2000 dps data streams.

## Supported Sensor Models
The current configuration explicitly models the noise, variance, and bias profiles of the following hardware:
* **IMU:** LSM6DSOX (Acceleration and Angular Velocity)
* **Barometer:** BME280
* **Thermometer:** DS18B20

## Pipeline and Architecture
1. **Input:** `.ork` project (OpenRocket) → Conversion via `RocketSerializer` to `.json`.
2. **Config:** Base settings located in the `/source_model` folder (configured via `paths.json`).
3. **Execution:** Concurrent thread execution using `pathos`. Each thread generates an independent instance of the `StochasticMotor` class.
4. **Logging:** The full process run is saved in `output/logs.txt`.
5. **Output:** `.parquet` files with raw sensor data (High Frequency Data).

## Usage

**Standard execution:**
```bash
python main.py
```

**FAR-OUT 2026 competition-day configuration:**
```bash
python main.py --competition-day 2020 2021
```

**Robustness/domain-randomization configuration:**
```bash
python main.py --paths paths_robustness.json \
  --scenario scenarios/robustness_oxidizer_40_100.json 2020 2021
```

The default `paths.json` file points to the nominal R7 configuration in
`source_model/R7_SIMLE/R7_OUTPUT`. The `--competition-day` flag switches the
generator to `paths_farout_26.json`, which points to the FAR-OUT 2026
configuration in `source_model/R7_SIMLE/R7_FAROUT_26`. This keeps the nominal
configuration and the launch-day approximation separate and traceable.
The FAR-OUT source thrust curve already represents approximately 85% of nominal
pressure. Its paths file records that source condition so the scenario's 0.85
pressure value is retained as metadata without scaling the curve a second time.

Flight-condition variation is controlled by scenario JSON files in
`generator/src/scenarios`. A scenario can sample:
* `oxidizer_fraction`: available oxidizer relative to the nominal load,
* `pressure_scale`: multiplier applied to the selected thrust curve before the
  simulation starts,
* `rocket_mass_scale`: multiplier applied to the configured rocket mass and
  inertia,
* `drag_multiplier`: latent multiplier applied to the RocketPy drag curve.

The oxidizer-aware scenario keeps the nominal paraffin grain at launch. The
fraction expected to burn is represented by the existing RocketPy `SolidMotor`,
while unburned paraffin is retained in the dry rocket mass. Oxidizer availability
scales burn duration and pressure scales thrust magnitude, so the first-order
total-impulse scale is their product. The nominal oxidizer and paraffin masses
are explicit in the scenario and must match the source motor's total propellant
mass.

For the R7 robustness surrogate, the revised motor configuration declares
`12.24 kg` of nominal propellant: `9.87 kg` oxidizer and `2.37 kg` paraffin.
The converted source geometry evaluates to `12.0 kg`; the generator reconciles
this small difference when constructing the conditioned motor. The paraffin
value describes the configured grain, not a direct measurement of consumed
paraffin during a firing. Likewise, the burn-time
exponent of `1.0` is a deliberately simple linear surrogate, not a fitted motor
parameter. Both values are kept explicit so they can be replaced after
calibration against additional firing data.

Scenario values and derived motor masses are written into generated Parquet
files as `Scenario_*` metadata columns. The current model pipeline consumes the
three scenario inputs. `Scenario_Propellant_Fraction` is temporarily emitted as
a compatibility alias for `Scenario_Oxidizer_Fraction`. The legacy `--fuel`
option is likewise retained as an alias for `--oxidizer`.

`Scenario_Drag_Multiplier` is recorded for traceability but is intentionally
excluded from the GRU input schema. The robustness scenario draws 75% of flights
from a near-nominal `0.75--1.5` range and 25% from a high-loss `1.5--6.0` tail.
The upper tail is an effective aerodynamic-loss surrogate that may also absorb
attitude and unmodeled flight losses; it should not be interpreted as a direct
measurement of clean-body drag coefficient.

Robustness generation uses `robustness_oxidizer_40_100.json` together with
`paths_robustness.json`. The separate paths file selects
`R7_ROBUSTNESS/parameters.json`, keeping the nominal and article-era physical
configuration unchanged. The scenario samples oxidizer and pressure from `0.4`
to `1.0`, rocket mass from `0.9` to `1.3`, and latent aerodynamic loss from the
mixture described above.
Scaling is performed in memory; no per-flight thrust CSV files are written.
Files produced by the legacy generic scenario must not be mixed into an
oxidizer-aware training batch. The model loader rejects variable legacy
`Scenario_Propellant_Fraction` data when the canonical
`Scenario_Oxidizer_Fraction` column is absent.

The legacy `--fuel` and `--oxidizer` command-line overrides continue to apply
generic grain-height scaling when no `oxidizer_model` is present. They should
not be used as a substitute for the oxidizer-aware robustness scenario.

Training or evaluating a model on this new batch must use the matching RK4
configuration:

```bash
python main.py ... \
  --parameters ../../../source_model/R7_SIMLE/R7_ROBUSTNESS/parameters.json \
  --thrust-curve ../../../source_model/R7_SIMLE/R7_OUTPUT/thrust_source.csv
```
