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