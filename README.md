`Polish version below!`

# Rocket Flight Prediction System
**Flight trajectory prediction system for suborbital platform telemetry**

This project implements a hybrid trajectory prediction system for suborbital rockets, tested on the **R7 Orzeł** model (KN SimLE, Gdańsk University of Technology). The goal is to maintain short-horizon trajectory estimates during telemetry degradation or loss by combining deterministic numerical integration with recurrent neural networks.

---

## Project Concept
In classical telemetry, losing the signal reduces direct access to the current vehicle state. This system uses recent telemetry to initialize a sequence model and then predicts future acceleration during a simulated cut-off interval. The predicted acceleration is integrated into position so that the trajectory estimate can be maintained until telemetry is restored.

### Core System Components:
* **Synthetic flight generator**: A configurable RocketPy-based pipeline for creating synthetic flight data under nominal and off-nominal launch conditions.
* **Prediction models**: GRU-based sequence models that combine neural forecasting with RK4-based physical baselines and trajectory-consistency losses.

### Model Architecture:
* **Segment B (Baseline)**: A deterministic physical component using a **4th-order Runge-Kutta (RK4)** model for thrust, changing mass, launch direction, and gravity.
* **Segment R (Residual model)**: A **Gated Recurrent Unit (GRU)** recurrent network that predicts either full acceleration or the residual acceleration not captured by the RK4 baseline.
* **Segment G (Persistence gate)**: Optional gated variants blend the learned RK4-residual forecast with a last-acceleration baseline using learned per-axis gate values.
* **Segment I (Integration)**: A trajectory-consistency layer that integrates predicted acceleration and compares the resulting position sequence with the reference trajectory.

---
## Tech Stack
* **Deep Learning & Math**: `torch` (PyTorch), `numpy`, `pandas`.
* **Rocketry & Physics**: `rocketpy`, `xarray` (multi-dimensional arrays).
* **Data Science & API**: `cdsapi` (Copernicus/ERA5 fetching), `pyarrow`, `fastparquet` (columnar data storage).
* **Performance & Parallelization**: `pathos` (multiprocessing), `cupy` (CUDA support).
* **Visualization & Profiling**: `seaborn`, `matplotlib`, `snakeviz` (browser-based profiling).

---

## Key Features

### Synthetic Flight Generator
* **Scenario conditioning**: Generation of flights with configurable oxidizer fraction, pressure/thrust scaling, rocket mass scaling, and latent drag variation.
* **Environmental Conditions**: Integration with ERA5 atmospheric reanalysis data.
* **Sensor Emulation**: Simulation of telemetry channels used by the prediction models, including acceleration, angular velocity, barometric pressure, and temperature.
* **Dynamic Auto-Ranging**: Automatic selection of the most precise measurement range (e.g., dynamically switching between 2g-16g accelerometer streams) during flight.

### Prediction Models
* **Residual RK4-GRU Learning**: The residual variants predict the acceleration correction relative to the deterministic RK4 baseline.
* **Trajectory-Consistency Loss**: Physics-informed variants integrate predicted acceleration and penalize accumulated position drift over the prediction horizon.
* **Persistence-Gated Forecasting**: Gated variants learn to blend the RK4-residual forecast with a last-acceleration baseline for improved robustness under simulation-to-real domain shift.
* **Spin-Up/Cut-Off Operation**: Models use a Spin-Up encoder window with available telemetry and a Cut-Off decoder window that predicts future acceleration without new measurements.

---

## Repository Structure
* `/docs` – Theoretical documentation and project schemes
* `/generator` – Source code for the synthetic data generator
* `/prediction_models` – Training and evaluation code for GRU-based and baseline trajectory predictors
* `/source_data` – Configuration files, `.ork` models, and input data

---

## Paper Configuration

The current paper experiments use conditioned GRU models with:
* 8 telemetry inputs: acceleration, angular velocity, barometric pressure, and temperature
* 3 scenario-conditioning inputs: oxidizer fraction, pressure scale, and rocket mass scale
* an internal Spin-Up/Cut-Off mode flag appended by the model
* a 120-sample historical Spin-Up window and a 60-sample prediction horizon
* synthetic 3D evaluation on held-out flights and real-flight vertical-axis replay on FAR-OUT 2026 telemetry

The real-flight files in `source_data/far_out_26_data` are included with project permission for research and reproducibility purposes.

---
**Project Team**: Alicja Macutkiewicz, Weronika Marszalik, Paweł Leczkowski, Wiktor Ludwichowski, Emilia Łukasiuk

---
---
<details>
<summary><b> Polish Version (click here)</b></summary>

# Rocket Flight Prediction System
**System predykcji toru lotu dla telemetrii platform suborbitalnych**

Projekt realizuje hybrydowy system przewidywania trajektorii rakiet suborbitalnych, testowany na modelu **R7 Orzeł** (KN SimLE PG). Celem jest utrzymanie krótkohoryzontowej estymacji trajektorii w przypadku degradacji lub utraty telemetrii poprzez połączenie deterministycznej integracji numerycznej z rekurencyjnymi sieciami neuronowymi.

---

## Idea projektu
W klasycznej telemetrii utrata sygnału ogranicza bezpośredni dostęp do aktualnego stanu pojazdu. System wykorzystuje ostatnie dostępne próbki telemetrii do zainicjalizowania modelu sekwencyjnego, a następnie przewiduje przyszłe przyspieszenie w symulowanym okresie odcięcia sygnału. Przewidywane przyspieszenie jest całkowane do pozycji, aby utrzymać estymację trajektorii do momentu odzyskania telemetrii.

### Główne komponenty systemu:
* **Generator lotów syntetycznych**: Konfigurowalny generator oparty o RocketPy, służący do tworzenia danych lotu w nominalnych i odchylonych warunkach startowych.
* **Modele predykcyjne**: Modele sekwencyjne GRU łączące predykcję neuronową, bazę fizyczną RK4 oraz funkcje straty wymuszające spójność trajektorii.

### Architektura modelu:
* **Segment B (Baseline)**: Deterministyczny komponent fizyczny wykorzystujący metodę **Rungego-Kutty 4. rzędu (RK4)** do modelowania ciągu, zmiennej masy, kierunku startu i grawitacji.
* **Segment R (Model resztkowy)**: Sieć rekurencyjna **GRU** (Gated Recurrent Unit), która przewiduje pełne przyspieszenie lub składową resztkową niewyjaśnioną przez bazę RK4.
* **Segment G (Brama persystencji)**: Opcjonalne warianty bramkowane łączą predykcję RK4-GRU z bazą ostatniego przyspieszenia przy użyciu uczonych wag dla każdej osi.
* **Segment I (Integration)**: Warstwa spójności trajektorii, która całkuje przewidywane przyspieszenie i porównuje uzyskaną pozycję z trajektorią referencyjną.

---

## Stack Technologiczny
* **Deep Learning i Matematyka**: `torch` (PyTorch), `numpy`, `pandas`.
* **Fizyka i Mechanika Lotu**: `rocketpy`, `xarray` (wielowymiarowe tablice danych).
* **Obsługa Danych i API**: `cdsapi` (pobieranie danych ERA5/Copernicus), `pyarrow`, `fastparquet` (optymalizacja I/O).
* **Wydajność i Równoległość**: `pathos` (multiprocessing), `cupy` (akceleracja CUDA).
* **Wizualizacja i Profilowanie**: `seaborn`, `matplotlib`, `snakeviz` (profilowanie kodu w przeglądarce).

---

## Kluczowe Funkcjonalności

### Generator lotów syntetycznych
* **Warunkowanie scenariuszy**: Generowanie lotów z konfigurowalnym ułamkiem utleniacza, skalą ciśnienia/ciągu, skalą masy rakiety i ukrytą zmiennością oporu.
* **Warunki Środowiskowe**: Integracja z reanalizą atmosferyczną ERA5.
* **Emulacja Sensorów**: Symulacja kanałów telemetrii używanych przez modele predykcyjne, w tym przyspieszenia, prędkości kątowej, ciśnienia barometrycznego i temperatury.
* **Dynamic Auto-Ranging**: Automatyczny dobór najbardziej precyzyjnego zakresu pomiarowego (np. akcelerometry 2g-16g) w trakcie lotu.

### Modele predykcyjne
* **Uczenie resztkowe RK4-GRU**: Warianty resztkowe przewidują poprawkę przyspieszenia względem deterministycznej bazy RK4.
* **Funkcja straty spójności trajektorii**: Warianty informowane fizycznie całkują przewidywane przyspieszenie i karzą narastający dryf pozycji w horyzoncie predykcji.
* **Brama persystencji**: Warianty bramkowane uczą się mieszać predykcję RK4-GRU z bazą ostatniego przyspieszenia, aby poprawić odporność przy przesunięciu symulacja-rzeczywistość.
* **Tryb Spin-Up/Cut-Off**: Modele używają historycznego okna Spin-Up z dostępną telemetrią oraz okna Cut-Off, w którym przewidują przyszłe przyspieszenie bez nowych pomiarów.

---

## Struktura Repozytorium
* `/docs` – Dokumentacja teoretyczna i schematy projektowe
* `/generator` – Kod źródłowy generatora danych syntetycznych
* `/prediction_models` – Kod treningu i ewaluacji modeli GRU oraz metod bazowych
* `/source_data` – Pliki konfiguracyjne, modele `.ork` oraz dane wejściowe

---

## Konfiguracja eksperymentów opisanych w artykule

Aktualne eksperymenty artykułowe wykorzystują warunkowane modele GRU z:
* 8 wejściami telemetrycznymi: przyspieszenie, prędkość kątowa, ciśnienie barometryczne i temperatura
* 3 wejściami opisującymi scenariusz: ułamek utleniacza, skala ciśnienia i skala masy rakiety
* wewnętrzną flagą trybu Spin-Up/Cut-Off dodawaną przez model
* historycznym oknem Spin-Up o długości 120 próbek i horyzontem predykcji 60 próbek
* syntetyczną ewaluacją 3D na lotach testowych oraz rzeczywistą ewaluacją osi Z na telemetrii FAR-OUT 2026

Pliki rzeczywistego lotu w `source_data/far_out_26_data` są dołączone za zgodą projektu w celach badawczych i reprodukowalności.

---
**Zespół projektowy**: Alicja Macutkiewicz, Weronika Marszalik, Paweł Leczkowski, Wiktor Ludwichowski, Emilia Łukasiuk

</details>
