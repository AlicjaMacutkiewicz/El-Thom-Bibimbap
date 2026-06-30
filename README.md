`Polish version below!`

# Rocket Flight Prediction System ("El-Thom-Bibimbap")
**Flight trajectory prediction system for suborbital platform telemetry**

This project implements an advanced trajectory prediction system for suborbital rockets, tested on the **R7 Orzeł** model (KN SimLE, Gdańsk University of Technology) The goal is to maintain continuous vehicle tracking in the event of communication loss or sensor failure by combining deterministic physics with deep learning

---

## Project Concept
In classical telemetry, losing the signal results in losing all knowledge of the object's position This system analyzes sensor data in real-time and "fits" it to a physical model enhanced by a neural network, allowing for position estimation until connectivity is restored

### Core System Components:
* **"Grzesiek" (The Generator)**: A high-performance engine for creating synthetic flight data and simulating sensor failures
* **"Maurycy" (The Model)**: The actual prediction algorithm that integrates neural networks with physical constraints

### Model Architecture:
* **Segment B (Base)**: A deterministic physical component utilizing the **4th-order Runge-Kutta (RK4)** method to solve Newtonian equations of motion
* **Segment S (Stochastic)**: A **Gated Recurrent Unit (GRU)** recurrent network responsible for modeling non-linear disturbances (wind, aero drag) that simplified analytical models do not cover
* **Segment I (Integration)**: A data fusion module utilizing the **IPINN** (Inverse Physics-Informed Neural Networks) architecture with "soft physical constraints"

---
## Tech Stack
* **Deep Learning & Math**: `torch` (PyTorch), `numpy`, `pandas`.
* **Rocketry & Physics**: `rocketpy`, `xarray` (multi-dimensional arrays).
* **Data Science & API**: `cdsapi` (Copernicus/ERA5 fetching), `pyarrow`, `fastparquet` (columnar data storage).
* **Performance & Parallelization**: `pathos` (multiprocessing), `cupy` (CUDA support).
* **Visualization & Profiling**: `seaborn`, `matplotlib`, `snakeviz` (browser-based profiling).

---

## Key Features

### Generator "Grzesiek" 
* **Stochastic Motor Modeling**: Simulation of deviations in nominal parameters such as grain geometry, nozzle specs, and total impulse
* **Environmental Conditions**: Integration with the ERA5 database for the Gdańsk University of Technology location
* **Error Emulation**: Simulation of bit-switch, sample-and-hold, and signal dropouts as a function of wind speed and g-loads
* **Dynamic Auto-Ranging**: Automatic selection of the most precise measurement range (e.g., dynamically switching between 2g-16g accelerometer streams) during flight

### Model "Maurycy"
* **Physics-Informed Architecture**: Implements a custom loss function $L_{Total} = MSE_{acc} + (\lambda \cdot PINN_{loss})$ to ensure estimated trajectories adhere to Newtonian kinematics.
* **Dynamic Operational Modes**: Supports **Spin-Up** mode for real-time tuning using live sensor data and **Cut-off** mode for autonomous sequence-to-sequence prediction during signal loss.
* **Analytical Integration (RK4)**: Utilizes the 4th-order Runge-Kutta method to calculate deterministic base components ($x_b$) such as gravity and thrust.
* **Non-linear Residual Learning**: The GRU network focuses exclusively on predicting the non-linear residual component ($x_s$), representing complex forces like wind and atmospheric drag.

---

## Repository Structure
* `/docs` – Theoretical documentation and project schemes
* `/generator` – Source code for the "Grzesiek" synthetic data generator
* `/prediction_models` – Implementations of **"Maurycy"** (including GRU, VAR, and Integration modules)
* `/source_data` – Configuration files, `.ork` models, and input data

---

## Project Status (Roadmap)

### PHASES 1 & 2 (Completed)
* Implementation of sensor support (BME280, GNSS, IMU)
* Creation of the atmospheric model and variable motor performance parameters
* Data I/O optimization using the `.parquet` columnar format

### PHASE 3 (In Progress)
* Implementation of the composite PINN loss function: $L_{Total} = MSE_{acc} + \lambda_{PINN} L_{PINN}$
* Introduction of operational modes: **Spin-Up** (sensor tuning) and **Cut-off** (autonomous prediction/looping)
* Automation of result visualization and training status reporting

---
**Project Team**: Alicja Macutkiewicz, Weronika Marszalik, Paweł Leczkowski, Wiktor Ludwichowski, Emilia Łukasiuk

---
---
<details>
<summary><b> Polish Version (click here)</b></summary>

# Rocket Flight Prediction System ("El-Thom-Bibimbap")
**System predykcji toru lotu dla telemetrii platform suborbitalnych**

Projekt realizuje zaawansowany system przewidywania trajektorii rakiet suborbitalnych (testowany na modelu **R7 Orzeł**, KN SimLE PG) Celem jest utrzymanie ciągłości śledzenia pojazdu w przypadku utraty łączności lub awarii sensorów poprzez połączenie fizyki deterministycznej z głębokim uczeniem

---

## Idea projektu
W klasycznej telemetrii utrata sygnału oznacza utratę wiedzy o pozycji obiektu System analizuje dane z czujników w czasie rzeczywistym i "dopasowuje" do nich model fizyczny wspomagany przez sieć neuronową, co pozwala oszacować pozycję aż do momentu odzyskania łączności

### Główne komponenty systemu:
* **"Grzesiek" (Generator)**: Wysokowydajny silnik do tworzenia syntetycznych danych lotu i symulacji awarii sensorów
* **"Maurycy" (Model)**: Właściwy algorytm predykcji integrujący sieci neuronowe z więzami fizycznymi

### Architektura modelu:
* **Segment B (Base)**: Deterministyczny komponent fizyczny wykorzystujący metodę **Rungego-Kutty 4. rzędu (RK4)** do rozwiązywania równań ruchu Newtona
* **Segment S (Stochastic)**: Sieć rekurencyjna **GRU** (Gated Recurrent Unit), odpowiedzialna za modelowanie nieliniowych zakłóceń (wiatr, opór aero), których nie obejmują uproszczone modele analityczne
* **Segment I (Integration)**: Moduł fuzji danych wykorzystujący architekturę **IPINN** (Inverse Physics-Informed Neural Networks) z zastosowaniem tzw. miękkich więzów fizycznych

---

## Stack Technologiczny
* **Deep Learning i Matematyka**: `torch` (PyTorch), `numpy`, `pandas`.
* **Fizyka i Mechanika Lotu**: `rocketpy`, `xarray` (wielowymiarowe tablice danych).
* **Obsługa Danych i API**: `cdsapi` (pobieranie danych ERA5/Copernicus), `pyarrow`, `fastparquet` (optymalizacja I/O).
* **Wydajność i Równoległość**: `pathos` (multiprocessing), `cupy` (akceleracja CUDA).
* **Wizualizacja i Profilowanie**: `seaborn`, `matplotlib`, `snakeviz` (profilowanie kodu w przeglądarce).

---

## Kluczowe Funkcjonalności

### Generator "Grzesiek"
* **Stochastyczne Modelowanie Silnika**: Symulacja odchyłów parametrów nominalnych (geometria ziarna, dyszy, impuls całkowity)
* **Warunki Środowiskowe**: Integracja z bazą ERA5 dla lokalizacji Politechniki Gdańskiej
* **Emulacja Błędów**: Symulacja bit-switch, sample-and-hold oraz dropoutów sygnału uzależnionych od przeciążeń i wiatru
* **Dynamic Auto-Ranging**: Automatyczny dobór najbardziej precyzyjnego zakresu pomiarowego (np. akcelerometry 2g-16g) w trakcie lotu

### Model "Maurycy"
* **Architektura Informowana Fizycznie**: Zaimplementowana funkcja straty $L_{Total} = MSE_{acc} + (\lambda \cdot PINN_{loss})$ wymuszająca zgodność estymowanych trajektorii z zasadami dynamiki Newtona.
* **Dynamiczne Tryby Pracy**: Obsługa trybu **Spin-Up** (dostrajanie do danych z sensorów w czasie rzeczywistym) oraz **Cut-off** (autonomiczna predykcja sekwencyjna po utracie sygnału).
* **Integracja Analityczna (RK4)**: Wykorzystanie metody Rungego-Kutty 4. rzędu do obliczania deterministycznych składowych bazowych ($x_b$), takich jak grawitacja i ciąg.
* **Uczenie Składowych Nieliniowych**: Sieć GRU koncentruje się wyłącznie na przewidywaniu nieliniowego członu resztkowego ($x_s$), reprezentującego złożone siły takie jak wiatr i opór atmosferyczny.

---

## Struktura Repozytorium
* `/docs` – Dokumentacja teoretyczna i schematy projektowe
* `/generator` – Kod źródłowy generatora danych syntetycznych
* `/prediction_models` – Implementacje modelu **"Maurycy"** (moduły GRU, VAR oraz Integracja)
* `/source_data` – Pliki konfiguracyjne, modele `.ork` oraz dane wejściowe

---

## Status Projektu (Roadmap)

### ETAP 1 & 2 (Zakończone)
* Implementacja obsługi czujników (BME280, GNSS, IMU)
* Stworzenie modelu atmosfery i zmiennych warunków pracy silnika
* Optymalizacja zapisu danych do formatu `.parquet`

### ETAP 3 (W realizacji)
* Implementacja złożonej funkcji straty PINN: $L_{Total} = MSE_{acc} + \lambda_{PINN} L_{PINN}$
* Wprowadzenie trybów pracy: **Spin-Up** (dostrajanie do czujników) oraz **Cut-off** (predykcja autonomiczna/zapętlenie)
* Automatyzacja wizualizacji wyników i raportowania stanu uczenia

---
**Zespół projektowy**: Alicja Macutkiewicz, Weronika Marszalik, Paweł Leczkowski, Wiktor Ludwichowski, Emilia Łukasiuk

</details>