import json
import sys
import unittest
from pathlib import Path

import numpy as np
from rocketpy import Environment, Flight

REPOSITORY = Path(__file__).resolve().parents[2]
GENERATOR_SRC = REPOSITORY / "generator" / "src"
sys.path.insert(0, str(GENERATOR_SRC))

from main import (  # noqa: E402
    apply_oxidizer_model,
    default_scenario,
    init_base_motor_from_JSON,
    init_rocket_from_JSON,
    load_scaled_thrust_curve,
    load_scenario,
    sample_scenario,
    select_thrust_path,
)


class GeneratorScenarioTest(unittest.TestCase):
    def test_thrust_path_prefers_canonical_key_without_evaluating_legacy_fallback(self):
        self.assertEqual(select_thrust_path({"thrust_source": "canonical.csv"}), "canonical.csv")
        self.assertEqual(select_thrust_path({"thrust_source_100": "legacy.csv"}), "legacy.csv")

    def build_motor(self, scenario_file, paths_file):
        scenario = (
            default_scenario()
            if scenario_file is None
            else load_scenario(GENERATOR_SRC / "scenarios" / scenario_file)
        )
        paths = json.loads((GENERATOR_SRC / paths_file).read_text(encoding="utf-8"))
        source = paths["source_model_path"]
        parameters_path = (GENERATOR_SRC / source["parameters"]).resolve()
        thrust_path = (GENERATOR_SRC / source["thrust_source"]).resolve()
        parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
        fraction, pressure, mass_scale = sample_scenario(
            scenario, np.random.default_rng(0)
        )
        metadata = apply_oxidizer_model(parameters, scenario, fraction)
        source_pressure = float(source.get("thrust_source_pressure_scale", 1.0))
        thrust = load_scaled_thrust_curve(
            thrust_path,
            pressure,
            metadata["burn_time_scale"],
            source_pressure,
        )
        motor = init_base_motor_from_JSON(parameters, thrust)
        return scenario, fraction, pressure, mass_scale, metadata, thrust, motor

    def test_all_documented_scenarios_construct_a_finite_motor(self):
        cases = [
            (None, "paths.json"),
            ("robustness_oxidizer_40_100.json", "paths_robustness.json"),
            ("farout_26.json", "paths_farout_26.json"),
        ]
        for scenario_file, paths_file in cases:
            with self.subTest(scenario=scenario_file or "default"):
                *_, thrust, motor = self.build_motor(scenario_file, paths_file)
                self.assertTrue(np.all(np.isfinite(thrust)))
                self.assertGreater(motor.propellant_initial_mass, 0.0)

    def test_oxidizer_scenario_matches_conditioned_rk4_parameters(self):
        scenario, fraction, _, _, metadata, thrust, motor = self.build_motor(
            "robustness_oxidizer_40_100.json", "paths_robustness.json"
        )
        parameters = json.loads(
            (
                REPOSITORY
                / "source_model"
                / "R7_SIMLE"
                / "R7_ROBUSTNESS"
                / "parameters.json"
            ).read_text(encoding="utf-8")
        )
        motor_parameters = parameters["motors"]
        model = scenario["oxidizer_model"]

        self.assertEqual(model["nominal_oxidizer_mass_kg"], motor_parameters["nominal_oxidizer_mass_kg"])
        self.assertEqual(model["fixed_fuel_mass_kg"], motor_parameters["fixed_fuel_mass_kg"])
        self.assertEqual(model["burn_time_exponent"], motor_parameters["oxidizer_burn_time_exponent"])
        configured_propellant_mass = (
            model["nominal_oxidizer_mass_kg"] + model["fixed_fuel_mass_kg"]
        )
        nominal_curve = np.loadtxt(
            REPOSITORY
            / "source_model"
            / "R7_SIMLE"
            / "R7_OUTPUT"
            / "thrust_source.csv",
            delimiter=",",
        )
        expected_isp = np.trapezoid(
            nominal_curve[:, 1], nominal_curve[:, 0]
        ) / (configured_propellant_mass * 9.80665)

        self.assertAlmostEqual(
            configured_propellant_mass, motor_parameters["fuel_mass"]
        )
        self.assertAlmostEqual(motor_parameters["isp"], expected_isp)
        self.assertAlmostEqual(motor.propellant_initial_mass, 12.24 * fraction)
        self.assertAlmostEqual(metadata["burn_time_scale"], fraction)
        self.assertAlmostEqual(thrust[-1, 0], 10.092 * fraction, places=5)

    def test_farout_curve_is_not_pressure_scaled_twice(self):
        _, _, pressure, _, _, thrust, motor = self.build_motor(
            "farout_26.json", "paths_farout_26.json"
        )
        source = np.loadtxt(
            REPOSITORY / "source_model" / "R7_SIMLE" / "R7_FAROUT_26" / "thrust_source.csv",
            delimiter=",",
        )

        self.assertEqual(pressure, 0.85)
        np.testing.assert_allclose(thrust, source)
        self.assertAlmostEqual(motor.propellant_initial_mass, 5.5, places=6)

    def test_low_impulse_high_mass_flight_remains_finite(self):
        scenario = load_scenario(
            GENERATOR_SRC / "scenarios" / "robustness_oxidizer_40_100.json"
        )
        parameters = json.loads(
            (
                REPOSITORY
                / "source_model"
                / "R7_SIMLE"
                / "R7_ROBUSTNESS"
                / "parameters.json"
            ).read_text(encoding="utf-8")
        )
        parameters["rocket"]["mass"] *= 1.3
        parameters["rocket"]["inertia"] = [
            value * 1.3 for value in parameters["rocket"]["inertia"]
        ]
        metadata = apply_oxidizer_model(parameters, scenario, 0.4)
        thrust = load_scaled_thrust_curve(
            REPOSITORY / "source_model" / "R7_SIMLE" / "R7_OUTPUT" / "thrust_source.csv",
            pressure_scale=0.4,
            burn_time_scale=metadata["burn_time_scale"],
        )
        motor = init_base_motor_from_JSON(parameters, thrust)
        rocket = init_rocket_from_JSON(
            parameters,
            str(
                REPOSITORY
                / "source_model"
                / "R7_SIMLE"
                / "R7_OUTPUT"
                / "drag_curve.csv"
            ),
            motor,
        )
        environment = Environment(
            latitude=35.34723084964506,
            longitude=-117.81006,
            elevation=0.0,
        )
        flight = Flight(
            rocket=rocket,
            environment=environment,
            rail_length=18.0,
            heading=90.0,
            inclination=80.0,
        )

        self.assertTrue(np.isfinite(flight.apogee))
        self.assertTrue(np.isfinite(flight.t_final))
        self.assertGreater(flight.apogee, 0.0)


if __name__ == "__main__":
    unittest.main()
