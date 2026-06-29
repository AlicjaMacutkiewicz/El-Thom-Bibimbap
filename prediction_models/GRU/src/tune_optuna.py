from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import optuna
except ImportError as exc:  # pragma: no cover - user-facing dependency guard
    raise SystemExit(
        "Optuna is not installed. Install it in the active venv with: pip install optuna"
    ) from exc


ITERATION_RE = re.compile(
    r"Iteration\s+(?P<epoch>\d+), train loss:\s+(?P<train>[0-9.eE+-]+), "
    r"test loss:\s+(?P<test>[0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an Optuna hyperparameter sweep for the gated GRU trajectory model. "
            "Each trial launches main.py in an isolated output directory and ranks "
            "the setup by best validation/test loss."
        )
    )
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    default_data = repo_root.parent / "data"
    parser.add_argument("--output-dir", type=Path, default=default_data)
    parser.add_argument("--study-dir", type=Path, default=repo_root / "prediction_models" / "GRU" / "optuna_runs")
    parser.add_argument("--study-name", default="gated_gru_seq120_pred60")
    parser.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to sqlite in --study-dir.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Optional wall-clock timeout in seconds.")
    parser.add_argument("--trial-rounds", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--pred-len", type=int, default=60)
    parser.add_argument("--start-flight", type=int, default=0)
    parser.add_argument("--num-flights", type=int, default=1652)
    parser.add_argument("--batch-size", type=int, default=36000)
    parser.add_argument("--downsample", type=int, default=25)
    parser.add_argument(
        "--model-type",
        choices=["gru_res_phys_persist_gate", "last_acc_gru"],
        default="gru_res_phys_persist_gate",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tune-architecture", action="store_true")
    parser.add_argument("--lambda-h-min", type=float, default=0.05)
    parser.add_argument("--lambda-h-max", type=float, default=0.6)
    parser.add_argument("--regret-min", type=float, default=0.05)
    parser.add_argument("--regret-max", type=float, default=1.0)
    parser.add_argument("--gate-smooth-min", type=float, default=0.0)
    parser.add_argument("--gate-smooth-max", type=float, default=1.0)
    parser.add_argument("--learning-rate-min", type=float, default=1e-4)
    parser.add_argument("--learning-rate-max", type=float, default=8e-4)
    parser.add_argument("--dropout-min", type=float, default=0.0)
    parser.add_argument("--dropout-max", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Allow each trial to create a W&B run. Disabled by default to keep sweeps quiet.",
    )
    return parser.parse_args()


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "lambda_h": trial.suggest_float("lambda_h", args.lambda_h_min, args.lambda_h_max),
        "persistence_regret_weight": trial.suggest_float(
            "persistence_regret_weight", args.regret_min, args.regret_max, log=True
        ),
        "gate_smooth_weight": trial.suggest_float(
            "gate_smooth_weight", args.gate_smooth_min, args.gate_smooth_max
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", args.learning_rate_min, args.learning_rate_max, log=True
        ),
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
    }
    if args.tune_architecture:
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 96, 128])
        params["num_layers"] = trial.suggest_categorical("num_layers", [2, 3])
    return params


def terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def run_trial(trial: optuna.Trial, args: argparse.Namespace, script_dir: Path) -> float:
    params = suggest_params(trial, args)
    trial_dir = args.study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "training_summary.json"
    log_path = trial_dir / "trial.log"
    command = [
        sys.executable,
        str(script_dir / "main.py"),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--start-flight",
        str(args.start_flight),
        "--num-flights",
        str(args.num_flights),
        "--batch-size",
        str(args.batch_size),
        "--training-rounds",
        str(args.trial_rounds),
        "--seq-len",
        str(args.seq_len),
        "--pred-len",
        str(args.pred_len),
        "--downsample",
        str(args.downsample),
        "--model-type",
        args.model_type,
        "--hidden-size",
        str(params["hidden_size"]),
        "--num-layers",
        str(params["num_layers"]),
        "--dropout",
        str(params["dropout"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--weight-decay",
        str(args.weight_decay),
        "--lambda-h",
        str(params["lambda_h"]),
        "--persistence-regret-weight",
        str(params["persistence_regret_weight"]),
        "--gate-smooth-weight",
        str(params["gate_smooth_weight"]),
        "--year",
        f"optuna_trial_{trial.number:04d}",
        "--training-summary",
        str(summary_path),
        "--skip-plots",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", str(trial_dir / "matplotlib-cache"))
    if not args.enable_wandb:
        env["WANDB_MODE"] = "disabled"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.write("PARAMS\n")
        log_file.write(json.dumps(params, indent=2) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=trial_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[trial {trial.number:04d}] {line}", end="")
            log_file.write(line)
            log_file.flush()
            match = ITERATION_RE.search(line)
            if match is None:
                continue
            epoch = int(match.group("epoch"))
            test_loss = float(match.group("test"))
            trial.report(test_loss, step=epoch)
            if epoch >= args.pruner_warmup_epochs and trial.should_prune():
                terminate_process(process)
                raise optuna.TrialPruned(f"pruned at epoch {epoch} with test loss {test_loss}")
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"trial {trial.number} failed with exit code {return_code}; see {log_path}")
    if not summary_path.exists():
        raise RuntimeError(f"trial {trial.number} did not write {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    score = float(summary["best_test_loss"])
    trial.set_user_attr("trial_dir", str(trial_dir.resolve()))
    trial.set_user_attr("best_checkpoint", summary.get("best_checkpoint"))
    trial.set_user_attr("final_model", summary.get("model_file"))
    trial.set_user_attr("final_test_loss", summary.get("final_test_loss"))
    return score


def write_study_outputs(study: optuna.Study, study_dir: Path) -> None:
    best = {
        "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
    }
    (study_dir / "best_trial.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "trial_dir",
        "best_checkpoint",
        "final_model",
        *sorted({key for trial in study.trials for key in trial.params}),
    ]
    with (study_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
                "trial_dir": trial.user_attrs.get("trial_dir"),
                "best_checkpoint": trial.user_attrs.get("best_checkpoint"),
                "final_model": trial.user_attrs.get("final_model"),
            }
            row.update(trial.params)
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.trial_rounds <= 0:
        raise ValueError("--trial-rounds must be positive.")
    args.study_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{(args.study_dir / 'study.sqlite3').resolve()}"

    sampler = optuna.samplers.TPESampler(seed=41)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=args.pruner_warmup_epochs,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    print(f"Study: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Study directory: {args.study_dir.resolve()}")
    started = time.time()
    study.optimize(
        lambda trial: run_trial(trial, args, script_dir),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )
    write_study_outputs(study, args.study_dir)
    elapsed = (time.time() - started) / 60.0
    print(f"\nFinished Optuna sweep in {elapsed:.1f} min")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.8e}")
    print("Best params:")
    print(json.dumps(study.best_trial.params, indent=2))
    print(f"Best checkpoint: {study.best_trial.user_attrs.get('best_checkpoint')}")
    print(f"Open: {args.study_dir / 'best_trial.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
