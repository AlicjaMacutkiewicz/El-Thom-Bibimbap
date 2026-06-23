import argparse

import numpy as np
import torch
import torch.optim as optim
from data_loader import INPUT_COLUMNS, make_sequences, read_flight_data, split_flights
from GRU_model import GRU
from physics import (
    TotalLoss,
    calculate_x_b_conditioned,
    default_physics_paths,
    load_parameters,
    load_thrust_curve,
)
from train import train_model
from visualize import plot_losses, plot_prediction


def get_best_device():
    """Identifies and returns the best available hardware accelerator."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        try:
            import intel_extension_for_pytorch  # type: ignore # noqa: F401

            dev = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
        except ImportError:
            dev = torch.device("cpu")

    print(f"running on {dev}")
    return dev


def parse_args():
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", default="../../../../data")
    parser.add_argument("--start-flight", type=int, default=0)
    parser.add_argument("--num-flights", type=int, default=1652)
    parser.add_argument("--batch-size", type=int, default=1280)
    parser.add_argument("--training-rounds", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--year", type=str, default="2025")
    parser.add_argument("--downsample", type=int, default=25)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="gru_res_phys")
    parser.add_argument("--parameters", type=str, default=None)
    parser.add_argument("--thrust-curve", type=str, default=None)

    return parser.parse_args()


def drop_last(tensors, batch_size):
    remainder = len(tensors[0]) % batch_size
    if remainder != 0:
        return [t[:-remainder] for t in tensors]
    return tensors


def main():
    args = parse_args()
    device = get_best_device()

    sampling_rate = 500.0 / args.downsample
    # pred_len is set equal to seq_len so the model predicts
    # the same number of future samples as it receives from the past
    pred_len = args.seq_len

    # load physics parameters
    parameters_path, thrust_curve_path = default_physics_paths()
    if args.parameters:
        parameters_path = args.parameters
    if args.thrust_curve:
        thrust_curve_path = args.thrust_curve
    parameters = load_parameters(parameters_path)
    thrust_curve = load_thrust_curve(thrust_curve_path)

    # load flight data
    flights_inputs, flights_conditions, flights_targets, flight_positions, flight_times = read_flight_data(
        args.start_flight,
        args.num_flights,
        output_dir=args.output_dir,
        downsample=args.downsample,
    )

    # train / test data split
    train_inputs, test_inputs = split_flights(flights_inputs)
    train_conditions, test_conditions = split_flights(flights_conditions)
    train_targets, test_targets = split_flights(flights_targets)
    train_positions, test_positions = split_flights(flight_positions)
    train_times, test_times = split_flights(flight_times)

    # normalization statistics
    all_train_inputs = np.concatenate(train_inputs, axis=0)
    mean_in = all_train_inputs.mean(axis=0)
    std_in = all_train_inputs.std(axis=0)
    std_in = np.where(std_in == 0, 1e-6, std_in)  # div by zero safeguard

    # x_total stats — kept only for denormalizing the history plot in visualize.py
    all_train_targets = np.concatenate(train_targets, axis=0)
    mean_acc = all_train_targets.mean(axis=0)
    std_acc = all_train_targets.std(axis=0)
    std_acc = np.where(std_acc == 0, 1e-6, std_acc)

    all_train_pos = np.concatenate(train_positions, axis=0)
    mean_pos = all_train_pos.mean(axis=0)
    std_pos = all_train_pos.std(axis=0)
    std_pos = np.where(std_pos == 0, 1e-6, std_pos)

    # residual stats — what the GRU actually predicts: x_s = x_total - x_b
    # must be computed on raw targets before normalizing, using raw times
    x_b_train = np.concatenate(
        [
            calculate_x_b_conditioned(
                torch.from_numpy(times),
                parameters,
                thrust_curve,
                sampling_rate,
                conditions,
            )
            for times, conditions in zip(train_times, train_conditions, strict=True)
        ],
        axis=0,
    )
    all_train_targets_raw = np.concatenate(train_targets, axis=0)
    x_s_train = all_train_targets_raw - x_b_train
    mean_xs = x_s_train.mean(axis=0)
    std_xs = x_s_train.std(axis=0)
    std_xs = np.where(std_xs == 0, 1e-6, std_xs)

    print(f"residual stats — mean_xs: {mean_xs},  std_xs: {std_xs}")
    print(f"x_total  stats — mean_acc: {mean_acc}, std_acc: {std_acc}")

    if args.model_type == "gru":
        target_mean = mean_acc
        target_std = std_acc
    else: # Covers "gru_res" and "gru_res_phys"
        target_mean = mean_xs
        target_std = std_xs

    # apply normalization
    train_inputs = [(f - mean_in) / std_in for f in train_inputs]
    test_inputs = [(f - mean_in) / std_in for f in test_inputs]

    # targets normalized with RESIDUAL stats, not x_total stats
    train_targets = [(f - target_mean) / target_std for f in train_targets]
    test_targets = [(f - target_mean) / target_std for f in test_targets]

    train_positions = [(p - mean_pos) / std_pos for p in train_positions]
    test_positions = [(p - mean_pos) / std_pos for p in test_positions]


    # sequence generation
    loss = TotalLoss(
        parameters, thrust_curve, mean_xs, std_xs, mean_pos, std_pos, sampling_rate, lambda_h=0.2, model_type = args.model_type,
    ).to(device)

    (
        X_train,
        condition_train,
        y_hist_train,
        y_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
    ) = make_sequences(
        train_inputs,
        train_conditions,
        train_targets,
        train_positions,
        train_times,
        args.seq_len,
        pred_len,
    )
    (
        X_test,
        condition_test,
        y_hist_test,
        y_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
    ) = make_sequences(
        test_inputs,
        test_conditions,
        test_targets,
        test_positions,
        test_times,
        args.seq_len,
        pred_len,
    )

    print("data preprocessing and sequence generation complete")
    print(f"model input columns ({len(INPUT_COLUMNS)}): {INPUT_COLUMNS}")

    (
        X_train,
        condition_train,
        y_hist_train,
        y_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
    ) = drop_last(
        [
            X_train,
            condition_train,
            y_hist_train,
            y_train,
            pos_train,
            t_train,
            initial_pos_train,
            initial_vel_train,
            initial_time_train,
        ],
        args.batch_size,
    )

    (
        X_test,
        condition_test,
        y_hist_test,
        y_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
    ) = drop_last(
        [
            X_test,
            condition_test,
            y_hist_test,
            y_test,
            pos_test,
            t_test,
            initial_pos_test,
            initial_vel_test,
            initial_time_test,
        ],
        args.batch_size,
    )

    # model init and training
    model = GRU(input_size=X_train.shape[-1], hidden_size=64, output_size=3, num_layers=2, dropout=0.2)

    if args.resume_from:
        print(f"resuming training from checkpoint: {args.resume_from}")
        state_dict = torch.load(args.resume_from, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):  # noqa: SIM108
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"found {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    train_losses, test_losses = train_model(
        model,
        X_train,
        condition_train,
        y_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
        X_test,
        condition_test,
        y_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
        loss,
        optimizer,
        device=device,
        batch_size=args.batch_size,
        training_rounds=args.training_rounds,
        pred_len=pred_len,
    )

    # visualization and saving

    model_filename = f"gru_model_rounds{args.training_rounds}_seq{args.seq_len}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"model weights saved to file: {model_filename}")

    with open("learning_state.txt", "a") as log_file:
        log_file.write(f"trained model: {model_filename}\n")
        log_file.write(
            f"parameters: epochs={args.training_rounds}, batch={args.batch_size}, year={args.year}\n"
        )
        log_file.write(f"flights utilized: {len(flights_inputs)}\n")
        log_file.write("-" * 40 + "\n")
    print("learning_state.txt updated")

    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    plot_losses(train_losses, test_losses)
    diagnostic_sample_indices = sorted(
        {
            0,
            len(X_test) // 2,
            len(X_test) - 1,
        }
    )
    for sample_idx in diagnostic_sample_indices:
        plot_prediction(
            model_to_save,
            X_test,
            y_hist_test,
            y_test,
            t_test,
            pred_len,
            parameters,
            thrust_curve,
            target_mean,
            target_std,
            mean_acc,
            std_acc,
            device,
            sampling_rate=sampling_rate,
            sample_idx=sample_idx,
            condition_test=condition_test,
        )


if __name__ == "__main__":
    main()
