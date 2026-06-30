import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

CLASSICAL_MODEL_SRC = Path(__file__).resolve().parents[2] / "GRU" / "src"
if str(CLASSICAL_MODEL_SRC) not in sys.path:
    sys.path.append(str(CLASSICAL_MODEL_SRC))

from data_loader import estimate_velocity, read_flight_data, split_flights  # noqa: E402
from GRU_model import GRU  # noqa: E402
from physics import (  # noqa: E402
    calculate_x_b,
    default_physics_paths,
    load_parameters,
    load_thrust_curve,
)


def get_best_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="baseline poly vs hybrid GRU")
    parser.add_argument("--model-path", type=str, required=True, help="path to your .pth file")
    parser.add_argument("--output-dir", default="../../../demo-flights/", help="path to data")
    parser.add_argument("--num-flights", type=int, default=48)
    parser.add_argument("--downsample", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=25.0, help="error threshold in meters")
    return parser.parse_args()

def integrate_acceleration_to_position(acc_z, initial_pos_z, initial_vel_z, dt):
    """Converts the GRU's acceleration array back into position using kinematics."""
    delta_v = cumulative_trapezoid(acc_z, dx=dt, initial=0)
    vel_z = initial_vel_z + delta_v
    
    delta_p = cumulative_trapezoid(vel_z, dx=dt, initial=0)
    pos_z = initial_pos_z + delta_p
    
    return pos_z

def main():
    args = parse_args()
    device = get_best_device()
    sampling_rate = 500.0 / args.downsample
    dt = 1.0 / sampling_rate
    
    lookback_steps = 100
    pred_steps = 100

    print("loading physics and data...")
    parameters_path, thrust_curve_path = default_physics_paths()
    parameters = load_parameters(parameters_path)
    thrust_curve = load_thrust_curve(thrust_curve_path)

    flights_inputs, flights_targets, flight_positions, flight_times = read_flight_data(
        0, args.num_flights, output_dir=args.output_dir, downsample=args.downsample
    )

    train_inputs, test_inputs = split_flights(flights_inputs)
    train_targets, _test_targets = split_flights(flights_targets)
    _train_positions, test_positions = split_flights(flight_positions)
    train_times, test_times = split_flights(flight_times)

    # calculate normalization stats for the inputs
    all_train_inputs = np.concatenate(train_inputs, axis=0)
    mean_in = all_train_inputs.mean(axis=0)
    std_in = all_train_inputs.std(axis=0)
    std_in = np.where(std_in == 0, 1e-6, std_in) 

    # calculate normalization stats for the targets
    # AFTER
    all_train_targets = np.concatenate(train_targets, axis=0)
    mean_acc = all_train_targets.mean(axis=0)
    std_acc = all_train_targets.std(axis=0)
    std_acc = np.where(std_acc == 0, 1e-6, std_acc)

    # residual stats — must match what main.py used during training
    all_train_times = np.concatenate(train_times, axis=0)
    x_b_train = calculate_x_b(
        torch.from_numpy(all_train_times.astype(np.float32)),
        parameters, thrust_curve, sampling_rate
    )
    x_s_train = all_train_targets - x_b_train
    mean_xs = x_s_train.mean(axis=0)
    std_xs  = x_s_train.std(axis=0)
    std_xs  = np.where(std_xs == 0, 1e-6, std_xs)

    print(f"mean_xs: {mean_xs.round(4)}")
    print(f"std_xs:  {std_xs.round(4)}")
    print(f"mean_acc: {mean_acc.round(4)}")
    print(f"std_acc:  {std_acc.round(4)}")

    print(f"loading GRU model from: {args.model_path}")
    model = GRU(input_size=8, hidden_size=64, output_size=3, num_layers=2)
    state_dict = torch.load(args.model_path, map_location=device)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    # trackers
    total_windows = 0
    math_failures = 0
    gru_failures = 0
    both_failed = 0

    # calculate total expected windows for the progress bar
    total_expected_windows = sum(
        max(0, len(t) - pred_steps - lookback_steps) 
        for t in test_times
    )

    print("\n running sliding window evaluation")
    
    with torch.no_grad():  # noqa: SIM117
        with tqdm(total=total_expected_windows, desc="evaluating windows", unit="win") as pbar:
            for flight_idx in range(len(test_times)):
                f_in = test_inputs[flight_idx]
                pos = test_positions[flight_idx]
                t = test_times[flight_idx]
                
                # precalculate velocities for the whole flight
                vel = estimate_velocity(pos, t)
                
                z_pos = pos[:, 2] # Extract Z Altitude
                
                for i in range(lookback_steps, len(t) - pred_steps):
                    total_windows += 1
                    
                    z_actual = z_pos[i : i + pred_steps]
                    
                    # math baseline prediction (2nd degree polynomial)
                    t_lookback = t[i - lookback_steps : i]
                    z_lookback = z_pos[i - lookback_steps : i]
                    t_norm = t_lookback - t_lookback[0]
                    
                    coeffs = np.polyfit(t_norm, z_lookback, 2)
                    poly_func = np.poly1d(coeffs)
                    
                    t_future = t[i : i + pred_steps]
                    t_future_norm = t_future - t_lookback[0]
                    z_pred_math = poly_func(t_future_norm)
                    
                    rmse_math = np.sqrt(np.mean((z_actual - z_pred_math)**2))
                    
                    # 2. pinn-gru + rk4 prediction
                    X_window = f_in[i - lookback_steps : i]
                    X_norm = (X_window - mean_in) / std_in
                    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    t_future_tensor = torch.tensor(t_future, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    predicted_acc_norm, _ = model(X_tensor, pred_len=pred_steps)
                    predicted_acc = (predicted_acc_norm[0].cpu().numpy() * std_xs) + mean_xs                    
                    base_acc = calculate_x_b(t_future_tensor, parameters, thrust_curve, sampling_rate)[0]
                    
                    hybrid_acc = predicted_acc + base_acc
                    hybrid_acc_z = hybrid_acc[:, 2] 
                    if total_windows <= 5:
                        print(f"window {total_windows}: predicted_acc_z mean={predicted_acc[:,2].mean():.3f}, base_acc_z mean={base_acc[:,2].mean():.3f}, hybrid_z mean={hybrid_acc_z.mean():.3f}")
                    
                    initial_pos_z = z_pos[i]
                    initial_vel_z = vel[i, 2]
                    z_pred_gru = integrate_acceleration_to_position(hybrid_acc_z, initial_pos_z, initial_vel_z, dt)
                    
                    rmse_gru = np.sqrt(np.mean((z_actual - z_pred_gru)**2))
                    
                    # compare models
                    math_tripped = rmse_math > args.threshold
                    gru_tripped = rmse_gru > args.threshold
                    
                    if math_tripped: math_failures += 1
                    if gru_tripped: gru_failures += 1
                    if math_tripped and gru_tripped: both_failed += 1
                    
                    pbar.update(1)

    # Print Results
    print("\n" + "="*40)
    print(f"results (threshold: >{args.threshold}m error)")
    print("="*40)
    print(f"total test windows evaluated: {total_windows:,}")
    print("-" * 40)
    print(f"baseline failures:  {math_failures:,} ({(math_failures/total_windows)*100:.2f}%)")
    print(f"gru failures:     {gru_failures:,} ({(gru_failures/total_windows)*100:.2f}%)")
    print("-" * 40)
    print(f"situations where both tripped: {both_failed:,}")
    print(f"situations where gru saved the math: {math_failures - both_failed:,}")
    print(f"situations where gru ruined good math: {gru_failures - both_failed:,}")
    print("="*40)

if __name__ == "__main__":
    main()