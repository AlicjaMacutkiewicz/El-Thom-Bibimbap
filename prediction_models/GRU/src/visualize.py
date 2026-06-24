import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from physics import calculate_x_b_conditioned


def _axis_metrics(prediction, target, axis):
    error = prediction[:, axis] - target[:, axis]
    return {
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error**2)),
        "bias": np.mean(error),
    }


def _format_metrics(name, metrics):
    return (
        f"{name}: MAE={metrics['mae']:.4g}, RMSE={metrics['rmse']:.4g}, Bias={metrics['bias']:.4g}"
    )


def plot_prediction(
    model,
    X_test,
    y_hist_test,
    y_test,
    t_test,
    pred_len,
    parameters,
    thrust_curve,
    mean_xs,
    std_xs,
    mean_acc,
    std_acc,
    device,
    sampling_rate,
    sample_idx=0,
    axis=0,
    filename_prefix="prediction_sample",
    condition_test=None,
    model_type="gru_res_phys",
):
    """
    Generates and saves a plot comparing the model's trajectory prediction against the ground truth.

    This function extracts a single sequence, passes it through the GRU to predict the
    nonlinear acceleration residual, adds back the deterministic RK4 baseline (x_b),
    and plots the fully reconstructed physical acceleration.

    Args:
        model (nn.Module): The trained GRU model.
        X_test (torch.Tensor): Tensor of historical input sequences.
        y_test (torch.Tensor): Tensor of target future sequences.
        t_test (torch.Tensor): Tensor of timestamps corresponding to the targets.
        pred_len (int): Number of future steps the model predicts.
        parameters (dict): Physical rocket parameters for RK4 integration.
        thrust_curve (np.ndarray): Engine thrust data.
        mean_acc (np.ndarray): Mean values for acceleration denormalization.
        std_acc (np.ndarray): Standard deviation for acceleration denormalization.
        device (torch.device): The hardware accelerator to use.
        sample_idx (int, optional): Index of the test sample to visualize. Defaults to 0.
        axis (int, optional): Spatial axis to plot (0=X, 1=Y, 2=Z). Defaults to 0.
    """

    model.eval()

    with torch.no_grad():
        # select a single test sample (batch size = 1)
        input_seq = X_test[sample_idx : sample_idx + 1].to(device)
        target = y_test[sample_idx]
        target_times = t_test[sample_idx : sample_idx + 1].to(device)
        condition = None
        if condition_test is not None:
            condition = condition_test[sample_idx : sample_idx + 1]

        # forward pass through the GRU to get the predicted residual (x_s)
        raw_prediction, _ = model(input_seq, pred_len=pred_len)

        # GRU output is a normalized residual — denormalize with residual stats
        predicted_x_s_denorm = raw_prediction[0, :, :3].cpu().numpy() * std_xs + mean_xs

        # Targets/history use the active model target scaler supplied as mean_xs/std_xs.
        target_denorm = target.cpu().numpy() * std_xs + mean_xs
        history_denorm = y_hist_test[sample_idx].cpu().numpy() * std_xs + mean_xs

        # calculate and add the known physics baseline (x_b)
        base_acc = calculate_x_b_conditioned(
            target_times, parameters, thrust_curve, sampling_rate, condition
        )[0]
        learned_candidate = predicted_x_s_denorm + base_acc
        if model_type == "gru_res_phys_persist_gate":
            gate = torch.sigmoid(raw_prediction[0, :, 3:6]).cpu().numpy()
            anchor = history_denorm[-1:, :3]
            prediction = anchor + gate * (learned_candidate - anchor)
        else:
            prediction = learned_candidate
        rk4_baseline = base_acc
        last_value_baseline = np.repeat(history_denorm[-1:, :3], pred_len, axis=0)
        residual_target = target_denorm - base_acc
        gru_metrics = _axis_metrics(prediction, target_denorm, axis)
        rk4_metrics = _axis_metrics(rk4_baseline, target_denorm, axis)
        last_value_metrics = _axis_metrics(last_value_baseline, target_denorm, axis)

    # define time axes for past (input) and future (prediction)
    seq_len = input_seq.shape[1]
    past_time = np.arange(seq_len)
    future_time = np.arange(seq_len, seq_len + pred_len)

    # plotting
    plt.figure(figsize=(10, 5))
    axes_labels = ["X", "Y", "Z"]

    plt.plot(past_time, history_denorm[:, axis], label="History (Target)", color="blue", marker="o")
    plt.plot(
        future_time,
        target_denorm[:, axis],
        label="Ground Truth (Target)",
        color="green",
        marker="s",
    )
    plt.plot(
        future_time,
        prediction[:, axis],
        label="Prediction",
        color="red",
        linestyle="--",
        marker="x",
    )
    plt.plot(
        future_time,
        rk4_baseline[:, axis],
        label="RK4 Baseline",
        color="purple",
        linestyle=":",
    )
    plt.plot(
        future_time,
        last_value_baseline[:, axis],
        label="Last-Value Baseline",
        color="gray",
        linestyle="-.",
    )
    plt.title(
        f"{axes_labels[axis]}-Axis Acceleration Prediction (Sample {sample_idx}) "
        f"| GRU RMSE={gru_metrics['rmse']:.4g}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{filename_prefix}_{sample_idx}_axis{axis}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot to {filename}")
    print(_format_metrics("GRU", gru_metrics))
    print(_format_metrics("RK4 baseline", rk4_metrics))
    print(_format_metrics("Last-value baseline", last_value_metrics))

    plt.figure(figsize=(10, 5))
    plt.plot(
        future_time,
        residual_target[:, axis],
        label="Ground Truth Residual",
        color="green",
        marker="s",
    )
    plt.plot(
        future_time,
        predicted_x_s_denorm[:, axis],
        label="Predicted Residual",
        color="red",
        linestyle="--",
        marker="x",
    )
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.3)
    plt.title(f"{axes_labels[axis]}-Axis Residual Prediction (Sample {sample_idx})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    residual_filename = f"{filename_prefix}_residual_{sample_idx}_axis{axis}.png"
    plt.savefig(residual_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved residual prediction plot to {residual_filename}")


def plot_losses(train_losses, test_losses):
    """
    Generates and saves a plot of the training and testing loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(test_losses, label="Testing Loss", color="orange")
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Loss (MSE, log scale)")
    plt.title("Model Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("loss_progress.png", bbox_inches="tight")
    plt.close()

    print("Saved loss plot to loss_progress.png")
