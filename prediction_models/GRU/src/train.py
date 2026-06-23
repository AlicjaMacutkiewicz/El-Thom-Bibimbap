import torch
import wandb


def train_model(
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
    pred_len,
    device,
    batch_size=64,
    training_rounds=10,
    year="all",
):
    """
    Executes the training loop for the GRU model, integrating Weights & Biases for live telemetry.

    Iterates through the dataset in mini-batches, computes predictions, evaluates the
    Physics-Informed Neural Network (PINN) loss, and performs backpropagation to update weights.
    Saves a model checkpoint every 5 epochs.

    Args:
        model (nn.Module): The GRU neural network.
        X_train ... initial_time_train (torch.Tensor): Training data tensors.
        X_test ... initial_time_test (torch.Tensor): Testing data tensors.
        loss (nn.Module): The composite PINN loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        pred_len (int): The number of future time steps to predict.
        device (torch.device): The hardware accelerator to use.
        batch_size (int, optional): Number of sequences per batch. Defaults to 64.
        training_rounds (int, optional): Total number of epochs. Defaults to 10.
        year (str, optional): Tag for Weights & Biases logging. Defaults to "all".

    Returns:
        tuple: Two lists containing the average training and testing loss per epoch.
    """

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")

    # initialize wandb session
    wandb.init(
        project="rocket-trajectory",
        name=f"gru_seq{pred_len}_yr{year}",
        config={"batch_size": batch_size, "epochs": training_rounds, "seq_len": pred_len},
    )

    # automatically track model weights and gradients
    wandb.watch(model, log="all")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    for training_round in range(training_rounds):
        round_loss = 0.0
        total_samples = 0

        model.train()

        for i in range(0, len(X_train), batch_size):
            # Load batch slices to the active device
            X_batch = X_train[i : i + batch_size].to(device)
            condition_batch = condition_train[i : i + batch_size].to(device)
            y_batch = y_train[i : i + batch_size].to(device)
            pos_batch = pos_train[i : i + batch_size].to(device)
            t_batch = t_train[i : i + batch_size].to(device)
            initial_pos_batch = initial_pos_train[i : i + batch_size].to(device)
            initial_vel_batch = initial_vel_train[i : i + batch_size].to(device)
            initial_time_batch = initial_time_train[i : i + batch_size].to(device)

            # pass input sequence through the GRU
            preds, _ = model(X_batch, pred_len=pred_len)

            # GRU outputs the predicted residual (x_s)
            # PINN loss adds base physics (x_b), integrates, and compares position
            batch_loss = loss(
                preds,
                y_batch,
                pos_batch,
                t_batch,
                initial_pos_batch,
                initial_vel_batch,
                initial_time_batch,
                condition_batch,
            )

            # backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_samples = len(X_batch)
            round_loss += batch_loss.item() * batch_samples
            total_samples += batch_samples

        # calcutate average loss over all training batches in this epoch
        avg_loss = round_loss / total_samples
        train_losses.append(avg_loss)

        # calcutate average loss over all test batches in this epoch
        avg_test_loss = evaluate_model(
            model,
            X_test,
            condition_test,
            y_test,
            pos_test,
            t_test,
            initial_pos_test,
            initial_vel_test,
            initial_time_test,
            loss,
            device,
            batch_size,
            pred_len,
        )
        test_losses.append(avg_test_loss)

        print(
            f"Iteration {training_round + 1}, train loss: {avg_loss:.8e}, test loss: {avg_test_loss:.8e}"
        )

        # tell the scheduler to check the test loss and decide if it needs to slow down
        scheduler.step(avg_test_loss)

        # log metrics to wandb dashboard
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "Train Loss": avg_loss,
                "Test Loss": avg_test_loss,
                "Learning Rate": current_lr,
                "Epoch": training_round + 1,
            }
        )

        # checkpoint saving
        if (training_round + 1) % 5 == 0:
            checkpoint_filename = f"gru_checkpoint_round_{training_round + 1}_seq{pred_len}.pth"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"checkpoint: {checkpoint_filename}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_checkpoint_filename = f"best_gru_model_seq{pred_len}.pth"
            torch.save(model.state_dict(), best_checkpoint_filename)
            print(f"best checkpoint: {best_checkpoint_filename} (test loss: {best_test_loss:.8e})")

    wandb.finish()
    return train_losses, test_losses


def evaluate_model(
    model,
    X_test,
    condition_test,
    y_test,
    pos_test,
    t_test,
    initial_pos_test,
    initial_vel_test,
    initial_time_test,
    loss,
    device,
    batch_size=64,
    pred_len=3,
):
    """
    Evaluates the model's performance on the test dataset without updating weights.

    Args:
        model (nn.Module): The GRU neural network.
        X_test ... initial_time_test (torch.Tensor): Testing data tensors.
        loss (nn.Module): The composite PINN loss function.
        device (torch.device): The hardware accelerator to use.
        batch_size (int, optional): Number of sequences per batch. Defaults to 64.
        pred_len (int, optional): The number of future time steps to predict. Defaults to 3.

    Returns:
        float: The average loss across all test batches.
    """

    model.eval()
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i : i + batch_size].to(device)
            condition_batch = condition_test[i : i + batch_size].to(device)
            y_batch = y_test[i : i + batch_size].to(device)
            pos_batch = pos_test[i : i + batch_size].to(device)
            t_batch = t_test[i : i + batch_size].to(device)
            initial_pos_batch = initial_pos_test[i : i + batch_size].to(device)
            initial_vel_batch = initial_vel_test[i : i + batch_size].to(device)
            initial_time_batch = initial_time_test[i : i + batch_size].to(device)

            # generate predictions
            preds, _ = model(X_batch, pred_len=pred_len)

            # compute pinn loss
            curr_loss = loss(
                preds,
                y_batch,
                pos_batch,
                t_batch,
                initial_pos_batch,
                initial_vel_batch,
                initial_time_batch,
                condition_batch,
            )

            batch_samples = len(X_batch)
            test_loss += curr_loss.item() * batch_samples
            total_samples += batch_samples

    return test_loss / total_samples
