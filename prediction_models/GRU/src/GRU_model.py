import torch
import torch.nn as nn
from GRU_cell import GRUCell


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) network with an integrated Teacher Forcing (Spin-Up/Cut-Off) switch.

    This architecture dynamically expands the input feature space by 1 to accommodate a mode flag:
    - Spin-Up Mode (Flag 1.0): The network ingests historical telemetry data to evolve its hidden state.
    - Cut-Off Mode (Flag 0.0): Sensor inputs are frozen. The network uses its hidden state to predict
      future sequences recursively (Seq2Seq).

    Args:
        input_size (int): Number of raw input features (sensors) before the mode flag is appended.
        hidden_size (int): Number of features in the hidden state.
        output_size (int): Number of predicted output features.
        num_layers (int): Number of recurrent layers.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout)

        internal_input_size = input_size + 1  # adding the mode flag as a layer
        self.layers = nn.ModuleList()  # keeping the layers ar torch's ModuleList

        for i in range(num_layers):  # adding all the layers to the network
            in_size = internal_input_size if i == 0 else hidden_size
            self.layers.append(GRUCell(in_size, hidden_size))

        self.fc = nn.Linear(hidden_size, output_size)  # switching back to the output size

    def forward(self, x, h0=None, pred_len=0):
        batch_size, seq_len, _ = x.size()  # tensor sizes checks
        device = x.device  # applying best device

        if h0 is None:  # initializing the values in hidden layers
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h = [h0[i] for i in range(self.num_layers)]

            # setting the flags as const vectors
        flag_spinup = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        flag_cutoff = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

        for t in range(seq_len):  # encoder logic / spin-up - more details!
            current_input = torch.cat([x[:, t, :], flag_spinup], dim=-1)
            for layer in range(self.num_layers):
                h[layer] = self.layers[layer](current_input, h[layer])
                current_input = h[layer]

                if (
                    layer < self.num_layers - 1
                ):  # apply dropout between layers, but not after the last recurrent layer
                    current_input = self.dropout(current_input)

        outputs = []  # getting the array for the outpu
        decoder_input = x[:, -1, :]  # decoder input - last point from the encoder

        for _ in range(pred_len):  # decoder / cut-off
            # taking the cut part of the input for the decoder
            current_input = torch.cat([decoder_input, flag_cutoff], dim=-1)
            for layer in range(self.num_layers):
                # applying the forward on the layers
                h[layer] = self.layers[layer](current_input, h[layer])
                current_input = h[layer]  # saving the last step

                if (
                    layer < self.num_layers - 1
                ):  # apply dropout between layers, but not after the last recurrent layer
                    current_input = self.dropout(current_input)

            y = self.fc(current_input)  # going back in width to the output layer
            outputs.append(y)  # saving the state

        if pred_len > 0:  # pushing the outputs into one tensor
            outputs = torch.stack(outputs, dim=1)
        else:
            outputs = torch.empty((batch_size, 0, self.fc.out_features), device=device)

        hn = torch.stack(h, dim=0)

        return outputs, hn


class PersistenceGatedGRU(GRU):
    """GRU head for blending persistence with the learned RK4-residual forecast.

    The first three outputs are normalized RK4 residuals. The final three are
    per-axis gate logits. The loss/evaluator constructs the physical forecast as

        a_hat = a_last + sigmoid(gate) * (a_rk4_residual - a_last).

    A negative gate bias starts training near the persistence expert. This is the
    carry-biased initialization used by Highway-style gates: the learned branch
    must earn influence instead of overriding a useful baseline immediately.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=6,
        num_layers=1,
        dropout=0.0,
        gate_bias=-2.0,
    ):
        if output_size != 6:
            raise ValueError("PersistenceGatedGRU requires six outputs: 3 residuals + 3 gates.")
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)
        with torch.no_grad():
            self.fc.weight[3:].zero_()
            self.fc.bias[3:].fill_(gate_bias)


class LastAccelerationGatedGRU(PersistenceGatedGRU):
    """GRU head for gated corrections around the persistence baseline.

    The architecture is intentionally the same as PersistenceGatedGRU: three
    learned acceleration-delta channels and three gate logits. The difference is
    semantic and handled in the loss/evaluation code:

        a_hat = a_last + sigmoid(gate) * delta.

    Keeping the same head makes the ablation clean: the model changes its
    baseline assumption from RK4-residual correction to persistence correction.
    """
