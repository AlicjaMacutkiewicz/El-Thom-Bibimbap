import torch
import torch.nn as nn
from GRU_cell import Gru_Cell as GRUCell

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # layers list GRUCell
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(GRUCell(in_size, hidden_size))

        # warstwa liniowa na wyjściu
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        
       # x : (batch_size, seq_len, input_size)
       # h0: (num_layers, batch_size, hidden_size)

        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h = [h0[i] for i in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):

            layer_input = x[:, t, :]

            for layer in range(self.num_layers):
#h_t = GRUCell(x_t, h_prev)
                h[layer] = self.layers[layer](layer_input, h[layer])

                layer_input = h[layer]

            # warstwa liniowa
            y = self.fc(layer_input)

            outputs.append(y)

        # składanie wyników
        outputs = torch.stack(outputs, dim=1)

        # końcowe stany ukryte
        hn = torch.stack(h, dim=0)

        return outputs, hn