import torch as tr

# note:
# @ - matmul (matrix)
# * - mul (element wise)

class Gru_Cell(tr.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Parameters meaning:
        # W - first weigh vector 
        # U - second weigh vector
        # (W @ U = weigh matrix)
        # b - bias

        # Reset gate parameters:
        self.W_r = tr.nn.Parameter(tr.randn(input_size,hidden_size))
        self.U_r = tr.nn.Parameter(tr.randn(hidden_size,hidden_size))
        self.b_r = tr.nn.Parameter(tr.zeros(hidden_size))

        # Update gate parameters:
        self.W_z = tr.nn.Parameter(tr.randn(input_size,hidden_size))
        self.U_z = tr.nn.Parameter(tr.randn(hidden_size,hidden_size))
        self.b_z = tr.nn.Parameter(tr.zeros(hidden_size))

        # Parameters linking the two layers:
        self.W = tr.nn.Parameter(tr.randn(input_size,hidden_size))
        self.U = tr.nn.Parameter(tr.randn(hidden_size,hidden_size))
        self.b = tr.nn.Parameter(tr.zeros(hidden_size))
    
    def get_reset_gate(self, x, h_prev):
        # r(t)  = sigma_function(W_r*x(t) + U_r*h(t-1) + b_r)
        return tr.sigmoid((x @ self.W_r) + (h_prev @ self.U_r) + self.b_r)

    def get_update_gate(self, x, h_prev):
        # z(t)  = sigma_function(W_z*x(t) + U_z*h(t-1) + b_z)
        return tr.sigmoid((x @ self.W_z) + (h_prev @ self.U_z) + self.b_z)
    
    def get_candidate_gate(self, x, h_prev):
        # h'(t) = tanh(U*r(t)*h(t-1) + W*x(t) + b_h')
        r = self.get_reset_gate(x, h_prev)
        return tr.tanh(((r * h_prev) @ self.U) + (x @ self.W) + self.b)
    
    def forward(self, x, h_prev):
        # h(t)  = (1-z(t))*h'(t) + h(t-1)*z(t)
        z = self.get_update_gate(x, h_prev)
        h_c = self.get_candidate_gate(x, h_prev)
        return (((1-z) * h_c) + (h_prev) * z)
    
