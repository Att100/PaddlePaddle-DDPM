import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SimpleNet(nn.Layer):
    def __init__(self, num_steps=100, n_dim=128):
        super().__init__()

        self.lin1 = nn.Linear(2, n_dim)
        self.lin2 = nn.Linear(n_dim, n_dim)
        self.lin3 = nn.Linear(n_dim, n_dim)
        self.lin4 = nn.Linear(n_dim, 2)
        self.time_embedding = nn.Embedding(num_steps, n_dim)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        out = F.relu(self.lin1(x)+temb)
        out = F.relu(self.lin2(out)+temb)
        out = F.relu(self.lin3(out)+temb)
        return self.lin4(out)
