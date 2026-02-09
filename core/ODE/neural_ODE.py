import torch
import torch.nn as nn
from torchdiffeq import odeint
from core.registry.ode import register_ode

class ODEFunc(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t, x):
        return self.net(x)

@register_ode("neural_ode")
class NeuralODE(nn.Module):
    def __init__(self, dim, hidden_dim, t0, t1, solver):
        """
        参数完全来自 config.yaml
        """
        super().__init__()
        self.func = ODEFunc(dim=dim, hidden_dim=hidden_dim)
        self.t0 = t0
        self.t1 = t1
        self.solver = solver

    def forward(self, x):
        t = torch.tensor([self.t0, self.t1]).to(x)
        out = odeint(self.func, x, t, method=self.solver)
        return out[-1]

    @torch.no_grad()
    def trajectory(self, x0, steps=50):
        t = torch.linspace(self.t0, self.t1, steps).to(x0)
        traj = odeint(self.func, x0, t, method=self.solver)
        return traj
