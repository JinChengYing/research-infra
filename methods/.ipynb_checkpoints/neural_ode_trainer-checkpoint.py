import torch
from torch import nn, optim
from infra.registry import register_method
from abstractions.method import Method
from core.registry.ODE import build_ode
from core.registry.loss import build_loss
from core.ODE.neural_ODE import NeuralODE  

@register_method("neural_ode_trainer")
class NeuralODETrainer(Method):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #model_cfg = cfg["model"]
        self.model = build_ode(self.cfg["model"]).to(self.device)
        self.loss_fn = build_loss(self.cfg["loss"])
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=cfg.get("optimizer", {}).get("lr", 1e-3))
        self.dim=cfg["model"]["params"]["dim"]
    def run(self, problem):
        # 样例训练循环
        for step in range(self.cfg["train"]["steps"]):
            x = problem.sample(self.cfg["train"]["batch_size"])
            x_pred = self.model(x)
            loss = self.loss_fn(x_pred, x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

    def train(self, problem):
        for epoch in range(self.cfg["epochs"]):
            z=torch.randn(self.cfg["batch_size"],self.dim).to(self.device)

            self.optimizer.zero_grad()
            x_gen = self.model(z)
            loss = self.loss_fn(x_gen, problem)
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")

    @torch.no_grad()
    def sample(self, n=2000):
        z = torch.randn(n, self.dim).to(self.device)
        x = self.model(z)
        return x.cpu()

