import yaml
from infra.registry import METHODS, PROBLEMS
from plot import plot_samples, plot_trajectories
import matplotlib.pyplot as plt
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 构建 Problem
problem_cls = PROBLEMS[cfg["problem"]["name"]]
problem = problem_cls(cfg["problem"].get("params", {}))

# 构建 Method / Trainer
method_cls = METHODS[cfg["method"]["name"]]
trainer = method_cls(cfg["method"]["params"])

# 训练
trainer.train(problem,
              epochs=cfg["method"]["params"].get("epochs", 100),
              batch_size=cfg["method"]["params"].get("batch_size", 256))
# 真实 GMM
x_real = problem.sample(2000)
plot_samples(x_real, "Ground Truth GMM")

# 生成样本
x_gen = trainer.sample(2000)
plot_samples(x_gen, "Neural ODE Generated")

# ODE trajectories
z0 = torch.randn(50, 2).to(trainer.device)
traj = trainer.model.trajectory(z0)
plot_trajectories(traj.cpu())

plt.show()