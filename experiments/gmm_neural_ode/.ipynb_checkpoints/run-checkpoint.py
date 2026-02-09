import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt

# 确保 repo 根目录在 sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from infra.registry import PROBLEMS, METHODS
import problems.gmm_2d
import methods.neural_ode_trainer
from experiments.gmm_neural_ode.showplot import plot_samples, plot_trajectories

# 读取配置
cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# 实例化 problem
problem_cls = PROBLEMS[cfg["problem"]["name"]]
problem = problem_cls(cfg["problem"].get("params", {}))

# 实例化 trainer
method_cls = METHODS[cfg["method"]["name"]]
trainer = method_cls(cfg["method"]["params"])

# 训练
trainer.run(problem)

# 可视化
x_real = problem.sample(2000)
plot_samples(x_real, "Ground Truth GMM")

x_gen = trainer.sample(2000)
plot_samples(x_gen, "Neural ODE Generated")

z0 = torch.randn(50, 2).to(trainer.device)
traj = trainer.model.trajectory(z0)
plot_trajectories(traj.cpu())

plt.show()

