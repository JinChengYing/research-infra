import matplotlib.pyplot as plt
import torch

def plot_samples(x, title):
    x = x.numpy()
    plt.figure(figsize=(4,4))
    plt.scatter(x[:,0], x[:,1], s=5, alpha=0.6)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)

def plot_trajectories(traj):
    """
    traj: (T, N, 2)
    """
    traj = traj.numpy()
    plt.figure(figsize=(4,4))
    for i in range(traj.shape[1]):
        plt.plot(traj[:,i,0], traj[:,i,1], alpha=0.3)
    plt.title("Neural ODE Flow Trajectories")
    plt.axis("equal")
    plt.grid(True)
