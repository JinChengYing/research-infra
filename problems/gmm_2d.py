import torch
from torch.distributions import  MultivariateNormal
from infra.registry import register_problem
from abstractions.problem import Problem

@register_problem("gmm_2d")
class GMM2D(Problem):
    def __init__(self, cfg):
        self.n_components = cfg.get("n_components", 3)
        self.dim = cfg.get("dim", 2)
        self.means = torch.randn(self.n_components, self.dim)
        self.covs = torch.stack([torch.eye(self.dim) for _ in range(self.n_components)])
        self.components = [
            MultivariateNormal(self.means[i],self.covs[i])
            for i in range(self.n_components)
        ]
    def sample(self, n=1000):
        idx=torch.multinomial(self.weights,n,replacement=True)
        #comp_idx = torch.randint(0, self.n_components, (n,))
        samples= torch.stack([
            self.components[i].sample() for i in idx
        ])
        return samples

    def log_prob(self, x):
        """
        x:(N, dim)
        """
        log_probs=torch.stack([
            comp.log_prob(x) for comp in self.components
        ],dim=1)

        log_weights=torch.log(self.weights).unsqueeze(0)
        return torch.logsumexp(log_probs + log_weights, dim=1)