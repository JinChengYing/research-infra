from core.registry.loss import register_loss

@register_loss("gmm_mle")
class GMM_MLE_Loss:
    def __call__(self, x_generated, problem):
        """
        maximize log p_GMM(x)
        """
        logp = problem.log_prob(x_generated)
        return -logp.mean()
