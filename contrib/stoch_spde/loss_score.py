import torch

class loss_fn(torch.nn.Module):

    def __init__(self, model):
        """
        Args:
        model: A PyTorch model instance that represents a 
          time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of 
          the perturbation kernel.
        """
        super(loss_fn, self).__init__()
        self.model = model
        self.marginal_prob_std = model.marginal_prob_std

    def forward(self, x, eps=1e-5): 
        """The loss function for training score-based generative models.
        Args:
        x: A mini-batch of training data.    
        eps: A tolerance value for numerical stability.
        """

        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
