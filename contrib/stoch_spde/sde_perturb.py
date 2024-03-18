#@title Set up the SDE

import torch
import numpy as np
import functools

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

class marginal_prob_std(torch.nn.Module):

    def __init__(self, sigma):
        """
        Args:
        sigma: The $\sigma$ in our SDE.
        """
        super(marginal_prob_std, self).__init__()
        self.sigma = sigma

    def forward(self,t):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
        Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
        Returns:
        The standard deviation
        """
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  
sigma =  25.0#@param {'type':'number'}
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
