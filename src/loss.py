# directly taken from notebook, probably some adaptation might be needed
import torch
from torch.nn.functional import binary_cross_entropy
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from params import use_cuda


def ELBO_loss(y, t, mu, log_var, weight):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(y, t, reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Regularization error:
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    sigma = torch.exp(log_var * 2)
    n_mu = torch.Tensor([0])
    n_sigma = torch.Tensor([1])
    if use_cuda:
        n_mu = n_mu.cuda()
        n_sigma = n_sigma.cuda()

    p = Normal(n_mu, n_sigma)
    q = Normal(mu, sigma)

    # The method signature is P and Q, but might need to be reversed to calculate divergence of Q with respect to P
    kl_div = kl_divergence(q, p)

    # In the case of the KL-divergence between diagonal covariance Gaussian and
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    # kl = -weight * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=(1,2))

    # Combining the two terms in the evidence lower bound objective (ELBO)
    # mean over batch
    ELBO = torch.mean(likelihood) - (weight * torch.mean(kl_div))  # add a weight to the kl using warmup

    # notice minus sign as we want to maximise ELBO
    return -ELBO, kl_div.mean(), weight * kl_div.mean()  # mean instead of sum

