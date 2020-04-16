# directly taken from notebook, probably some adaptation might be needed
import torch
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np

from params import use_cuda, NUM_PITCHES, m_key_count


def ELBO_loss(y, t, mu, log_var, weight, with_logits = False):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    print("ELBO_LOSS")
    print(y.shape)
    print(torch.argmax(t, dim = -1)[0])
    print(torch.argmax(y, dim = -1)[0])
    ccd = torch.argmax(t, dim = -1) == torch.argmax(y, dim = -1)
    print(torch.mean(ccd.type(torch.float)))
    #clamp
    #y = torch.clamp(y, 0, 1)
    if with_logits == False:
        likelihood = -binary_cross_entropy(y, t, reduction="none")
    else:
        likelihood = -binary_cross_entropy_with_logits(y, t, reduction="none")

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

def ELBO_loss2(y, t, mu, log_var, weight, multi_notes = None):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    if np.random.rand() < 0.2:
        print("ELBO_LOSS2")
        print(y.shape)
        print(torch.argmax(t, dim = -1)[0])
        print(torch.argmax(y, dim = -1)[0])
        ccd = torch.argmax(t, dim = -1) == torch.argmax(y, dim = -1)
        print(torch.mean(ccd.type(torch.float)))

        if multi_notes != None:
            g1_matrix = multi_notes[:,:,0,:]
            g2_matrix = multi_notes[:, :, 1, :]
            cce = torch.argmax(g1_matrix, dim = -1) == torch.argmax(g2_matrix, dim = -1)
            print("g1 g2", torch.sum((g1_matrix - g2_matrix)**2))
            print("g1 g2 overlap rate",torch.mean(cce.type(torch.float)))

    #clamp
    #y = torch.clamp(y, 0, 1)
    y_reshape = y.view(-1, NUM_PITCHES) + 1e-6
    t_reshape = t.view(-1, NUM_PITCHES)

    #focal loss
    w1 = (1 - y_reshape)**2
    focal = w1 * torch.log(y_reshape)
    likelihood = torch.sum(t_reshape * focal, dim=0)


    w = torch.sum(t_reshape, dim = 0) / torch.sum(t)
    #
    likelihood = likelihood * (1-w)**10
    #
    # likelihood = t_reshape * torch.log(y_reshape) + (1 - t_reshape) * torch.log(1 - y_reshape)
    #
    # likelihood = torch.sum(likelihood, dim=0)
    #
    # print("likelihood:", likelihood)
    #
    # likelihood = likelihood * (1 - w)**12

    #print("balance likelihood:", likelihood)

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
    ELBO = torch.sum(likelihood) - (weight * torch.mean(kl_div))  # add a weight to the kl using warmup

    divergence_loss = 0.0
    if multi_notes != None:
        divergence_loss +=  0.001 * torch.sum((multi_notes[:,:,0,:] - multi_notes[:,:,1,:])**2)


    # notice minus sign as we want to maximise ELBO
    return -ELBO - divergence_loss, kl_div.mean(), weight * kl_div.mean()  # mean instead of sum


def ELBO_loss_Multi(multi_notes, t, mu, log_var, weight):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    print_or_not = np.random.rand() < 0.2

    t = t.view(-1, NUM_PITCHES)
    t_index = torch.arange(t.size(0), requires_grad=False)
    multi_notes = multi_notes.view(-1, m_key_count, NUM_PITCHES)+ 1e-6
    key_counts = torch.sum(t, dim = -1)

    if print_or_not:
        print("ELBO_LOSS_MULTI")
        print(torch.sum(t.view(-1, NUM_PITCHES), dim = -1)[:100])
        print(torch.argmax(t, dim = -1)[0:100])
        print(torch.argmax(multi_notes[:,0,:], dim = -1)[0:100])
        print(torch.argmax(multi_notes[:,1,:], dim=-1)[0:100])
        # ccd = torch.argmax(t, dim = -1) == torch.argmax(y, dim = -1)
        # print(torch.mean(ccd.type(torch.float)))
        t_hat = torch.zeros_like(t)
        if multi_notes != None:
            g1_matrix = multi_notes[:,0,:]
            g2_matrix = multi_notes[:, 1, :]
            cce = torch.argmax(g1_matrix, dim = -1) == torch.argmax(g2_matrix, dim = -1)
            print("g1 g2", torch.sum((g1_matrix - g2_matrix)**2))
            print("g1 g2 overlap rate",torch.sum(cce.type(torch.float))/ torch.sum(key_counts))

            for j in range(m_key_count):
                multi_notes_j = multi_notes[:, j, :]
                max_index = torch.argmax(multi_notes_j, dim=-1)
                t_hat[t_index, max_index] = 1

            ccf = t_hat[t == 1] - t[t == 1]
            print("t_hat t miss overlap rate: ", torch.mean(ccf.type(torch.float)**2))

    likelihood = 0.0

    # fill key
    t = t.clone()
    t[key_counts < m_key_count, 60] += m_key_count - key_counts[key_counts < m_key_count]

    #print("after fill t", torch.sum(t, dim = -1)[:100])
    w = torch.sum(t, dim=0) / torch.sum(t)
    w = (-w + 1.0)**10
    for j in range(m_key_count):
        multi_notes_j = multi_notes[:,j,:]
        #if print_or_not:
        #    print("multi_notes",j, multi_notes_j[0])
        # focal loss
        w1 = (1 - multi_notes_j) ** 2
        focal = w1 * torch.log(multi_notes_j)
        likelihood += torch.sum(t * focal, dim=0) * w

        #modify t
        max_index = torch.argmax(multi_notes_j, dim = -1)

        #print("max_index", max_index.requires_grad, max_index.shape)
        t = t.clone()
        t[t_index, max_index] = -0.25

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
    ELBO = torch.sum(likelihood) - (weight * torch.mean(kl_div))  # add a weight to the kl using warmup

    # divergence_loss = 0.0
    # if multi_notes != None:
    #     divergence_loss +=  0.01 * torch.sum((multi_notes[:,0,:] - multi_notes[:,1,:])**2)

    # notice minus sign as we want to maximise ELBO
    return -ELBO, kl_div.mean(), weight * kl_div.mean()  # mean instead of sum