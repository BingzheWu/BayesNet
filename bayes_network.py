import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from utils import Gaussian, ScaleMixtureGaussian
import math
import torch.nn.functional as F
class BayesLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device = None):
        super().__init__()
        SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
        SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, device = self.device)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, device = self.device)
        self.weight_prior = ScaleMixtureGaussian(0.5, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(0.5, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample = False, calculate_log_probs = False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            #self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        return F.linear(input, weight, bias)
class BayesMLP(nn.Module):
    def __init__(self, num_classes = 10, device = None):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.l1 = BayesLinearLayer(28*28, 400, device)
        self.l2 = BayesLinearLayer(400, 400, device)
        self.l3 = BayesLinearLayer(400, 10, device)
    def forward(self, x, sample = False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim = 1)
        return x
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=2):
        batch_size = input.size(0)
        outputs = torch.zeros(samples, batch_size, self.num_classes).to(device = self.device)
        log_priors = torch.zeros(samples).to(device = self.device)
        log_variational_posteriors = torch.zeros(samples).to(device = self.device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior)/batch_size + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


        