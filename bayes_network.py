import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from utils import Gaussian, ScaleMixtureGaussian
import math
import torch.nn.functional as F
from bayes_layers import BayesConvLayer, BayesLinearLayer
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

class BayesLeNet(nn.Module):
    def __init__(self, num_classes, device = None):
        super(BayesLeNet, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.conv1 = BayesConvLayer(1, 6, 5, device = device)
        self.conv2 = BayesConvLayer(6, 16, 5, device = device)
        self.fc1 = BayesLinearLayer(16*5*5, 120, device = device)
        self.fc2 = BayesLinearLayer(120, 84, device = device)
        self.fc3 = BayesLinearLayer(84, 10, device = device)
    def forward(self, x, sample = False):
        x = F.relu(self.conv1(x, sample))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x,2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = self.fc3(x, sample)
        x = F.log_softmax(x, dim = 1)
        return x
    def log_prior(self):
        return self.conv1.log_prior \
               + self.conv2.log_prior \
               + self.fc1.log_prior \
               + self.fc2.log_prior \
               + self.fc3.log_prior
    def log_variational_posterior(self):
        return self.conv1.log_variational_posterior \
               + self.conv2.log_variational_posterior \
               + self.fc1.log_variational_posterior \
               + self.fc2.log_variational_posterior \
               + self.fc3.log_variational_posterior
    
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