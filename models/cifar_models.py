import sys
import torch
from models.ops import BayesConvLayer, BayesLinearLayer
import torch.nn.functional as F
import torch.nn as nn


class BayesCifarNet(nn.Module):

    def __init__(self, num_classes, device=None):
        super(BayesCifarNet, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.conv1 = BayesConvLayer(3, 64, 3, padding=1, device=device)
        self.conv2 = BayesConvLayer(64, 64, 3, padding=1, device=device)
        self.conv3 = BayesConvLayer(64, 64, 3, padding=1, device=device)
        self.conv4 = BayesConvLayer(64, 64, 3, padding=1, device=device)
        self.conv5 = BayesConvLayer(64, 64, 3, padding=1, device=device)
        self.conv6 = BayesConvLayer(64, 64, 1, padding=0, device=device)
        self.conv7 = BayesConvLayer(64, 16, 1, padding=0, device=device)
        self.fc1 = BayesLinearLayer(1024, num_classes, device=device)

    def log_prior(self):
        log_prior = 0
        for i in range(7):
            log_prior += self.__getattr__('conv%d'%(i+1)).log_prior
        log_prior += self.fc1.log_prior
        return log_prior

    def log_variational_posterior(self):
        posterior = 0
        for i in range(7):
            posterior += self.__getattr__('conv%d'%(i+1)).log_variational_posterior
        posterior += self.fc1.log_variational_posterior
        return posterior

    def sample_elbo(self, input, target, samples=2):
        batch_size = input.size(0)
        outputs = torch.zeros(samples, batch_size, self.num_classes).to(device=self.device)
        log_priors = torch.zeros(samples).to(device=self.device)
        log_variational_posteriors = torch.zeros(samples).to(device=self.device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / batch_size + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

    def forward(self, x, sample=False):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = x.view(x.size(0), -1)
        x = self.fc1(x, sample)
        x = F.log_softmax(x, dim=1)
        return x


class BayesLeNetCifar(nn.Module):
    def __init__(self, num_classes, device=None):
        super(BayesLeNetCifar, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.conv1 = BayesConvLayer(3, 6, 5, padding=0, device=device)
        self.conv2 = BayesConvLayer(6, 16, 5, padding=0, device=device)
        self.fc1 = BayesLinearLayer(5*5*16, 120, device=device)
        self.fc2 = BayesLinearLayer(120, 84, device=device)
        self.fc3 = BayesLinearLayer(84, num_classes, device=device)

    def forward(self, x, sample=False):
        x = F.relu(self.conv1(x, sample))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, sample))
        x = self.fc2(x, sample)
        x = F.relu(x)
        x = self.fc3(x, sample)
        x = F.log_softmax(x, dim=1)
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

    def sample_elbo(self, input, target, samples=5):
        batch_size = input.size(0)
        outputs = torch.zeros(samples, batch_size, self.num_classes).to(device=self.device)
        log_priors = torch.zeros(samples).to(device=self.device)
        log_variational_posteriors = torch.zeros(samples).to(device=self.device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / batch_size + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

