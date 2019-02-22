import torch.nn as nn
import torch
import sys
sys.path.append('.')
from models.ops import BayesLinearLayer, BayesConvLayer
import torch.nn.functional as F


class BayesAlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, num_classes, device=None):
        super(BayesAlexNet, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.conv1 = BayesConvLayer(3, 64, kernel_size=11, stride=4, padding=5, device=device)
        self.soft1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BayesConvLayer(64, 192, kernel_size=5, padding=2, device=device)
        self.soft2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BayesConvLayer(192, 384, kernel_size=3, padding=1, device=device)
        self.soft3 = nn.ReLU()

        self.conv4 = BayesConvLayer(384, 256, kernel_size=3, padding=1, device=device)
        self.soft4 = nn.ReLU()

        self.conv5 = BayesConvLayer(256, 128, kernel_size=3, padding=1, device=device)
        self.soft5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = BayesLinearLayer(1* 1 * 128, num_classes, device=device)
        features = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                    self.conv4, self.soft4, self.conv5, self.soft5, self.pool3]
        classifier = [self.fc1]
        self.features = nn.ModuleList(features)
        self.classifier = nn.ModuleList(classifier)

    def forward(self, x, sample=False):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.features:
            if hasattr(layer, 'bayes_forward'):
                x = layer.bayes_forward(x, sample)
            else:
                x = layer(x)
        x = x.view(-1, 128)
        for layer in self.classifier:
            if hasattr(layer, 'bayes_forward'):
                x = layer.bayes_forward(x, sample)
        x = F.log_softmax(x, dim=1)
        return x

    def log_prior(self):
        log_prior = 0
        for layer in self.features:
            if hasattr(layer, 'bayes_forward'):
                log_prior += layer.log_prior
        for layer in self.classifier:
            if hasattr(layer, 'bayes_forward'):
                log_prior += layer.log_prior
        return log_prior

    def log_variational_posterior(self):
        posterior = 0
        for layer in self.features:
            if hasattr(layer, 'bayes_forward'):
                posterior += layer.log_variational_posterior
        for layer in self.classifier:
            if hasattr(layer, 'bayes_forward'):
                posterior += layer.log_variational_posterior
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


def test():
    x = torch.ones((1, 3, 32, 32))
    net = BayesAlexNet(num_classes=10)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()