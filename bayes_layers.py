import torch
import math
import  torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from utils import Gaussian, ScaleMixtureGaussian
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
class BayesConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, 
                padding = 0, dilation = 1, groups = 1, bias = True, device = None):
            super(BayesConvLayer, self).__init__()
            SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(device)
            SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(device)
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(stride)
            dilation = _pair(dilation)
            self.device = device
            self.kernel_size = kernel_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.padding = padding
            self.dilation = dilation
            self.stride = stride
            self.groups = groups
            self.bias = bias
            self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.weight = Gaussian(self.weight_mu, self.weight_rho, device = self.device)
            self.weight_prior = ScaleMixtureGaussian(0.5, SIGMA_1, SIGMA_2)
            if bias:
                self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
                self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
                self.bias = Gaussian(self.bias_mu, self.bias_rho, device = self.device)
                self.bias_prior = ScaleMixtureGaussian(0.5, SIGMA_1, SIGMA_2)
            self.log_prior = 0
            self.log_variational_posterior = 0
            self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.uniform_(-5, -4)
        if self.bias is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_rho.data.uniform_(-5, -4)
    def forward(self, input, sample = True, calculate_log_probs = False):
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
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
def test_bayes_conv():
    device = 'cpu'
    test_conv =  BayesConvLayer(1, 6, 5, padding=1, stride = 2, device = device)
    test_conv.to(device)
    x = torch.zeros((1, 1, 28, 28)).to(device)
    y = test_conv(x, sample = True, calculate_log_probs = True)
    print(y.size())

if __name__ == '__main__':
    test_bayes_conv()


        
    