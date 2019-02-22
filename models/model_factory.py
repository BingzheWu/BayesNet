import sys
sys.path.append('.')
from models.point_estimation_models import LeNet, CifarNet, CifarLeNet
from models.minist_models import BayesLeNet
from models.cifar_models import BayesCifarNet, BayesLeNetCifar
from models.alexnet import AlexNet, SquareAlexNet
from models.bayes_alexnet import BayesAlexNet
import torch
from models.lenet import SquareLeNet, CryptoNet


def model_creator(cfg, device='cuda'):
    if cfg.arch == 'LeNet':
        model = LeNet()
    if cfg.arch == 'BayesLeNet':
        model = BayesLeNet(cfg.num_classes, device)
    if cfg.arch == 'BayesCifarNet':
        model = BayesCifarNet(cfg.num_classes, device)
    if cfg.arch == 'BayesLeNetCifar':
        model = BayesLeNetCifar(cfg.num_classes, device)
    if cfg.arch == 'CifarNet':
        model = CifarNet(cfg.num_classes)
    if cfg.arch == 'CifarLeNet':
        model = CifarLeNet()
    if cfg.arch == 'AlexNet':
        model = AlexNet(cfg.num_classes)
    if cfg.arch == 'BayesAlexNet':
        model = BayesAlexNet(cfg.num_classes, device=device)
    if cfg.arch == 'SquareLeNet':
        model = SquareLeNet()
    if cfg.arch == 'SquareAlexNet':
        model = SquareAlexNet(cfg.num_classes)
    if cfg.arch == 'CryptoNet':
        model = CryptoNet()
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint['state_dict'])
    return model

