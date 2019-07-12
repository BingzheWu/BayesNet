import torch
import torch.nn.functional as F
from collections import OrderedDict
from argparse import ArgumentParser
import logging
from mmcv import Config
import datetime
from mmcv.runner import Runner
from mmcv.runner.hooks import Hook, OptimizerHook, IterTimerHook, CheckpointHook
import random
import sys
sys.path.append('.')
from  optimizer.sgld import SGLD
import optimizer.sgld

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def batch_processor(model, data, train_mode):
    img, label = data
    img = img.cuda()
    label = label.cuda(non_blocking=True)
    pred = model(img)
    loss = F.cross_entropy(pred, label, reduction='sum')
    acc = accuracy(pred, label, topk=(1,))[0]
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['accuracy'] = acc.item()
    # outputs = dict(loss = loss)
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def bayes_batch_processor(model, data, train_mode):
    img, label = data
    img = img.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    pred = model.forward(img, sample=False)
    loss, log_prior, log_variational_posterior, negative_log_likelehood = model.sample_elbo(img, label)
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    acc = accuracy(pred, label, topk=(1,))[0]
    log_vars['accuracy'] = acc.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def get_logger(log_level):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger


def parse_args():
    parser = ArgumentParser(description='Train Ankon')
    parser.add_argument('cfg_file', help='train config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job laucnher'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def parse_cfg(cfg_file):
    cfg = Config.fromfile(cfg_file)
    print(cfg.optimizer)
    print(sys.modules['sgld'])


def parse_optimizer(cfg, runner):
    if cfg.optimizer.type == 'SGLD':
        optimizer = SGLD(params=runner.model.parameters(), lr=0.01, norm_sigma=0.1)
        runner.optimizer = optimizer
    return runner


if __name__ == '__main__':
    parse_cfg('config/alexnet_idc.yaml')
