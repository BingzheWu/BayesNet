import sys
import os
import torch
from mmcv import Config
import torchvision
sys.path.append('.')
from attack.utils import load_model
from train.train_utils import batch_processor
from dataset.data_factory import make_dataset, make_dataloader


def val(cfg, data_loader, **kwargs):
    model = load_model(cfg)
    model.cuda()
    acc = 0
    for i, data_batch in enumerate(data_loader):
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, **kwargs)
        acc += outputs['log_vars']['accuracy']
    acc = acc /(i+1)
    print("accuracy%f"%acc)


if __name__ == '__main__':
    cfg_file = sys.argv[1]
    cfg = Config.fromfile(cfg_file)
    dataset = make_dataset(cfg, True)
    data_loader = make_dataloader(dataset, cfg.batch_size, num_workers=16, shuffle=False, data_sampler=None)
    val(cfg, data_loader)


