import sys
sys.path.append('.')
from train.train_utils import batch_processor, parse_args, get_logger, parse_optimizer
from mmcv import Config
from models.model_factory import model_creator
from train.train_runner import Runner
from torch.nn.parallel import DataParallel
from dataset.data_factory import make_dataset, make_dataloader
import os


def run_by_fold_idx(fold_idx=1):
    args = parse_args()
    cfg = Config.fromfile(args.cfg_file)
    print(cfg)
    if args.launcher == 'none':
        dist = False
    else:
        pass
    if dist:
        pass
    else:
        num_workers = cfg.data_workers
        batch_size = cfg.batch_size
        train_sampler = None
        val_sampler = None
        shuffle = True
    train_fold_file = os.path.join(cfg.dataroot, 'kfold2', 'fold_'+str(fold_idx)+'_train.txt')
    val_fold_file = os.path.join(cfg.dataroot, 'kfold2', 'fold_' + str(fold_idx) + '_test.txt')
    cfg.fold_file = train_fold_file
    train_dataset = make_dataset(cfg)
    train_loader = make_dataloader(train_dataset, batch_size, num_workers, shuffle, train_sampler)
    cfg.fold_file = val_fold_file
    val_dataset = make_dataset(cfg)
    val_loader = make_dataloader(val_dataset, batch_size, num_workers, shuffle, val_sampler)
    model = model_creator(cfg)
    if dist:
        pass
    else:
        model = DataParallel(model, device_ids=[0, 1, 2]).cuda()

    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        log_level=cfg.log_level
    )
    log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
        ]
    )
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=None,
        checkpoint_config=cfg.checkpoint_config,
        log_config=log_config
    )
    workflow = [('train', 1), ('val', 1)]
    runner.run([train_loader, val_loader], workflow, cfg.total_epochs)


if __name__ == '__main__':
    run_by_fold_idx()
