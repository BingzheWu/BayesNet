
from dataset.mnist import mnist
from torch.utils.data import DataLoader
from dataset.cifar10 import cifar10
from dataset.idc import IDC


def make_dataset(cfg, is_train = True):
    if cfg.dataset == 'mnist':
        dataset = mnist(is_train = is_train)
    elif cfg.dataset == 'cifar10':
        dataset = cifar10(is_train=is_train)
    elif cfg.dataset == 'idc':
        if is_train:
            mode = 'train'
        else:
            mode = 'val'
        dataset = IDC(cfg, mode)
    return dataset


def make_dataloader(dataset, batch_size, num_workers, shuffle, data_sampler):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=data_sampler,
    )
    return dataloader


def test_():
    dataloader = make_dataloader('mnist', 100)
    print(len(dataloader))


if __name__ == '__main__':
    test_()