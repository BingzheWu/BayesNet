from torchvision import datasets, transforms
import torchvision.transforms as tv


def cifar10(is_train = True):
    if is_train:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10('./data/cifar10', train = is_train, download = True, transform = train_transforms)
    else:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = datasets.CIFAR10('./data/cifar10', train = is_train, download = True, transform = test_transforms)
    return dataset
