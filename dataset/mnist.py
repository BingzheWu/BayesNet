from torchvision import datasets, transforms
import torchvision as tv


def mnist(is_train = True):
    if is_train:
        train_transforms = tv.transforms.Compose([
            #tv.transforms.RandomRotation(10),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data/mnist', train = is_train, download = True, transform = train_transforms)
    else:
        test_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data/mnist', train = is_train, download = True, transform = test_transforms)
    return dataset
