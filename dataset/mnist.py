from torchvision import datasets, transforms

def mnist(is_train = True):
    dataset = datasets.MNIST('./data/mnist', train = is_train, download = True, transform = transforms.ToTensor())
    return dataset
