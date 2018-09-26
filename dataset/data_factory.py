
from dataset.mnist import mnist
import torch
def make_dataloader(name, batch_size, shuffle = True, is_train = True):
    if name == 'mnist':
        dataset = mnist(is_train = is_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, )
    

    return dataloader

def test_():
    dataloader = make_dataloader('mnist', 100)
    print(len(dataloader))

if __name__ == '__main__':
    test_()