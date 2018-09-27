import torch
from bayes_network import BayesMLP
from dataset import data_factory
import numpy as np
TEST_SAMPLES = 2
TEST_BATCH_SIZE = 10
def test_ensemble(net, test_loader, device, test_size):
    net.eval()
    net.to(device)
    correct = 0
    corrects = np.zeros(TEST_SAMPLES+1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = torch.zeros(TEST_SAMPLES+1, TEST_BATCH_SIZE, 10).to(device)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)
            outputs[TEST_SAMPLES] = net(data, sample=False)
            output = outputs[0:TEST_SAMPLES].mean(0)
            preds = preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1] # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, test_size))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, test_size))
    print('Ensemble Accuracy: {}/{}'.format(correct, test_size))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BayesMLP(num_classes = 10, device = device)
    net.load_state_dict(torch.load('checkpoint/mnist/bayes/bayesMlp.pth'))
    test_loader = data_factory.make_dataloader('mnist', 10, is_train = False)
    test_size = len(test_loader)
    test_ensemble(net, test_loader, device, test_size)

if __name__ == '__main__':
    main()
    
