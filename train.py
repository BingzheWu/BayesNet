
import torch
from utils import write_loss_scalars
from utils import write_weight_histograms
from bayes_network import BayesMLP
from dataset import data_factory
def train(net, optimizer, epoch, train_loader, device = None):
    net.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        net.zero_grad()
        loss, los_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        if batch_idx % 10 == 0:
            print(loss)
        loss.backward()
        optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader = data_factory.make_dataloader('mnist', 100, is_train = True)
    net = BayesMLP(device = device).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(50):
        train(net, optimizer, epoch, train_loader, device = device)
    torch.save(net.state_dict(), 'checkpoint/mnist/bayes/bayesMlp.pth')
if __name__ == '__main__':
    main()
    
