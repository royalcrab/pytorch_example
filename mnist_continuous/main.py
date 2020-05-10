from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)    # 入力1, 出力32層、カーネル3
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)   # 入力32層, 出力64層、かーねる3
        self.dropout1 = nn.Dropout2d(0.25)     # 出力は 1/4 ドロップ
        self.dropout2 = nn.Dropout2d(0.5)      # 出力は 1/2 ドロップ
        self.fc1 = nn.Linear(9216, 128)        # in 9216 = 64 x 8 x 2 x 9
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(128,1)
        # self.mse = nn.MSELoss()
        

    def forward(self, x):
        x = self.conv1(x)           # in 1 out 32, 3 x 3
        x = F.relu(x)
        x = self.conv2(x)           # in 32 out 64, 3 x 3
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # ここの捜査が多分いちばんわかってない。
        x = self.dropout1(x)        # 1/4 ドロップ
        x = torch.flatten(x, 1)     # 複数層 (3) を 1層にフラット化
        x = self.fc1(x)             # ここの in が 9616 で 64 * 32 * 9 になってる。out は 128
        x = F.relu(x)
        x = self.dropout2(x)        # 1/2 ドロップ。ドロップしてもサイズはかわらない。
        x = self.fc2(x)             # 入力 128
        # x = F.log_softmax(x, dim=1) # softmax は
        # output = self.mse(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        a = model(data)
        b = torch.t(a)
        output = torch.flatten(b)
        c = target.float()
        # print(output)
        # print(target)
        #loss = F.nll_loss(output, target)
        mse = nn.MSELoss()
        # loss = mse(output, target)
        loss = mse(output, c )
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #output = model(data)
            a = model(data)
            b = torch.t(a)
            output = torch.flatten(b)
            c = target.float()
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #mse = nn.MSELoss()
            # loss = mse(output, target)
            print(output)
            print(c)
            #test_loss += mse(output, c).item() # とりあえずここまではうまくいった
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args() # ここまではパラメータの処理
    use_cuda = not args.no_cuda and torch.cuda.is_available() 
    # cuda が available なら device としてcuda を使う

    torch.manual_seed(args.seed)
    # torch.manual_seed(1)
    
    device = torch.device("cuda" if use_cuda else "cpu")

    # cuda 使用時の設定
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # MNIST データのロード
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

#net = Net()
#print(net)

