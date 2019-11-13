import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon, frequence, weighted = True):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)
    self.FREC = torch.Tensor(frequence).cuda()
    self.weighted = weighted
    
  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    if(self.weighted):
        loss = (-targets * log_probs / self.num_classes / self.FREC).mean(0).sum()
    else:
        loss = -(targets * log_probs).mean(0).sum()
    return loss

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import _data_transforms

train_transforms, valid_transforms = _data_transforms(4)
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=train_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=valid_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
"""
from model import Net
model = Net()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')
"""
from genotype import DARTS
from NN import Network
model = Network(36, 200, 
                      20, True, DARTS, 
                      num_reduction = 2, 
                      input_size = 32)

model.cuda()
def get_frequency():
    with torch.no_grad():
        count = []
        for batch_idx, (data, target) in enumerate(train_loader):
            count.append(target.data.cpu().numpy())
        import numpy as np
        count = np.sum(np.array(count), axis = 0)
        return count/np.sum(count)
frequency = get_frequency()
num_class = frequency.shape[0]
print(frequency)
print(num_class)

criterion_train = CrossEntropyLabelSmooth(num_class, 0.1, frequency, True)

optimizer = torch.optim.SGD(
          model.parameters(),
          0.025,
          momentum=0.9,
          weight_decay=3e-4
          )

epochs = args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))
drop_path_prob = 0.2


        
        
def train(epoch):
    scheduler.step()
    model.train()
    model.drop_path_prob = drop_path_prob * epoch / epochs
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, output_aux = model(data)
        #criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion_train(output, target)
        loss += 0.4 * criterion_train(output_aux, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    model.drop_path_prob = 0
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output, _ = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
torch.save(model.state_dict(), model_file)
print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
