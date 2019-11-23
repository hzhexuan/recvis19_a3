
import torch.nn as nn
import torch
use_cuda = torch.cuda.is_available()

#state_dict = torch.load(args.model)
import torchvision.models as models
model_path = ['../gdrive/My Drive/experiment/wideresnet.pth', '../gdrive/My Drive/experiment/resnext101.pth', '../gdrive/My Drive/experiment/densenet161.pth']
networks = ["wideresnet", "resnext", "densenet161"]

model_cache = []
num_class = 20
for i in range(len(model_path)):
    path = model_path[i]
    network = networks[i]
    if(network == "densenet161"):
        model = models.densenet161(pretrained=False)
        model.classifier = nn.Linear(2208, num_class)
    if(network == "wideresnet"):
        model = models.wide_resnet101_2(pretrained=False)
        model.fc = nn.Linear(2048, num_class)
    if(network == "resnext"):
        model = models.resnext101_32x8d(pretrained=False)
        model.fc = nn.Linear(2048, num_class)
    if(network == "resnet152"):
        model = models.resnet152(pretrained=False)
        model.fc = nn.Linear(2048, num_class)
    model.load_state_dict(torch.load(path))
    model.cuda()
    model.eval()

    model_cache.append(model)
from data import _data_transforms
from torchvision import datasets

_, valid_transforms = _data_transforms(0)
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('bird_dataset' + '/train_images',
                         transform=valid_transforms),
    batch_size=1, shuffle=False, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('bird_dataset' + '/val_images',
                         transform=valid_transforms),
    batch_size=1, shuffle=False, num_workers=1)

import numpy as np
def extract_features(loader, model_cache):
    features = []
    targets = []
    for data, target in loader:
        targets.append(target.data.numpy())
        f = []
        for model in model_cache:
            f.append(model(data).cpu().data.numpy())
        f = np.array(f).flatten()
        features.append(f)
    return np.array(features), np.array(targets)

f_train, t_train = extract_features(train_loader, model_cache)
        


