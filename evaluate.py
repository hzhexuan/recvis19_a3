import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch.nn as nn
import torch

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--network', type=str, default='resnet152')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
import torchvision.models as models
if(args.network == "densenet161"):
    model = models.densenet161(pretrained=True)
    model.classifier = nn.Linear(2208, 20)
if(args.network == "wideresnet"):
    model = models.wide_resnet101_2(pretrained=True)
    model.fc = nn.Linear(2048, 20)
if(args.network == "resnext"):
    model = models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(2048, 20)
if(args.network == "resnet152"):
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(2048, 20)
if(args.network == "inceptionv3"):
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, 20)

model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

from data import _data_transforms, _data_transforms299

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

if(args.network == "inceptionv3"):
    _, data_transforms = _data_transforms299(0)
else:
    _, data_transforms = _data_transforms(0)
output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        


