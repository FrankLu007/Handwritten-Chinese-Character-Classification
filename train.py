import torch, os
import argparse
from PIL import Image
from random import seed, shuffle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import torch.optim as optim 
from time import time
import datetime
from transformers import AdamW

class HCCDataset(Dataset):
    def __init__(self, DataList, transform, NULLData = None):
        super(HCCDataset, self).__init__()
        self.DataList = DataList
        self.transform = transform
        self.ToTensor = transforms.ToTensor()
        self.hflip = transforms.functional.hflip
        self.NULLData = NULLData
    def __len__(self):
        return len(self.DataList)
    def __getitem__(self, index):
        file = self.DataList[index]
        if self.NULLData and torch.rand(size = (1,)) > 0.95:
            file = self.NULLData[torch.randint(0, len(self.NULLData), size = (1,))]

        label = int(file[file.index('_') + 1 : -4])

        if file[-3:] == 'jpg':
            return self.transform(self.ToTensor(Image.open('../train2/' + file))), label
        elif label < 800:
            return self.transform(self.ToTensor(Image.open('../pretrain/' + file))), label
        else:
            return self.transform(self.ToTensor(Image.open('../pretrain2/' + file))), label
def forward(DataLoader, model, LossFunction, optimizer = None):

    correct = 0
    TotalLoss = 0
    cases = 0

    print('%5d/%5d'%(0, DataLoader.__len__()), flush = True, end = '\b' * 11)
    for index, (inputs, labels) in enumerate(DataLoader):
        torch.cuda.empty_cache()
        if optimizer:
            optimizer.zero_grad()

        inputs = inputs.half().cuda()
        with torch.cuda.amp.autocast():
            labels = labels.cuda()
            outputs = model(inputs)
            del inputs
            loss = LossFunction(outputs, labels)

        if loss.item() != loss.item():
            print('NAN')
            exit()
        TotalLoss += loss.item() * labels.shape[0]
        if optimizer:
            loss.backward()
            optimizer.step()

        pred = outputs.argmax(1)
        cases += labels.shape[0]
        correct += (pred == labels).sum().item()

        del outputs, pred, loss, labels
        print('%5d/%5d'%(index + 1, DataLoader.__len__()), flush = True, end = '\b' * 11)

    TotalLoss /= cases
    print('Loss: %5.3f'%TotalLoss, 'Accuracy: %5.3f%%'%(correct / cases * 100), end = ' ')
    return TotalLoss

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 1000)
    parser.add_argument("--bs", type = int, default = 64)
    parser.add_argument("--sz", type = int, default = 224)
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument("--lr", type = float, default = 0.0001)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None)
    parser.add_argument("--version", type = str)
    args = parser.parse_args()
    print(args)

    DataList = list(os.walk('../train2/'))[0][2]
    AdditionalData = list(os.walk('../pretrain/'))[0][2]

    TestData = list(os.walk('../test/'))[0][2]
    for i in range(len(TestData)):
        TestData[i] = '../test/' + TestData[i]
    AdditionalNULLData = list(os.walk('../pretrain2/'))[0][2]

    seed(args.seed)
    shuffle(DataList)
    shuffle(AdditionalNULLData)


    train_transform = torch.nn.Sequential(
        transforms.Pad((5, 5), fill = 0.941176),
        transforms.RandomRotation(30, fill = 0.941176),
        transforms.RandomPerspective(0.25, fill = 0.941176),
        transforms.RandomCrop((67, 100), fill = 0.941176, pad_if_needed = True),
        transforms.Resize((args.sz, args.sz)),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.5, 1.5), contrast = (0.5, 1.5), hue = 0.1),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    )

    test_transform = torch.nn.Sequential(
        transforms.Pad((60, 30), fill = 0.941176),
        transforms.CenterCrop((67, 100)),
        transforms.Resize((args.sz, args.sz)),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    )


    TrainingSet = HCCDataset(DataList[:-7000] + TestData, train_transform)
    ValidationSet = HCCDataset(DataList[-7000:], test_transform)
    NULLSet = HCCDataset(AdditionalNULLData[-13000:], test_transform)
    print(TrainingSet.__len__(), ValidationSet.__len__())

    TrainingLoader = DataLoader(TrainingSet, batch_size = args.bs, shuffle = True, pin_memory = True, drop_last = True, num_workers = 4)
    ValidationLoader = DataLoader(ValidationSet, batch_size = 64, pin_memory = True, num_workers = 4)
    NULLLoader = DataLoader(NULLSet, batch_size = 64, pin_memory = True, num_workers = 4)
    if args.ld:
        model = torch.load(args.ld)
    else:
        model = timm.create_model(args.version, pretrained = True, num_classes = 801)
        model = model.cuda()
        # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    # optimizer = AdamW(model.parameters(), lr = args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    LossFunction = torch.nn.CrossEntropyLoss()

    model.eval()
    
    with torch.no_grad():
        if args.ld:
            print('Validation', end = ' ')
            t = time()
            BestLoss = forward(ValidationLoader, model, LossFunction)
            print('Time %5d sec'%(time() - t))
        else:
            BestLoss = 9999999

    for epoch in range(args.ep):
        print('\nEpoch : %3d'%epoch)
        print('Training  ', end = ' ')
        model.train()
        t = time()
        forward(TrainingLoader, model, LossFunction, optimizer)
        print('Time %5d sec'%(time() - t))

        model.eval()
        print('Validation', end = ' ')
        with torch.no_grad():
            t = time()
            loss = forward(ValidationLoader, model, LossFunction)
            print('Time %5d sec'%(time() - t))
            print('NULL', end = ' ')
            t = time()
            forward(NULLLoader, model, LossFunction)
            print('Time %5d sec'%(time() - t))
        
        if loss < BestLoss:
            BestLoss = loss
            torch.save(model, args.sv)
