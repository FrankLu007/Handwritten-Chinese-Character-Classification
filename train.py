import torch, os
import argparse
import geffnet
from PIL import Image
from random import seed, shuffle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, AdamW
import timm
from lvvit import lvvit_m
import torch.optim as optim 

class HCCDataset(Dataset):
	def __init__(self, DataList, transform, classes):
		super(HCCDataset, self).__init__()
		self.DataList = DataList
		self.transform = transform
		self.classes = classes
		self.ToTensor = transforms.ToTensor()
	def __len__(self):
		return len(self.DataList)
	def __getitem__(self, index):
		label = int(self.DataList[index][self.DataList[index].index('_') + 1 : -4])

		# import numpy, cv2
		# img = numpy.array(t_transform(Image.open('../Downloads/train/' + self.DataList[index])))
		# cv2.imshow(self.DataList[index], img)
		# cv2.waitKey()
		if self.DataList[index][-3:] == 'jpg':
			return self.transform(self.ToTensor(Image.open('../train2/' + self.DataList[index]))), label
		else:
			return self.transform(self.ToTensor(Image.open('../pretrain/' + self.DataList[index]))), label

def forward(DataLoader, model, LossFunction, optimizer = None):
	correct = 0
	TotalLoss = 0
	cases = 0

	print('%5d/%5d'%(0, DataLoader.__len__()), flush = True, end = '\b' * 11)
	for index, (inputs, labels) in enumerate(DataLoader):
		torch.cuda.empty_cache()
		if optimizer:
			optimizer.zero_grad()

		inputs = inputs.half()
		labels = labels.cuda()
		outputs = model(inputs)
		del inputs
		loss = LossFunction(outputs, labels)
		TotalLoss += loss.mean().item() * labels.shape[0]
		if optimizer:
			loss.mean().backward()
			optimizer.step()
		pred = outputs.argmax(1)
		cases += labels.shape[0]
		correct += (pred == labels).sum().item()
		del outputs, pred, loss, labels
		print('%5d/%5d'%(index + 1, DataLoader.__len__()), flush = True, end = '\b' * 11)

	TotalLoss /= cases
	print('Loss: %5.3f'%TotalLoss, 'Accuracy: %5.3f%%'%(correct / cases * 100))
	return TotalLoss

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 1000)
    parser.add_argument("--bs", type = int, default = 64)
    parser.add_argument("--sz", type = int, default = 224)
    parser.add_argument("--lr", type = float, default = 0.0001)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None) #cait_m48_448
    parser.add_argument("--version", type = str)
    args = parser.parse_args()
    print(args)

    with open('class.txt', 'r', encoding = 'utf8') as file:
    	classes = file.read().splitlines()

    # DataList = list(filter(lambda x : int(x[:x.index('_')]) <= 6000, list(os.walk('../Downloads/train/'))[0][2]))
    DataList = list(os.walk('../train2/'))[0][2]
    AdditionalData = list(os.walk('../pretrain/'))[0][2]
    seed(999)
    shuffle(DataList)


    train_transform = torch.nn.Sequential(
        transforms.Pad((60, 30), fill = 0.941176),
#         transforms.Pad((60, 30), padding_mode = 'edge'),
        transforms.RandomRotation(30, fill = 0.941176),
        transforms.RandomPerspective(0.25, fill = 0.941176),
        transforms.CenterCrop((67, 100)),
        transforms.Resize((args.sz, args.sz)),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.5, 1.5), contrast = (0.5, 1.5), hue = 0.1),
#         transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
	)

    test_transform = torch.nn.Sequential(
		transforms.Pad((60, 30), fill = 0.941176),
		transforms.CenterCrop((67, 100)),
		transforms.Resize((args.sz, args.sz)),
	    transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
	)

    TrainingSet = HCCDataset(DataList[:-7000] + AdditionalData, train_transform, classes)
    ValidationSet = HCCDataset(DataList[-7000:], test_transform, classes)
    print(TrainingSet.__len__(), ValidationSet.__len__())

    TrainingLoader = DataLoader(TrainingSet, batch_size = args.bs, shuffle = True, pin_memory = True, drop_last = True, num_workers = 32)
    ValidationLoader = DataLoader(ValidationSet, batch_size = 256, pin_memory = True, num_workers = 32)
    if args.ld:
        model = torch.load(args.ld)
    else:
        model = timm.create_model(args.version, pretrained = True, num_classes = 801).half().cuda()
        # model = lvvit_m(pretrained = True, img_size = args.sz)
        # model.load_state_dict(torch.load('LV_Vit_M_448.weight', map_location = 'cpu'))
        # model.reset_classifier(801)
        # model = model.half().cuda()
        model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    optimizer = AdamW(model.parameters(), lr = args.lr, eps = 1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr = args.lr, eps = 1e-4)
    LossFunction = torch.nn.CrossEntropyLoss()

    model.eval()
    print('Validation', end = ' ')
    with torch.no_grad():
    	if args.ld:
    		BestLoss = forward(ValidationLoader, model, LossFunction)
    	else:
    		BestLoss = 9999999

    for epoch in range(args.ep):
    	print('\nEpoch : %3d'%epoch)
    	print('Training  ', end = ' ')
    	model.train()
    	forward(TrainingLoader, model, LossFunction, optimizer)

    	model.eval()
    	print('Validation', end = ' ')
    	with torch.no_grad():
    		loss = forward(ValidationLoader, model, LossFunction)
        
    	if loss < BestLoss:
    		BestLoss = loss
    		torch.save(model, args.sv)
