import torch, os
import argparse
from PIL import Image
from model import ResNet, EffNet
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

t_transform = transforms.Compose([
	transforms.Pad((60, 10), padding_mode = 'edge'),
	transforms.RandomRotation(10, fill = 240),
	transforms.CenterCrop((67, 100)),
	transforms.GaussianBlur(3, 2),
	transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
])

class HCCDataset(Dataset):
	def __init__(self, DataList, transform, classes):
		super(HCCDataset, self).__init__()
		self.DataList = DataList
		self.transform = transform
		self.classes = classes
	def __len__(self):
		return len(self.DataList)
	def __getitem__(self, index):
		target = self.DataList[index][-5]
		if target not in self.classes:
			label = 800
		else:
			label = self.classes.index(target)
		# import numpy, cv2
		# img = numpy.array(t_transform(Image.open('../Downloads/train/' + self.DataList[index])))
		# cv2.imshow(self.DataList[index], img)
		# cv2.waitKey()
		return self.transform(Image.open('../Downloads/train/' + self.DataList[index])), label

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
		labels = labels.cuda()
		outputs = model(inputs)
		del inputs
		loss = LossFunction(outputs, labels)
		TotalLoss += loss.item()
		if optimizer:
			loss.backward()
			optimizer.step()
		value, pred = outputs.max(1)
		correct += (pred == labels).sum().item()
		cases += labels.shape[0]
		del outputs, value, pred, loss, labels
		print('%5d/%5d'%(index, DataLoader.__len__()), flush = True, end = '\b' * 11)

	TotalLoss /= DataLoader.__len__()
	print('Loss: %5.3f'%TotalLoss, 'Accuracy: %5.3f%%'%(correct / cases * 100))
	return correct

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 1000)
    parser.add_argument("--bs", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None)
    args = parser.parse_args()
    print(args)

    with open('class.txt', 'r', encoding = 'utf8') as file:
    	classes = file.read().splitlines()

    # DataList = list(filter(lambda x : int(x[:x.index('_')]) <= 6000, list(os.walk('../Downloads/train/'))[0][2]))
    DataList = list(os.walk('../Downloads/train/'))[0][2]
    shuffle(DataList)

    train_transform = transforms.Compose([
    	transforms.Pad((60, 10), padding_mode = 'edge'),
		transforms.RandomRotation(10, fill = 240),
		transforms.CenterCrop((67, 100)),
		transforms.GaussianBlur(3, 2),
		transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
	    transforms.ToTensor(),
	    transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828)),
	])

    test_transform = transforms.Compose([
		transforms.Pad((60, 10), padding_mode = 'edge'),
		transforms.CenterCrop((67, 100)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828)),
	])

    TrainingSet = HCCDataset(list(filter(lambda x : int(x[:x.index('_')]) > 6000, DataList)), train_transform, classes)
    ValidationSet = HCCDataset(list(filter(lambda x : int(x[:x.index('_')]) <= 6000, DataList)), test_transform, classes)
    print(TrainingSet.__len__(), ValidationSet.__len__())

    TrainingLoader = DataLoader(TrainingSet, batch_size = args.bs, shuffle = True, pin_memory = True, drop_last = True, num_workers = 4)
    ValidationLoader = DataLoader(ValidationSet, batch_size = 256, pin_memory = True, num_workers = 4)
    if args.ld:
    	model = torch.load(args.ld)
    else:
    	model = EffNet().half().cuda()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    LossFunction = torch.nn.CrossEntropyLoss()

    model.eval()
    print('Validation', end = ' ')
    with torch.no_grad():
    	BestAccuracy = forward(ValidationLoader, model, LossFunction)

    for epoch in range(args.ep):
    	print('\nEpoch : %3d'%epoch)
    	print('Training  ', end = ' ')
    	model.train()
    	forward(TrainingLoader, model, LossFunction, optimizer)

    	model.eval()
    	print('Validation', end = ' ')
    	with torch.no_grad():
    		accuracy = forward(ValidationLoader, model, LossFunction)
        
    	if accuracy > BestAccuracy:
    		BestAccuracy = accuracy
    		torch.save(model, args.sv)
