import torch, os
import argparse
from PIL import Image
from model import ResNet, EffNet, ViT
from random import seed, shuffle
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
		label = int(self.DataList[index][self.DataList[index].index('_') + 1 : -4])

		# import numpy, cv2
		# img = numpy.array(t_transform(Image.open('../Downloads/train/' + self.DataList[index])))
		# cv2.imshow(self.DataList[index], img)
		# cv2.waitKey()
		if self.DataList[index][-3:] == 'jpg':
			return self.transform(Image.open('../train2/' + self.DataList[index])), label
		else:
			return self.transform(Image.open('../pretrain/' + self.DataList[index])), label

def forward(DataLoader, model, LossFunction, optimizer = None):
	correct = [0] * 801
	TotalLoss = 0
	cases = 0
	precision = [0] * 801

	print('%5d/%5d'%(0, DataLoader.__len__()), flush = True, end = '\b' * 11)
	for index, (inputs, labels) in enumerate(DataLoader):
		torch.cuda.empty_cache()
		if optimizer:
			optimizer.zero_grad()

		inputs = inputs.half()
		labels = labels
		loss, outputs = model(inputs, labels)
		# outputs = model(inputs)
		del inputs
		# loss = LossFunction(outputs, labels)
		TotalLoss += loss.mean().item()
		if optimizer:
			loss.mean().backward()
			optimizer.step()
		value, pred = outputs.max(1)
		cases += labels.shape[0]
		for i, s in enumerate(pred.tolist()):
			precision[s] += 1
			if labels[i].item() == s:
				correct[s] += 1
		del outputs, value, pred, loss, labels
		print('%5d/%5d'%(index, DataLoader.__len__()), flush = True, end = '\b' * 11)

	TotalLoss /= DataLoader.__len__()
	avg = 0
	for i in range(801):
		if precision[i]:
			avg += correct[i] / precision[i]
	print('Loss: %5.3f'%TotalLoss, 'Accuracy: %5.3f%%'%(sum(correct) / cases * 100), 'Precision: %5.3f%%'%(avg / 801 * 100))
	return TotalLoss

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 1000)
    parser.add_argument("--bs", type = int, default = 64)
    parser.add_argument("--sz", type = int, default = 224)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None)
    parser.add_argument("--cuda", type = int, default = 1)
    args = parser.parse_args()
    print(args)

    with open('class.txt', 'r', encoding = 'utf8') as file:
    	classes = file.read().splitlines()

    # DataList = list(filter(lambda x : int(x[:x.index('_')]) <= 6000, list(os.walk('../Downloads/train/'))[0][2]))
    DataList = list(os.walk('../train2/'))[0][2]
    AdditionalData = list(os.walk('../pretrain/'))[0][2]
    seed(999)
    shuffle(DataList)


    train_transform = transforms.Compose([
    	transforms.Pad((60, 30), padding_mode = 'edge'),
		transforms.RandomRotation(10, fill = 240),
		transforms.CenterCrop((67, 100)),
		transforms.Resize((args.sz, args.sz)),
		transforms.GaussianBlur(3, 2),
		transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
	    transforms.ToTensor(),
	    transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828)),
	])

    test_transform = transforms.Compose([
		transforms.Pad((60, 30), padding_mode = 'edge'),
		transforms.CenterCrop((67, 100)),
		transforms.Resize((args.sz, args.sz)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828)),
	])

    TrainingSet = HCCDataset(DataList[:-7000] + AdditionalData, train_transform, classes)
    ValidationSet = HCCDataset(DataList[-7000:], test_transform, classes)
    print(TrainingSet.__len__(), ValidationSet.__len__())

    TrainingLoader = DataLoader(TrainingSet, batch_size = args.bs, shuffle = True, pin_memory = True, drop_last = True, num_workers = 12)
    ValidationLoader = DataLoader(ValidationSet, batch_size = 256, pin_memory = True, num_workers = 12)
    if args.ld:
    	model = torch.load(args.ld)
    else:
    	# model = EffNet().half().cuda(1)
    	model = ViT().half().cuda()
    	model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, eps = 1e-4)
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
