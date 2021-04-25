import argparse
import torch, numpy, cv2
from PIL import ImageFont, ImageDraw, Image
from model import ResNet, EffNet
import torchvision.transforms as transforms

font = ['魏碑體.TTC', '雅風體W3.ttc', '鋼筆體W2.TTC', '采風體W3.ttc', '行書體.TTC', 
		'細圓體.ttc', '粗黑體.ttc', '粗圓體.ttc', '竹風體W4.ttc', '秀風體W3.ttc', 
		'流風體W3.ttc', '兒風體W4.ttc', 'kaiu.ttf']
augment = transforms.Compose([
		transforms.RandomRotation(10, fill = 240),
		transforms.CenterCrop((67, 100)),
		transforms.GaussianBlur(3, 2),
		transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
	    transforms.ToTensor(),
	    transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828)),
	])

def CreataData(label : int) -> torch.Tensor :
	
	# background
	color = torch.randint(low = 220, high = 256, size = (3, )).tolist()
	image = numpy.zeros((67 * 2, 100 * 2, 3), dtype = 'uint8')
	image[:, :, 0].fill(color[0])
	image[:, :, 1].fill(color[1])
	image[:, :, 2].fill(color[2])

	# font
	size = torch.randint(low = 30, high = 70, size = (1, )).item()
	FontType = ImageFont.truetype('C:\\WINDOWS\\FONTS\\' + font[torch.randint(low = 0, high = len(font), size = (1, )).item()], size)
	PIL_Image = Image.fromarray(image)
	draw = ImageDraw.Draw(PIL_Image)
	pos = (torch.randint(low = 50, high = 100, size = (1, )).item(), torch.randint(low = 34, high = 80 - size // 2, size = (1, )).item())
	if label == 800:
		draw.text(pos, classes[torch.randint(low = 0, high = 800, size = (1, )).item()], font = FontType, fill = tuple(torch.randint(low = 100, high = 150, size = (1, )).tolist() + torch.randint(low = 0, high = 50, size = (2, )).tolist()))
		PIL_Image = PIL_Image.transpose(Image.FLIP_LEFT_RIGHT)
	else:
		draw.text(pos, classes[label], font = FontType, fill = tuple(torch.randint(low = 100, high = 150, size = (1, )).tolist() + torch.randint(low = 0, high = 50, size = (2, )).tolist()))

	
	# PIL_Image = augment(PIL_Image)
	# image = numpy.array(PIL_Image)
	# cv2.imshow(str(torch.randint(low = 0, high = 256, size = (1, )).item()), image)
	# cv2.waitKey(0)
	
	return augment(PIL_Image)
	# return torch.Tensor([0])

if __name__ == '__main__':

	# parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--it", type = int, default = 1000)
    parser.add_argument("--bs", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None)
    args = parser.parse_args()
    print(args)

    with open('class.txt', 'r', encoding = 'utf8') as file:
    	classes = file.read().splitlines()

    # basic setting
    if args.ld:
    	model = torch.load(args.ld)
    else:
    	model = EffNet().half().cuda()
    	model = ResNet().cuda()
    torch.backends.cudnn.benchmark = True
    LossFunction = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    BestLoss = 99999

    for iteration in range(args.it):

    	# initialize
    	torch.cuda.empty_cache()
    	optimizer.zero_grad()

    	# create data
    	labels = torch.randint(low = 0, high = 801, size = (args.bs, ), device = 'cuda')
    	inputs = torch.stack([CreataData(label.item()) for label in labels]).cuda()

    	# forward and backward
    	outputs = model(inputs)
    	loss = LossFunction(outputs, labels)
    	loss.backward()
    	optimizer.step()
    	del inputs

    	# calcualte performance
    	tmp, pred = outputs.max(1)
    	print('Iteration : %3d'%iteration, 'Acc: %5.3f%%'%((pred == labels).sum().item() / args.bs * 100), ' Loss: %5.4f'%loss.item())
    	if loss.item() < BestLoss and iteration:
    		BestLoss = loss.item()
    		torch.save(model, args.sv)

    	del tmp, pred, outputs, labels, loss


