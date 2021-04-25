import torch.nn as nn
from torchvision.models import resnet18, resnext101_32x8d
from efficientnet_pytorch import EfficientNet

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()
		# self.ImageNet = resnet18(pretrained = True, progress = False)
		self.ImageNet = resnext101_32x8d(pretrained = True, progress = False)
		self.ImageNet.fc = nn.Linear(self.ImageNet.fc.in_features, 801)
	def forward(self, x):
		return self.ImageNet(x)

class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b7')
        self.ImageNet._fc = nn.Linear(self.ImageNet._fc.in_features, 801)
    def forward(self, x):
        return self.ImageNet(x)