import torch
import torch.nn as nn
from torchvision.models import resnet18
from efficientnet_pytorch import EfficientNet
from transformers import ViTForImageClassification

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()
		self.ImageNet = resnet18(pretrained = True, progress = False)
		self.ImageNet.fc = nn.Linear(self.ImageNet.fc.in_features, 801)
	def forward(self, x):
		return self.ImageNet(x)

class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.ImageNet = EfficientNet.from_name('efficientnet-b7')
        # s = torch.rand((1, 3, 224, 224))
        # with torch.no_grad():
            # sz = self.ImageNet.extract_features(s).shape
        # self.linear = nn.Sequential(nn.Dropout(0.5, inplace = True), nn.Linear(sz[1] * sz[2] * sz[3], 801))
        self.ImageNet._fc = nn.Linear(self.ImageNet._fc.in_features, 801)
    def forward(self, x):
        # x = self.ImageNet.extract_features(x) # [2560, 2, 3]
        # return self.linear(x.reshape(-1, self.linear[1].in_features))
        return self.ImageNet(x)
class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.ImageNet = ViTForImageClassification.from_pretrained('google/vit-huge-patch14-224-in21k')
        self.ImageNet.classifier = nn.Linear(self.ImageNet.classifier.in_features, 801)
        self.ImageNet.num_labels = 801
    def forward(self, x, labels):
        return self.ImageNet(x, output_attentions = False, output_hidden_states = False, labels = labels)[:2]
