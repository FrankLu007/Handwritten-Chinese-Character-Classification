import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

filenames = list(os.walk('../Downloads/train/'))[0][2]
labels = set()
l = []
for file in filenames:
	print(file)
	print(os.path.isfile('../Downloads/train/' + file))
	img = cv2.imread('../Downloads/train/' + file)
	l += [img.shape[0], img.shape[1]]
	if img.shape[1] == 20 :
		print('GG', file, img.shape)
l.sort()
print(l[:10], l[-10:])