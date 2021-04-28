import cv2, os
from PIL import ImageFont, ImageDraw, Image
import numpy as np

filenames = list(os.walk('../Downloads/train/'))[0][2]
labels = set()
l = []
for file in filenames:
	img = Image.open('../Downloads/train/' + file)
	img = np.array(img)
	l += [img.shape[0], img.shape[1]]
	if img.shape[1] == 20 :
		print('GG', file, img.shape)
l.sort()
print(l[:10], l[-10:])