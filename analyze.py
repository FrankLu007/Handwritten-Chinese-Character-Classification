import cv2
import os

filenames = list(os.walk('../train/'))[0][2]
labels = set()
l = []
for file in filenames:
	img = cv2.imread('../train/' + file)
	l += [img.shape[0], img.shape[1]]
	if img.shape[1] == 20 :
		print('GG', file, img.shape)
l.sort()
print(l[:10], l[-10:])