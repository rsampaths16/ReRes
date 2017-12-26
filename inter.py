#!/bin/python
import numpy as np
from scipy.misc import imread, imshow
from scipy import ndimage
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
def GetBilinearPixel(imArr, posX, posY):
	out = []
	modXi = int(posX)
	modYi = int(posY)
	modXf = posX - modXi
	modYf = posY - modYi
	modXiPlusOneLim = min(modXi+1,imArr.shape[1]-1)
	modYiPlusOneLim = min(modYi+1,imArr.shape[0]-1)
	for chan in range(imArr.shape[2]):
		bl = imArr[modYi, modXi, chan]
		br = imArr[modYi, modXiPlusOneLim, chan]
		tl = imArr[modYiPlusOneLim, modXi, chan]
		tr = imArr[modYiPlusOneLim, modXiPlusOneLim, chan]
		b = modXf * br + (1. - modXf) * bl
		t = modXf * tr + (1. - modXf) * tl
		pxf = modYf * t + (1. - modYf) * b
		out.append(int(pxf+0.5))
	return out
if __name__=="__main__":
	im = imread("/Users/raghuveeramalla/Desktop/Low Resolution/2.jpg")
	imShape = list(map(int, [im.shape[0]*1.6, im.shape[1]*1.6, im.shape[2]]))
	imenlarge = np.empty(imShape, dtype=np.uint8)
	rowScale = float(im.shape[0]) / float(imenlarge.shape[0])
	colScale = float(im.shape[1]) / float(imenlarge.shape[1])
	for r in range(imenlarge.shape[0]):
		for c in range(imenlarge.shape[1]):
			orir = r * rowScale
			oric = c * colScale
			imenlarge[r, c] = GetBilinearPixel(im, oric, orir)
blurred_f = ndimage.gaussian_filter(imenlarge, 0.05)
filter_blurred_f = ndimage.gaussian_filter(blurred_f, 0.25)
alpha = 3
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
sharpened2=sharpened + alpha * (blurred_f - filter_blurred_f)
#brightness=ImageEnhance.Brightness(sharpened)
#brightness.show()
plt.imshow(im, interpolation="bicubic")
plt.show()
plt.imshow(im, interpolation="nearest")
plt.show()
plt.imshow(np.uint8(sharpened))
plt.show()
