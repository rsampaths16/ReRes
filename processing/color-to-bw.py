import glob
from PIL import Image
import os
import numpy
import cv2

ORIGINAL = '../data/offline-data/black-and-white-images/original/images'
#BW_IMAGES = 'data/sample-data/black-and-white-images/bw-images'

unique_id = 1
for image_path in glob.glob(ORIGINAL + '/*.jpg'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    head, tail = os.path.split(image_path)
    cv2.imwrite(head + '/../' + tail, image)
    img = numpy.asarray(image)
    print tail, img.shape
