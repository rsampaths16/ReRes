import numpy
import scipy
import glob
from matplotlib import pyplot
from scipy import misc
from numpy import random

random.seed(0)
SIZE = 128
ORIGINAL = '../data/offline-data/black-and-white-images/original'
HIGH = '../data/offline-data/black-and-white-images/train/high'
LOW = '../data/offline-data/black-and-white-images/train/low'

def sample_patch(image):
    x = random.randint(0, image.shape[0] - SIZE, dtype=numpy.int)
    y = random.randint(0, image.shape[1] - SIZE, dtype=numpy.int)
    high = numpy.copy(image[x:x+SIZE, y:y+SIZE])
    low = numpy.copy(high)
    low = misc.imresize(low, (SIZE // 4, SIZE // 4))
    low = misc.imresize(low, (SIZE, SIZE))
    return low, high

unique_id = 1
for image_path in glob.glob(ORIGINAL + '/*.jpg'):
    print(image_path)
    sample = 1
    image = misc.imread(image_path)
    while sample > 0:
        low, high = sample_patch(image)
        misc.imsave(HIGH + '/' + str(unique_id) + '.jpg', high)
        misc.imsave(LOW + '/' + str(unique_id) + '.jpg', low)
        sample -= 1
        unique_id += 1
