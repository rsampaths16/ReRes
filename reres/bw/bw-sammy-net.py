import numpy
import scipy
import cv2
from scipy import misc
from matplotlib import pyplot
from numpy import random
from keras.layers import Input, LeakyReLU, BatchNormalization, concatenate
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Flatten, Dense, Reshape
from keras.models import Model
from keras.optimizers import SGD
from keras.losses import squared_hinge

random.seed(0)
n_train = 30
batch_size = 32

HIGH = '../../data/offline-data/black-and-white-images/train/high'
LOW = '../../data/offline-data/black-and-white-images/train/low'
N_CHANNEL = 1

def unet():
    inputs = Input(shape=(128, 128, 1))
    conv1 = Conv2D(8, (2, 2), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.33)(conv1)
    conv1 = Conv2D(8, (2, 2), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.33)(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(8, (2, 2), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.33)(conv2)
    conv2 = Conv2D(8, (2, 2), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.33)(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(8, (2, 2), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.33)(conv3)
    conv3 = Conv2D(8, (2, 2), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.33)(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = Conv2D(8, (2,2), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.33)(conv4)
    conv4 = Conv2D(8, (2, 2), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.33)(conv4)
    pool4 = MaxPooling2D()(conv4)

    flat5 = Flatten()(pool4)
    dense5 = Dense(64, activation='relu')(flat5)
    dense5 = Dense(64, activation='relu')(dense5)
    dense5 = Dense(128, activation='relu')(dense5)
    dense5 = Dense(64, activation='relu')(dense5)
    dense5 = Dense(64, activation='relu')(dense5)
    uflat5 = Reshape((8, 8, 8))(Dense(512, activation='relu')(dense5))

    upool6 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(uflat5), conv4], axis=3)
    conv6 = Conv2D(8, (2, 2), padding='same')(upool6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.33)(conv6)
    conv6 = Conv2D(8, (2, 2), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.33)(conv6)

    upool7 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(8, (2, 2), padding='same')(upool7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.33)(conv7)
    conv7 = Conv2D(8, (2, 2), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.33)(conv7)

    upool8 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(8, (2, 2), padding='same')(upool8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.33)(conv8)
    conv8 = Conv2D(8, (2, 2), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.33)(conv8)

    upool9 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (2, 2), padding='same')(upool9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.33)(conv9)
    conv9 = Conv2D(8, (2, 2), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.33)(conv9)

    outputs = concatenate([inputs, conv9], axis=3)
    outputs = Conv2D(1, (1, 1), padding='same', activation='relu')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

cnn = unet()
sgd = SGD(lr=0.99, decay=1e-5, momentum=0.9, nesterov=True)
cnn.compile(
    loss='mean_squared_error',
    optimizer='adamax',
    metrics=['accuracy']
)

x_train = numpy.empty((n_train, 128, 128, N_CHANNEL))
y_train = numpy.empty((n_train, 128, 128, N_CHANNEL))

for image_id in range(1, n_train + 1):
    x_train[image_id - 1] = numpy.expand_dims(misc.imread(LOW + '/' + str(image_id) + '.jpg'), axis=3)
    y_train[image_id - 1] = numpy.expand_dims(misc.imread(HIGH + '/' + str(image_id) + '.jpg'), axis=3)
print x_train.shape

'''
z_train = numpy.split(x_train, 3,axis=3)
misc.imshow(z_train[0][0])
print z_train[0][0].shape

for x in range(128):
    for y in range(128):
        for z in range(n_train):
            x_train[z][x][y][1] = 255
            y_train[z][x][y][1] = 255
'''
gap = numpy.ones((128, 16, N_CHANNEL)) * 255
epoch = 1
cnn.load_weights('../../weights/my_weights.h5')
while True:
    cnn.fit(x_train, y_train, batch_size=batch_size, verbose=2)
    cnn.save_weights('../../weights/my_weights.h5')
    for sample in range(1, 8):
        image_id = random.randint(0, n_train)
        #image_id = sample - 1
        original = y_train[image_id]
        foriginal = x_train[image_id]
        predicted = cnn.predict(numpy.expand_dims(x_train[image_id], axis=0))[0]
        predicted = numpy.ndarray.round(predicted)
        numpy.clip(predicted, 0, 255, out=predicted)

        predicted2 = cnn.predict(numpy.expand_dims(y_train[image_id], axis=0))[0]
        predicted2 = numpy.ndarray.round(predicted2)
        numpy.clip(predicted2, 0, 255, out=predicted2)

        to_save = numpy.concatenate((foriginal, gap, original, gap, predicted, gap, predicted2), axis=1)
        print to_save.shape
        #misc.imsave('data/sample-data/black-and-white-images/results/epoch-' + str(epoch) + '-sample-' + str(sample) + '.jpg', to_save)
        cv2.imwrite('../../data/offline-data/black-and-white-images/results/epoch-' + str(epoch) + '-sample-' + str(sample) + '.jpg', to_save)
    epoch += 1

