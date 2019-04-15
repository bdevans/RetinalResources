
# coding: utf-8

from __future__ import print_function
import sys
import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras
from keras import backend as K
from keras import activations, initializers, regularizers, constraints, metrics
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, Layer,
                          BatchNormalization, LocallyConnected2D,
                          ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose,
                          GaussianNoise, UpSampling2D, Input)
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces


# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--data_set', type=str, default='cifar10',
                    help='Data set to use')
parser.add_argument('--trial_label', default='Trial1',
                    help='For labeling different runs of the same model')
parser.add_argument('--noise_start', type=float, default=0.0,
                    help='Input noise')
parser.add_argument('--noise_end', type=float, default=0.0,
                    help='Retinal output noise')
parser.add_argument('--retina_out_weight_reg', type=float, default=0.0,
                    help='L1 regularization on retinal output weights')
parser.add_argument('--reg', type=float, default=0.0,
                    help='L1 weight regularization for layers besides the retinal output layer')
parser.add_argument('--retina_hidden_channels', type=int, default=32,
                    help='Channels in hidden layers of retina')
parser.add_argument('--retina_out_stride', type=int, default=1,
                    help='Stride at output layer of retina')
parser.add_argument('--task', default='classification',
                    help='e.g. classification or reconstruction')
parser.add_argument('--filter_size', type=int, default=9,
                    help='Convolutional filter size')
parser.add_argument('--retina_layers', type=int, default=2,
                    help='Number of layers in retina')
parser.add_argument('--vvs_layers', type=int, default=2,
                    help='Number of convolutional layers in VVS')
parser.add_argument('--use_b', type=int, default=1,
                    help='Whether or not to use bias terms in retinal output layer')
parser.add_argument('--actreg', type=float, default=0.0,
                    help='L1 regularization on retinal output')
parser.add_argument('--retina_out_width', type=int, default=1,
                    help='Number of output channels in Retina')
parser.add_argument('--vvs_width', type=int, default=32,
                    help='Number of output channels in VVS layers')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train model')
parser.add_argument('--data_augmentation', type=int, default=1,
                    help='Flag to use data augmentation in training')


args = parser.parse_args()

data_set = args.data_set
trial_label = args.trial_label
noise_start = args.noise_start
noise_end = args.noise_end
retina_out_weight_reg = args.retina_out_weight_reg
retina_hidden_channels = args.retina_hidden_channels
retina_out_stride = args.retina_out_stride
task = args.task
filter_size = args.filter_size
retina_layers = args.retina_layers
vvs_layers = args.vvs_layers
use_b = args.use_b
actreg = args.actreg
retina_out_width = args.retina_out_width
vvs_width = args.vvs_width
epochs = args.epochs
reg = args.reg
data_augmentation = args.data_augmentation

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = (
    f"{data_set}_type_{trial_label}_noise_start_{noise_start}"
    f"_noise_end_{noise_end}_reg_{reg}_retina_reg_{retina_out_weight_reg}"
    f"retina_hidden_channels_{retina_hidden_channels}_SS_{retina_out_stride}"
    f"_task_{task}_filter_size_{filter_size}_retina_layers_{retina_layers}"
    f"_vvs_layers{vvs_layers}_bias_{use_b}_actreg_{actreg}"
    f"_retina_out_channels_{retina_out_width}_vvs_width_{vvs_width}"
    f"_epochs_{epochs}"
)

batch_size = 64
num_classes = 10

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

if use_b == 1:
    use_b = True
else:
    use_b = False

if data_augmentation == 1:
    data_augmentation = True
else:
    data_augmentation = False

if data_set == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif data_set == 'pixel':
    data_root = '/workspace/data/pixel'  # TODO: Pass in
    (noise_type, noise_cond, trial) = trial_label.split("_")

    if noise_type == 'Salt-and-pepper':
        data_path = os.path.join(data_root, 'salt_n_pepper', 'set_1')
    elif noise_type == 'Additive':
        data_path = os.path.join(data_root, 'uniform', 'set_1')
    elif noise_type == 'Single-pixel':
        data_path = os.path.join(data_root, 'single_pixel', 'set_1')
    else:
        sys.exit(f"Unknown noise type requested: {noise_type}")

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, f"test_{noise_cond.lower()}")

    def load_images(path):

        image_set = {}
        for root, dirs, files in os.walk(path):
            if root == path:
                categories = sorted(dirs)
                image_set = {cat: [] for cat in categories}
            else:
                image_set[os.path.basename(root)] = sorted(files)

        n_cat_images = {cat: len(files) for (cat, files) in image_set.items()}
        n_images = sum(n_cat_images.values())
        image_dims = plt.imread(os.path.join(path, categories[0], image_set[categories[0]][0])).shape

        X = np.zeros((n_images, *image_dims), dtype='float32')
        y = np.zeros((n_images, len(categories)), dtype=int)
        # y = np.zeros(n_images, dtype=int)

        for c, (cat, files) in enumerate(tqdm(image_set.items(), desc=path)):
            for i, image in enumerate(files):
                X[i] = plt.imread(os.path.join(path, cat, image))
            y[c*n_cat_images[cat]:(c+1)*n_cat_images[cat], c] = True

        return image_set, X, y

    if os.path.isfile(os.path.join(train_path, 'x_train.npy')):
        print(f'Loading {data_set} data arrays.')
        x_train = np.load(os.path.join(train_path, 'x_train.npy'))
        y_train = np.load(os.path.join(train_path, 'y_train.npy'))
        # num_classes = len(os.listdir(train_path)) - 1
        num_classes = len([os.path.join(train_path, o) for o in os.listdir(train_path)
                           if os.path.isdir(os.path.join(train_path, o))])
    else:
        print(f'Loading {data_set} image files.')
        train_images, x_train, y_train = load_images(train_path)
        num_classes = len(train_images)
        np.save(os.path.join(train_path, 'x_train.npy'), x_train)
        np.save(os.path.join(train_path, 'y_train.npy'), y_train)

    if os.path.isfile(os.path.join(test_path, 'x_test.npy')):
        x_test = np.load(os.path.join(test_path, 'x_test.npy'))
        y_test = np.load(os.path.join(test_path, 'y_test.npy'))
    else:
        test_images, x_test, y_test = load_images(test_path)
        np.save(os.path.join(test_path, 'x_test.npy'), x_test)
        np.save(os.path.join(test_path, 'y_test.npy'), y_test)

else:
    sys.exit(f"Unknown data set requested: {data_set}")

x_train = np.mean(x_train, 3, keepdims=True)  # Average over RGB channels
x_test = np.mean(x_test, 3, keepdims=True)  # Average over RGB channels
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

if data_set == 'cifar10':
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


# filters = 64
# NX = 32
# NY = 32
# NC = 1
# img_rows, img_cols, img_chns = NX, NY, NC
intermediate_dim = 1024
x = Input(shape=x_train[0].shape)
gn = GaussianNoise(noise_start)(x)
if retina_layers > 2:
    conv1_nonlin = Conv2D(retina_hidden_channels, (filter_size, filter_size),
                          kernel_regularizer=keras.regularizers.l1(reg),
                          padding='same', name='retina_1', activation='relu',
                          input_shape=x_train.shape[1:])(gn)
    retina_out = Conv2D(retina_hidden_channels, (filter_size, filter_size),
                        kernel_regularizer=keras.regularizers.l1(reg),
                        padding='same', activation='relu', name='retina_2',
                        trainable=True)(conv1_nonlin)
    for iterationX in range(retina_layers - 2):
        if iterationX == retina_layers - 3:
            retina_out = Conv2D(retina_out_width, (filter_size, filter_size),
                                strides=(retina_out_stride, retina_out_stride),
                                kernel_regularizer=keras.regularizers.l1(retina_out_weight_reg),
                                activity_regularizer=keras.regularizers.l1(actreg),
                                padding='same', name='retina_'+str(iterationX+3),
                                activation='relu', use_bias=use_b)(retina_out)
        else:
            retina_out = Conv2D(retina_hidden_channels, (filter_size, filter_size),
                                kernel_regularizer=keras.regularizers.l1(reg),
                                padding='same', name='retina_'+str(iterationX+3),
                                activation='relu')(retina_out)

if retina_layers == 2:
    conv1_nonlin = Conv2D(retina_hidden_channels, (filter_size, filter_size),
                          kernel_regularizer=keras.regularizers.l1(reg),
                          padding='same', input_shape=x_train.shape[1:],
                          name='retina_1', activation='relu', trainable=True)(gn)
    retina_out = Conv2D(retina_out_width, (filter_size, filter_size),
                        strides=(retina_out_stride, retina_out_stride),
                        kernel_regularizer=keras.regularizers.l1(retina_out_weight_reg),
                        padding='same', activation='relu',
                        activity_regularizer=keras.regularizers.l1(actreg),
                        use_bias=use_b, name='retina_2', trainable=True)(conv1_nonlin)


elif retina_layers == 1:
    retina_out = Conv2D(retina_out_width, (filter_size, filter_size),
                        strides=(retina_out_stride, retina_out_stride),
                        kernel_regularizer=keras.regularizers.l1(specalreg),  # specalreg is not defined!
                        activity_regularizer=keras.regularizers.l1(actreg),
                        padding='same', input_shape=x_train.shape[1:],
                        use_bias=use_b, name='retina_1', activation='relu',
                        trainable=True)(gn)

elif retina_layers == 0:
    retina_out = gn


if noise_end > 0:
    retina_out = GaussianNoise(noise_end)(retina_out)


if vvs_layers > 2:
    vvs_1 = Conv2D(vvs_width, (filter_size, filter_size),
                   kernel_regularizer=keras.regularizers.l1(reg),
                   padding='same', name='vvs_1', activation='relu')(retina_out)
    vvs_2 = Conv2D(vvs_width, (filter_size, filter_size),
                   kernel_regularizer=keras.regularizers.l1(reg),
                   padding='same', name='vvs_2', activation='relu')(vvs_1)
    for iterationX in range(vvs_layers - 2):
        vvs_2 = Conv2D(vvs_width, (filter_size, filter_size),
                       kernel_regularizer=keras.regularizers.l1(reg),
                       padding='same', name='vvs_'+str(iterationX+3),
                       activation='relu')(vvs_2)
    flattened = Flatten()(vvs_2)

if vvs_layers == 2:
    vvs_1 = Conv2D(vvs_width, (filter_size, filter_size),
                   kernel_regularizer=keras.regularizers.l1(reg),
                   padding='same', name='vvs_1', activation='relu',
                   trainable=True)(retina_out)
    vvs_2 = Conv2D(vvs_width, (filter_size, filter_size),
                   kernel_regularizer=keras.regularizers.l1(reg),
                   padding='same', name='vvs_2', activation='relu',
                   trainable=True)(vvs_1)
    flattened = Flatten()(vvs_2)

elif vvs_layers == 1:
    vvs_1 = Conv2D(vvs_width, (filter_size, filter_size),
                   kernel_regularizer=keras.regularizers.l1(reg),
                   padding='same', name='vvs_1', activation='relu')(retina_out)
    flattened = Flatten()(vvs_1)

elif vvs_layers == 0:
    flattened = Flatten()(retina_out)

hidden = Dense(intermediate_dim, kernel_regularizer=keras.regularizers.l1(reg),
               name='dense1', activation='relu', trainable=True)(flattened)
output = Dense(num_classes, name='dense2', activation='softmax', trainable=True)(hidden)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

if task == 'classification':
    model = Model(x, output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

else:
    sys.exit("No other task types besides classification configured yet")

#model.load_weights(model_path, by_name=True) -- Uncomment to load from saved model

model.summary()

if not data_augmentation:
    print('Not using data augmentation.')
    if task == 'classification':
        hist = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_test, y_test),
                         shuffle=True)

else:
    print('Using data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(x_train)

    if task == 'classification':
        hist = model.fit_generator(datagen.flow(x_train, y_train,
                                                batch_size=batch_size),
                                   steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                                   epochs=epochs,
                                   validation_data=(x_test, y_test),
                                   workers=4)

print('History', hist.history)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'SAVED'+'_'+model_name
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
np.save(os.path.join('Logs', f'{model_name}_VALACC.npy'), hist.history['val_acc'])
np.save(os.path.join('Logs', f'{model_name}_ACC.npy'), hist.history['acc'])

print(f'Saved trained model at {model_path}')
