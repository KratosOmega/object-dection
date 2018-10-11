######################################################################################################## imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import json
import cv2
import os
import ntpath
import cairo


######################################################################################################## Methods
def load_fileNames(data_path):
    fileNames = []
    for data in glob.glob(data_path):
        fileName = ntpath.basename(data)
        nameOnly = os.path.splitext(fileName)[0]
        fileNames.append(nameOnly)
    return fileNames

def getImgs(dir_path, fileNames, h, w):
    imgs = []
    for filename in fileNames:
        img = cv2.imread(dir_path + filename + '.jpg')
        resized = cv2.resize(img,(h, w))
        imgs.append(resized)
    return np.array(imgs)

def getLabels(dir_path, fileNames):
    labels = []
    for filename in fileNames:
        with open(dir_path + filename + '.json') as file:
            label = json.load(file)
            shape_map = labelCat.index(label['shape_type'])
            char_map = labelCat.index(label['character'])
            labels.append([shape_map, char_map])
    return labels

def getBboxWithLabels(imgs, data_labels, num_obj):
    num_imgs = len(imgs)
    bboxes = np.zeros((num_imgs, num_obj, 4))
    labels = np.zeros((num_imgs, num_obj), dtype=int)
    #num_shapes = 3
    #shape_labels = ['rectangle', 'circle', 'triangle']
    #colors = np.zeros((num_imgs, num_objects), dtype=int)
    #num_colors = 3
    #color_labels = ['r', 'g', 'b']

    for i_img in range(num_imgs):
        label = data_labels[i_img]
        for i_obj in range(num_obj):
            #shape = np.random.randint(num_shapes)
            #shapes[i_img, i_obj] = shape
            labels[i_img, i_obj] = data_labels[i_img][i_obj]
            if i_obj == 0:  # rectangle
                w, h = 28, 28
                x = 2
                y = 2
                bboxes[i_img, i_obj] = [x, y, w, h]
            elif i_obj == 1:  # circle
                w, h = 12, 12
                x = 12
                y = 12
                bboxes[i_img, i_obj] = [x, y, w, h]
    return bboxes, labels

def label_onehot(num_imgs, num_obj, num_labels, labels):
    onehot = np.zeros((num_imgs, num_obj, num_labels))
    for i_img in range(num_imgs):
        for i_obj in range(num_obj):
            onehot[i_img, i_obj, labels[i_img][i_obj]] = 1
    return onehot

def create_model(X, y):
    # TODO: Make one run with very deep network (~10 layers).
    filter_size = 3
    pool_size = 2

    # TODO: Maybe remove pooling bc it takes away the spatial information.

    model = Sequential([
        Convolution2D(32, 6, 6, input_shape=X.shape[1:], dim_ordering='tf', activation='relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Convolution2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
        # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
        Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
        # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
        Flatten(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(y.shape[-1])
    ])

    return model

# Flip bboxes during training.
# Note: The validation loss is always quite big here because we don't flip the bounding boxes for the validation data.
def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  # TODO: Check if its more performant if tensor elements are accessed directly below.
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U

def dist(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))


def train(model, train_X, train_y, test_X, test_y):
    num_epochs_flipping = 50
    num_epochs_no_flipping = 0  # has no significant effect

    flipped_train_y = np.array(train_y)
    flipped = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    ious_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    dists_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    mses_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    acc_shapes_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    #acc_colors_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))

    flipped_test_y = np.array(test_y)
    flipped_test = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    ious_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    dists_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    mses_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    acc_shapes_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    #acc_colors_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))

    # TODO: Calculate ious directly for all samples (using slices of the array pred_y for x, y, w, h).
    for epoch in range(num_epochs_flipping):
        print 'Epoch', epoch
        model.fit(train_X, flipped_train_y, nb_epoch=1, validation_data=(test_X, test_y), verbose=2)
        pred_y = model.predict(train_X)

        for sample, (pred, exp) in enumerate(zip(pred_y, flipped_train_y)):
            # TODO: Make this simpler.
            pred = pred.reshape(num_obj, -1)
            exp = exp.reshape(num_obj, -1)

            pred_bboxes = pred[:, :4]
            exp_bboxes = exp[:, :4]

            ious = np.zeros((num_obj, num_obj))
            dists = np.zeros((num_obj, num_obj))
            mses = np.zeros((num_obj, num_obj))
            for i, exp_bbox in enumerate(exp_bboxes):
                for j, pred_bbox in enumerate(pred_bboxes):
                    ious[i, j] = IOU(exp_bbox, pred_bbox)
                    dists[i, j] = dist(exp_bbox, pred_bbox)
                    mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))
            new_order = np.zeros(num_obj, dtype=int)

            for i in range(num_obj):
                # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
                ind_exp_bbox, ind_pred_bbox = np.unravel_index(ious.argmax(), ious.shape)
                ious_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
                dists_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
                mses_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
                ious[ind_exp_bbox] = -1  # set iou of assigned bboxes to -1, so they don't get assigned again
                ious[:, ind_pred_bbox] = -1
                new_order[ind_pred_bbox] = ind_exp_bbox

            flipped_train_y[sample] = exp[new_order].flatten()

            flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_obj, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
            ious_epoch[sample, epoch] /= num_obj
            dists_epoch[sample, epoch] /= num_obj
            mses_epoch[sample, epoch] /= num_obj

            acc_shapes_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_labels], axis=-1) == np.argmax(exp[:, 4:4+num_labels], axis=-1))
            #acc_colors_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_labels:4+num_labels+num_colors], axis=-1) == np.argmax(exp[:, 4+num_labels:4+num_labels+num_colors], axis=-1))

        # Calculate metrics on test data.
        pred_test_y = model.predict(test_X)
        # TODO: Make this simpler.
        for sample, (pred, exp) in enumerate(zip(pred_test_y, flipped_test_y)):
            # TODO: Make this simpler.
            pred = pred.reshape(num_obj, -1)
            exp = exp.reshape(num_obj, -1)

            pred_bboxes = pred[:, :4]
            exp_bboxes = exp[:, :4]

            ious = np.zeros((num_obj, num_obj))
            dists = np.zeros((num_obj, num_obj))
            mses = np.zeros((num_obj, num_obj))
            for i, exp_bbox in enumerate(exp_bboxes):
                for j, pred_bbox in enumerate(pred_bboxes):
                    ious[i, j] = IOU(exp_bbox, pred_bbox)
                    dists[i, j] = dist(exp_bbox, pred_bbox)
                    mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))

            new_order = np.zeros(num_obj, dtype=int)

            for i in range(num_obj):
                # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
                ind_exp_bbox, ind_pred_bbox = np.unravel_index(mses.argmin(), mses.shape)
                ious_test_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
                dists_test_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
                mses_test_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
                mses[ind_exp_bbox] = 1000000#-1  # set iou of assigned bboxes to -1, so they don't get assigned again
                mses[:, ind_pred_bbox] = 10000000#-1
                new_order[ind_pred_bbox] = ind_exp_bbox

            flipped_test_y[sample] = exp[new_order].flatten()

            flipped_test[sample, epoch] = 1. - np.mean(new_order == np.arange(num_obj, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
            ious_test_epoch[sample, epoch] /= num_obj
            dists_test_epoch[sample, epoch] /= num_obj
            mses_test_epoch[sample, epoch] /= num_obj

            acc_shapes_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_labels], axis=-1) == np.argmax(exp[:, 4:4+num_labels], axis=-1))
            #acc_colors_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_labels:4+num_labels+num_colors], axis=-1) == np.argmax(exp[:, 4+num_labels:4+num_labels+num_colors], axis=-1))

        print 'Flipped {} % of all elements'.format(np.mean(flipped[:, epoch]) * 100.)
        print 'Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch]))
        print 'Mean dist: {}'.format(np.mean(dists_epoch[:, epoch]))
        print 'Mean mse: {}'.format(np.mean(mses_epoch[:, epoch]))
        print 'Accuracy Class: {}'.format(np.mean(acc_shapes_epoch[:, epoch]))
        #print 'Accuracy colors: {}'.format(np.mean(acc_colors_epoch[:, epoch]))

        print '--------------- TEST ----------------'
        print 'Flipped {} % of all elements'.format(np.mean(flipped_test[:, epoch]) * 100.)
        print 'Mean IOU: {}'.format(np.mean(ious_test_epoch[:, epoch]))
        print 'Mean dist: {}'.format(np.mean(dists_test_epoch[:, epoch]))
        print 'Mean mse: {}'.format(np.mean(mses_test_epoch[:, epoch]))
        print 'Accuracy Class: {}'.format(np.mean(acc_shapes_test_epoch[:, epoch]))
        #print 'Accuracy colors: {}'.format(np.mean(acc_colors_test_epoch[:, epoch]))
        print

        shape_acc = np.mean(acc_shapes_test_epoch[:, epoch])
        #color_acc = np.mean(acc_colors_test_epoch[:, epoch])

#        if(shape_acc > 0.95 and color_acc > 0.95):
        if(shape_acc > 0.95):
            break
        model.save_weights('./model_' + str(shape_acc) + "_" + ".h5")
#        model.save_weights('./model_' + str(shape_acc) + "_" + str(color_acc) + ".h5")

######################################################################################################## Global Variables
labelCat = [
    'circle', 'semicircle', 'quartercircle', 'triangle', 'square', 'rectangle', 'trapezoid', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'star', 'cross',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

dir_path = './cnn_training_data/'
data_path = './cnn_training_data/*.jpg';
img_size = 32
fileNames = load_fileNames(data_path)
data = getImgs(dir_path, fileNames, img_size, img_size)
data_labels = getLabels(dir_path, fileNames)
num_imgs = len(data)
num_obj = 2
num_labels = len(labelCat)
bboxes, labels = getBboxWithLabels(data, data_labels, 2)
num_colors = 3

######################################################################################################## Get One_Hot
#colors_onehot = np.zeros((num_imgs, num_obj, 3))
labelOnehot = label_onehot(num_imgs, num_obj, num_labels, labels)

X = (data - 128.) / 255.
#y = np.concatenate([bboxes / img_size, labelOnehot, colors_onehot], axis=-1).reshape(num_imgs, -1)
y = np.concatenate([bboxes / img_size, labelOnehot], axis=-1).reshape(num_imgs, -1)

######################################################################################################## Get Train & Validation
i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = data[i:]
test_bboxes = bboxes[i:]

######################################################################################################## Execute Training Process
model = create_model(X, y)
model.compile('adadelta', 'mse')
train(model, train_X, train_y, test_X, test_y)

########################################################################################################

