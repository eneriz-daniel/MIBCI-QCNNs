# Copyright (C) 2021 Daniel Enériz Orta and Ana Caren Hernández Ruiz
# 
# This file is part of MIBCI-QCNNs.
# 
# MIBCI-QCNNs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# MIBCI-QCNNs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MIBCI-QCNNs.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from typing import List
import time
import os
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, LeakyReLU, AveragePooling2D, DepthwiseConv2D, SeparableConv2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def normalize(dataset: str, norm_range: List[float] = [-1,1]) -> List[np.ndarray]:
    """Takes a get_data-geenrated dataset path and normalizes it by channel
    using a given normalization range.

    Args:
        dataset (str): The path containing the get_data-generated dataset, with
                       a `samples.npy`, `labels.npy` and `subs.npy` files.
        norm_range (List[int]): Normalization range, the default is [-1, 1]

    Returns:
        List[np.ndarray]: The samples normalized
                          by channels with the third dimmension expanded and an
                          array with the label and subject of each sample.
    """

    X = np.load(dataset+'samples.npy')
    Y = np.load(dataset+'labels.npy')
    S = np.load(dataset+'subs.npy')    
    Y = np.vstack([Y,S])
    Y = np.transpose(Y)

    a = norm_range[0]
    b = norm_range[1]

    min_max = np.zeros((64,2))
    min_max[:,0] = X.min(axis=(0,2))
    min_max[:,1] = X.max(axis=(0,2))

    for chan in range(64):
        X[:, chan, :]= (b-a)*((X[:, chan, :]-min_max[chan,0])/(min_max[chan,1] - min_max[chan,0])) + a
    
    X = np.expand_dims(X, -1)

    return X, Y

def data_loader(global_fold: int, samples: np.ndarray, labels: np.ndarray, classes: int) -> List[np.ndarray]:
    """Takes the dataset, the fold and the number of classes and creates the
    training set and validation set of that fold.

    Args:
        global_fold (int): The fold of the CV
        samples (np.ndarray): Samples of the entire dataset
        labels (np.ndarray): Labels of the entire dataset
        classes (int): Number of classes

    Returns:
        List[np.ndarray]: A list with the training set samples and lables,
                          validation set samples and labels and subjects of
                          the validation set.
    """
    test_subjects = np.unique(labels[:,1])[global_fold*21:(global_fold+1)*21]
    
    testmask = np.isin(labels[:,1], test_subjects)
    trainmask = np.logical_not(testmask)

    train_samples = samples[trainmask]
    train_labels = labels[trainmask]
    test_samples = samples[testmask]
    test_labels = labels[testmask]

    train_samples, train_labels= shuffle(train_samples, train_labels, random_state=0)
    test_subjects=test_labels[:,1]
    train_labels=to_categorical(train_labels[:,0], num_classes=classes)
    test_labels=to_categorical(test_labels[:,0], num_classes=classes)

    return train_samples, train_labels, test_samples, test_labels, test_subjects

def get_model(T: int, ds: int, Nchans: int, Nclasses: int, fs: float = 160) -> tf.keras.Model:
    """Creates the EEG-based model for the selected data reduction settings.

    Args:
        T (int): Time window in seconds used to generate the dataset. Is an
                 `int` since we only evaluated 1, 2 and 3 seconds time windows 
        ds (int): Downsampling factor used to generate the dataset.
        Nchans (int): Number of channels used in the dataset.
        Nclasses (int): Number of classes.
        fs (float): Sampling frequency in Hz. Defaults to 160 Hz.

    Returns:
        tf.keras.Model: The EEG-based model.
    """
    classifier = Sequential()

    classifier.add(InputLayer(input_shape=(Nchans, int(fs*T/ds), 1)))
    classifier.add(Conv2D(
                   filters=4, 
                   kernel_size=(1, int(fs/(2*ds))),
                   activation=None,
                   padding='same',
                   use_bias=False))
    classifier.add(LeakyReLU(alpha=0.6))
    classifier.add(DepthwiseConv2D(
                   kernel_size=(Nchans,1),
                   depth_multiplier = 2,
                   use_bias = False,
                   padding = 'valid'))
    classifier.add(LeakyReLU(alpha=0.5))
    classifier.add(AveragePooling2D((1,int(6/ds))))

    classifier.add(SeparableConv2D(
                   filters=8,
                   kernel_size=(1, 16),
                   use_bias = False,
                   padding = 'same'))
    classifier.add(LeakyReLU(alpha=0.4))
    classifier.add(AveragePooling2D((1,8)))

    classifier.add(Flatten())       
    classifier.add(Dense(Nclasses,
                   activation='softmax'))

    return classifier

def lr_scheduler(epoch, lr):
    """A simple learning rate scheduler.
    """
    if (epoch < 20):
        lr = 1e-2
    elif (epoch <40):
        lr = 1e-2/5
    elif (epoch < 60):
        lr = 1e-2/10
    elif (epoch < 80):
        lr = 1e-2/50
    else:
        lr = 1e-2/100
    return lr

def train_global(datapath: str, folds: int = 5, fs: float = 160, T: int = 3,
                 ds: int = 1, Nclasses: int = 4, Nchans: int = 64,
                 epochs: int = 100, LR: float = 1e-2, batch_size: int = 16) -> None:
    """Trains the EEG-based model, crates the training evolution and stats of
    each fold and computes the global accuracy.

    Args:
        datapath (str): Path of the get_data-generated dataset.
        folds (int, optional): Number of global folds. Defaults to 5.
        fs (float, optional): Sampling frequency in Hertzes. Defaults to 160.
        T (int, optional): Time window of the selected dataset. Defaults to 3.
        ds (int, optional): Downsamplig factor applied to the dataset. Defaults to 1.
        Nclasses (int, optional): Number of classes of the dataset. Defaults to 4.
        Nchans (int, optional): Number of EEG channels used in the dataset. Defaults to 64.
        epochs (int, optional): Number of epochs to train each fold. Defaults to 100.
        LR (float, optional): Initial learning rate. Defaults to 1e-2.
        batch_size (int, optional): Batch size. Defaults to 16.
    """

    #The path to save the trained model is created:
    saved_model_path = 'Global_T{}s_ds{}_{}chans_{}classes-{}'.format(T, ds, Nchans, Nclasses, time.strftime('%Y-%m-%d_%H-%M'))

    os.mkdir(saved_model_path)

    #Normalization of the dataset
    X, Y = normalize(datapath, [-1, 1])

    np.save(saved_model_path+'/normX.npy', X)
    np.save(saved_model_path+'/Y.npy', Y)

    #Arrays to save the accuracy data of each fold in order to compute they mean at the end
    val_acc_data = np.empty(folds)
    acc_data = np.empty(folds)

    #Iterating over the CV folds
    for fold in range(folds):

        print('TRAINING FOLD {}/{}'.format(fold+1, folds))

        t0 = time.time()
        init_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())

        #The fold's folder
        os.mkdir('{}/fold_{}'.format(saved_model_path, fold))

        #Splitting the data for this fold
        train_samples, train_labels, test_samples, test_labels, _ = data_loader(fold, X, Y, Nclasses)

        #Instantation of the model
        classifier = get_model(T, ds, Nchans, Nclasses)
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #The callback to save the best weights
        model_checkpoint_callback = ModelCheckpoint(
            filepath='{}/fold_{}/weights-improvement.tf'.format(
                saved_model_path,
                fold),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        
        #Training the model
        classifier_train = classifier.fit(
            x = train_samples,
            y = train_labels,
            epochs=epochs,
            validation_data=(test_samples, test_labels),
            shuffle = True,
            batch_size=batch_size,
            use_multiprocessing = False,
            workers = 8,
            callbacks = [model_checkpoint_callback,
                        LearningRateScheduler(lr_scheduler)]
        )
        

        t1 = time.time() - t0
        
        #Plotting the training evolution
        fig, ax1 = plt.subplots()
        ax2 = plt.twinx(ax1)

        ax1.plot(classifier_train.history['accuracy'], label='Train')
        ax1.plot(classifier_train.history['val_accuracy'], label='Valid')
        ax2.plot(classifier_train.history['loss'], '--')
        ax2.plot(classifier_train.history['val_loss'], '--')

        ax1.set_ylabel('Accuracy')
        ax2.set_ylabel('Loss (- -)')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        fig.suptitle('Training evolution')

        plt.savefig('{}/fold_{}/training-ev.png'.format(saved_model_path, fold))
        
        #Saving the accuracy data
        val_acc = np.asarray(classifier_train.history['val_accuracy'])
        acc = np.asarray(classifier_train.history['accuracy'])
        val_acc_data[fold] = val_acc.max()
        acc_data[fold] = acc[val_acc==val_acc.max()][-1]

        #Saving the details of the training
        classifier = tf.keras.models.load_model('{}/fold_{}/weights-improvement.tf'.format(saved_model_path, fold))
        with open('{}/fold_{}/Resume.txt'.format(saved_model_path, fold), 'w') as f:
            f.write(init_time)
            f.write('\nTraining time: {} mins\n\n'.format(t1/60))
            f.write('T: {} s\nds: {}\nNchans: {}\nNclasses: {}\n\n'.format(T, ds, Nchans, Nclasses))
            f.write('Epochs: {}\nLearning rate: {}\nBath size: {}\n\n'.format(epochs, LR, batch_size))
            classifier.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write('\n\nAcc: {}\n'.format(val_acc_data[fold]))
            f.write('Confusion matrix:\n{}'.format(confusion_matrix(np.argmax(test_labels, 1), np.argmax(classifier(test_samples),1))))
            f.write('\nClassification report\n{}'.format(classification_report(np.argmax(test_labels, 1), np.argmax(classifier(test_samples),1))))

    #Plotting the the 5-fold accuracy data and computing they means
    np.save('{}/val_acc.npy'.format(saved_model_path), val_acc_data)
    np.save('{}/train_acc.npy'.format(saved_model_path), acc_data)

    plt.figure()

    plt.plot(100*val_acc_data, '.', label='Valid {:.2f}%'.format(100*val_acc_data.mean()))
    plt.plot(100*acc_data, '.', label='Train {:.2f}%'.format(100*acc_data.mean()))

    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.title('Global 5-folds result')
    plt.legend()
    plt.savefig('{}/Global-acc.png'.format(saved_model_path))

    os.system('mv {} {:.2f}-{}'.format(saved_model_path, 100*val_acc_data.mean(), saved_model_path))
