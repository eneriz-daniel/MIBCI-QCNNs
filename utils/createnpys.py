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

# This file creates the npy files containing the trained global models
# parameters, saving them in the same directory that the Cross Validation
# folds, under the subdirectory npyparams. The validation data subset is
# also saved in the subdirectory validationDS.

from tensorflow.keras.models import load_model
import numpy as np
from os import mkdir, listdir
from shutil import copytree, rmtree
from parse import *
from sklearn.metrics import accuracy_score


def createnpys(modelPath: str) -> None:
    """This functions prepares the model path to be ready for its Vivado HLS C
    simulation. It generates the validation set for each fold in a
    `validationDS` subfolder and the fold's model parameters are saved under the
    `npyparams`subfolder.

    Args:
        modelPath (str): Path to the model to prepare to be valdiated using HLS
        C simulation.
    """
    
    npyParamsNames = ['conv2d_w', 'depthconv2d_w', 'sepdepthconv2d_w', 'seppointconv2d_w', 'dense_w', 'dense_b']

    for fold in range(5):

        print('PROCESSING FOLD {}/5'.format(fold+1))

        model = load_model(modelPath+'fold_{}/weights-improvement.tf'.format(fold))

        try:
            mkdir(modelPath+'fold_{}/npyparams/'.format(fold))
        except(FileExistsError):
            pass

        for j, name in enumerate(npyParamsNames):
            np.save(modelPath+'fold_{}/npyparams/{}.npy'.format(fold, name), model.get_weights()[j])

        Y = np.load(modelPath+'Y.npy')
        test_subjects = np.unique(Y[:,1])[fold*21:(fold+1)*21]

        X = np.load(modelPath+'normX.npy')[np.isin(Y[:,1], test_subjects)]
        y = Y[np.isin(Y[:,1], test_subjects)]

        y_pred = np.argmax(model(np.expand_dims(X, -1)), 1)

        try:
            mkdir(modelPath+'fold_'+str(fold)+'/validationDS/')
        except(FileExistsError):
            pass

        try:
            mkdir(modelPath+'fold_'+str(fold)+'/validationDS/X_samples/')
        except(FileExistsError):
            pass

        np.save(modelPath+'fold_'+str(fold)+'/validationDS/X.npy', X)
        for i in range(X.shape[0]):
            np.save(modelPath+'fold_'+str(fold)+'/validationDS/X_samples/X_{}.npy'.format(i), X[i,:,:])
        np.save(modelPath+'fold_'+str(fold)+'/validationDS/y_true.npy', y[:,0])
        np.save(modelPath+'fold_'+str(fold)+'/validationDS/y_pred.npy', y_pred)

        print(accuracy_score(y[:,0], y_pred))