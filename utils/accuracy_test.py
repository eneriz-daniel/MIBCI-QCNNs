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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def accuracy_test(modelPath: str) -> None:
    """A simple function to evaluate the accuracy of the Keras model and its HLS
    implementation using the Vivado HLS C simulation output files.

    Args:
        modelPath (str): Path to the model to be evaluated.
    """

    active_folds = 5

    acc_HLS = np.empty(active_folds)
    acc_keras= np.empty(active_folds)


    for fold in range(active_folds):
        y_float = np.load('{}/fold_{}/validationDS/y_pred.npy'.format(modelPath, fold))
        y_true = np.load('{}/fold_{}/validationDS/y_true.npy'.format(modelPath, fold))
        y_fixed = np.loadtxt('{}/fold_{}/validationDS/y_hls_16_8.txt'.format(modelPath, fold), usecols=[0])

        acc_HLS[fold] = accuracy_score(y_true[:len(y_fixed)], y_fixed)
        acc_keras[fold] = accuracy_score(y_true[:len(y_fixed)], y_float[:len(y_fixed)])

    plt.plot(100*acc_HLS, '.', label='HLS {:.2f}%'.format(100*acc_HLS.mean()))
    plt.plot(100*acc_keras, '.', label='Keras {:.2f}%'.format(100*acc_keras.mean()))

    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.title('Global 5-folds result')
    plt.legend()
    plt.savefig('{}/Global-acc-HLS.png'.format(modelPath))
