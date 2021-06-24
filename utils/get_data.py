#* ORIGINAL WORK LICENSE STATEMENT
#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Original Authors: Batuhan Toemekce, Burak Kaya, Michael Hersche            *
#*----------------------------------------------------------------------------*

# MODIFICATIONS LICENSE STATEMENT
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

__version__ = '1.0'

"""
 Loads (and downloads if necessary) data form Physionet's EEG-MI dataset.
 The user can choose the subjects to use (some of them are automatically
 excluded), the number of classes (L/R, L/R/0, L/R/0/F), the channels, the 
 time window and the downsampling factor (if this is used a data augmentation
 technique is performend recycling the discarded samples, generating new
 trainable examples).

 The modifications are:
    - The use of mne as the Python module to read .edf files.
    - The get_data functions and their dependencies have been adapted to depend
      on more parameters, concretely, the channels, the time window, the
      downsampling factor, the normalization boolean and the rest data selection
      method. This allows the creation of a more customizable dataset.
    - A subject-realted numpy array is also returned, identifying the subject of
      the selected trial.
    - The downsampling factor does not discard samples, since it is coded to
      perform a data augmentation technique, adding this samples as new trials.
    - The read_data function calls the download_function that downloads the
      requested file it is not located in the path.
    - The random_rest arg of get_data can be used to change the rest data
      selection method between linespaced (all the possible) or random.
"""

from typing import Tuple
import os
import subprocess
import numpy as np
from sklearn.utils import shuffle
import mne

mne.set_log_level('error')


def get_data(path: str, subjects: list = list(range(1,110)), n_classes: int = 4,
             chans: list = list(range(64)), T: int = 3, ds: int = 1,
             bp_filter: bool = False, random_rest: bool = False,
             verbosity: bool = True) -> Tuple[np.array, np.array, np.array]:
    """Generates the dataset from the Physionet EEG-MI dataset.

    It allows a lot of customization options, in the number of subjects,
    classes, EEG channels, time widow, downsampling and normalization. If the
    dataset folder is empty it will dowload the requested files.

    Args:
        path (str): Path to the dataset folder containing the de /S001/-/S109/
            subject folders. If it is empty, the required files will be
            downloaded using :func:`~download_data`. Check its dependencies.
        subjects (list, optional): List of subjects to add to the dataset.
            Defaults to list(range(1,110)).
        n_classes (int, optional): Number of classes to use in the dataset. If 
            it is 2 it uses L/R, with 3 L/R/0 and L/R/0/F is used with 4. 
            Defaults to 4.
        chans (list, optional): List of the EEG channels to use. Defaults to
            list(range(64)).
        T (int, optional): Time window of the trials must be an integer between 
            1 and 6. Defaults to 3.
        ds (int, optional): Downsampling factor. If it is used, the discarded
            samples in a trial are added as new trial, as a data augmentation
            technique. Must be an integer between 1 and 3. Defaults to 1.
        bp_filter (bool, optional): Performs a bandpass filter between 7 Hz and
            30 Hz in the EEG signals. Defaults to False.
        random_rest (bool, optional): If True all the rest ('0' class) are
            selected randomly in the first run. Else, all the possible trials
            are selected linespaced and the reaming are random. Defaults to
            False.
        verbosity (bool, optional): Print the progress if True. Defaults to
            True.

    Returns:
        Tuple[np.array, np.array, np.array]:
            X: A :code:`[len(subjects)*21*ds, len(chans), 160*T/ds]` np.array
                with the dataset's input data, EEG recordings of each trial.
            y: A :code:`[len(subjects)*21*ds]` with the labels of each trial.
            subs: A :code:`[len(subjects)*21*ds]` with the subject number of
                each trial.
    """

    # Define subjects whose data is not taken, for details see data tester added 106 again to analyze it, deleted from the exluded list
    excluded_subjects = [88,92,100,104]
    
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects = [x for x in subjects if (x not in excluded_subjects)]

    #Motor-imagery runs numbers
    mi_runs = [1, 4, 6, 8, 10, 12, 14]

    # Extract only requested number of classes
    if(n_classes == 3):
        print('Returning 3 Class data')
        mi_runs.remove(6) # feet
        mi_runs.remove(10) # feet
        mi_runs.remove(14) # feet

    elif(n_classes == 2):
        print('Returning 2 Class data')
        mi_runs.remove(6) # feet
        mi_runs.remove(10) # feet
        mi_runs.remove(14) # feet
        mi_runs.remove(1) #rest 

    print(f'Data from runs: {mi_runs}')

    X, y, subs = read_data(path, subjects=subjects, runs=mi_runs, chans=chans,
                       T=T, ds=ds, bp_filter=bp_filter, random_rest=random_rest, verbosity=verbosity)
        
    return X, y, subs
     
def read_data(path: str, subjects: list, runs: list, chans: list,T: int, ds: int,
     bp_filter: bool, random_rest: bool, verbosity: bool) -> Tuple[np.array, np.array, np.array]:
    """Reads the data of the EEG-MI dataset and process it.

    Here the data is read form the Physionet EEG-MI dataset. It selects the
    trials depending on the runs and the time window. If the downsampling factor
    is 2 or 3, the EEG readings are downsampled, using the discarded samples as 
    new trials. The data is located by download_data, which downloads it if not 
    present.

    Args:
        path (str): Path to the dataset folder containing the de /S001/-/S109/
            subject folders. If it is empty, the required files will be
            downloaded using :func:`~download_data`. Check its dependencies.
        subjects (list): List of subjects to add to the dataset
        runs (list): List of the runs to use.
        chans (list): List of the EEG channels to use.
        T (int): Time window of the trials must be an integer between 
            1 and 6.
        ds (int): Downsampling factor. If it is used, the discarded
            samples in a trial are added as new trial, as a data augmentation
            technique. Must be an integer between 1 and 3.
        bp_filter (bool): Performs a bandpass filter between 7 Hz and 30 Hz in
            the EEG signals. Defaults to False.
        random_rest (bool): If True all the rest ('0' class) are selected
            randomly in the first run. Else, all the possible trials are
            selected linespaced and the reaming are random. Defaults to False.
        verbosity (bool): Print the progress if True. Defaults to True.

    Raises:
        ValueError: If ds is not either 1, 2 or 3.

    Returns:
        Tuple[np.array, np.array, np.array]: 
            X: A :code:`[len(subjects)*21*ds, len(chans), 160*T/ds]` np.array
                with the dataset's input data, EEG recordings of each trial.
            y: A :code:`[len(subjects)*21*ds]` with the labels of each trial.
            subs: A :code:`[len(subjects)*21*ds]` with the subject number of
                each trial.
    """     

    """
    DATA EXPLANATION:
        
        LABELS:
        both first_set and second_set
            T0: rest
        first_set (imagined motion in runs 4, 8, and 12)
            T1: the left fist 
            T2: the right fist
        second_set (imagined motion in runs 6, 10, and 14)
            T1: both fists
            T2: both feet
        
        Here, we get data from the first run, baseline (rest), first_set (left
        fist and right fist) and also data from the second_set (both feet). We
        ignore data for T1 from the second_set and thus return data for four
        classes/categories of events: Rest, Left Fist, Right Fist, Both Feet.
    """
    
    # Define runs where the two different sets of tasks were performed
    baseline = np.array([1])
    first_set = np.array([4,8,12])
    second_set = np.array([6,10,14])
    
    # Number of EEG channels
    NO_channels = len(chans)

    # Number of Trials extracted per Run, thus 21 per class in a same subject.
    NO_trials = 7
    
    # Define Sample size per Trial. The ds factor will be introduced later.
    n_samples = int(160*T)
    
    # initialize empty arrays to concatenate with itself later
    X = np.empty((0, NO_channels, n_samples))
    y = np.empty(0)
    subs = np.empty(0)
    
    for subj_index, subject in enumerate(subjects):

        if verbosity: print('Processing subject %03d %d/%d...' % (subject,
                                                                  subj_index+1,
                                                                  len(subjects)
                                                                  ), end='')
        
        np.random.seed(7) # The seed is set to be the same for every subject

        for run in runs:
            #For each run, a certain number of trials from corresponding classes should be extracted
            counter_0 = 0
            counter_L = 0
            counter_R = 0
            counter_F = 0

            # Download the file if necessary and get its name
            file_name = download_data(path, subject, run)
            
            # Read file
            f=mne.io.read_raw_edf(file_name, preload=True)

            if bp_filter: 
                f.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

            info = f.info

            # Signal Parameters - measurement frequency
            fs = info['sfreq']

            #Get Raw Data
            sigbufs = f.get_data()

            #Chosse the channels
            sigbufs = sigbufs[chans, :]

            # Get Label information
            annotations=mne.events_from_annotations(f)

            # close the file
            f.close()
            
            # Get the specific label information
            labels = annotations[0][:,2]
            points = annotations[0][:,0]
            
            labels_int = np.empty(0)
            subs_in_run = np.empty(0)
            data_step = np.empty((0,NO_channels, n_samples))             
            
            if run in second_set:
                for ii in range(np.size(labels)):
                    if(labels[ii] == 1 and counter_0 < NO_trials):
                        continue
                        counter_0 += 1
                        labels_int = np.append(labels_int,[2])
                        
                    elif(labels[ii] == 3 and counter_F < NO_trials):
                        counter_F += 1
                        labels_int = np.append(labels_int,[3])
                        subs_in_run = np.append(subs_in_run, [subject])
                        # change data shape and seperate events
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))        
                
            elif run in first_set:
                for ii in range(np.size(labels)):
                    if(labels[ii] == 1 and counter_0 < NO_trials):
                        continue
                        counter_0 += 1
                        labels_int = np.append(labels_int, [2])
                        
                    elif(labels[ii] == 2 and counter_L < NO_trials):
                        counter_L += 1
                        labels_int = np.append(labels_int, [0])
                        subs_in_run = np.append(subs_in_run, [subject])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))
                        
                    elif(labels[ii] == 3 and counter_R < NO_trials):
                        counter_R += 1
                        labels_int = np.append(labels_int, [1])
                        subs_in_run = np.append(subs_in_run, [subject])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))
                        
            elif run in baseline:
                if random_rest:
                    for ii in range(NO_trials*3):  
                        index = np.random.randint(0,(60-T)*fs)
                        labels_int = np.append(labels_int, [2])
                        subs_in_run = np.append(subs_in_run, [subject])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,(index):(index+n_samples)])[None]))
                else:
                    for ii in range(min(int(60/T),NO_trials*3)): # The maximun number of rest trials we want is NO_trials*3
                        labels_int = np.append(labels_int, [2])
                        subs_in_run = np.append(subs_in_run, [subject])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,(ii*160):(ii*160+n_samples)])[None]))
                    for ii in range(NO_trials*3-int(60/T)): #If negative, the loop is passed
                        index = np.random.randint(0,(60-T)*fs)
                        labels_int = np.append(labels_int, [2])
                        subs_in_run = np.append(subs_in_run, [subject])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,(index):(index+n_samples)])[None]))
            
            # concatenate arrays in order to get the whole data in one input array    
            X = np.concatenate((X,data_step))
            y = np.concatenate((y,labels_int))
            subs = np.concatenate((subs,subs_in_run))
        
        if verbosity: print('Ok')

    if ds==2: 
        X = np.concatenate((X[:,:,0::2], X[:,:,1::2])) #It doesn't need the extra indexing since the sample freq is even
        y = np.concatenate((y,y))
        subs = np.concatenate((subs,subs))
    elif ds==3: #An extra indexing is used to make the sets of subsamples of the same number of frames, since fs*T/ds doesn't have to be an integer
        X = np.concatenate((X[:,:,0::3][:,:,:int(fs*T/ds)], X[:,:,1::3][:,:,:int(fs*T/ds)], X[:,:,2::3][:,:,:int(fs*T/ds)]))
        y = np.concatenate((y,y,y))
        subs = np.concatenate((subs,subs,subs))
    elif ds!=1:
        raise ValueError('ds must be 1, 2 or 3')

    return X, y, subs

def download_data(path: str, subject: int , run: int) -> str:
    """Downloads the requested file if it doesn't exist yet, returning its path.

    First it checks if the file exits in the /SXXX/SXXXRXX.edf directory and if 
    doesn't it downloads it from the
    `https://archive.physionet.org/pn4/eegmmidb/`_ archive. Either the files has
    been downloaded or it already existed, the function return its directory.

    .. note::
        To download the file it used the curl bash command.

    Args:
        path (str): Path to the dataset folder containing the de /S001/-/S109/
            subject folders or where they must be downloaded.
        subject (int): Subject of the file to seek.
        run (int): Run of the file to seek.

    Returns:
        str: The directory of the requested .edf file.
    """    

    file_name = 'S%03d/S%03dR%02d.edf' % (subject, subject, run)
    if os.path.isfile(path+file_name):
        return path+file_name
    else:

        try:
            os.makedirs(path+'S%03d/' % subject)
        except FileExistsError:
            pass
        
        subprocess.check_call(['curl', 'https://archive.physionet.org/pn4/eegmmidb/'+file_name, '-o', path+file_name], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        #os.system('curl https://archive.physionet.org/pn4/eegmmidb/%s -o %s' % (file_name, path+file_name))
        return path+file_name

def normalize(X, Y, S, normalization, norm_range):

    Y=np.vstack([Y,S])
    Y=np.transpose(Y)
    
    a=norm_range[0]
    b=norm_range[1]
    ## Normalize
    if normalization=='None':
        X=X
    if normalization=='Linear':
        x_min= np.amin(X)
        x_max= np.amax(X)
        X = (b-a)*( (X- x_min)/(x_max-x_min) ) +a

    if normalization=='Channel':
        min_max=np.zeros((64,2))
        for channel in range(64):
            min_max[channel,0]=np.amin(X[:,channel,:])
            min_max[channel,1]=np.amax(X[:,channel,:])
        for channel in range(64):
            X[:,channel,:]= (b-a)*( (X[:,channel,:]-min_max[channel,0]) / (min_max[channel,1]-min_max[channel,0])  ) +(a)
        
    ## Reshape
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))  
    return X, Y


# Example of usage
"""
X, y, subs = get_data('./Dataset/', T=1, ds=3)

try:
    os.mkdir(processedPath)
except FileExistsError:
    pass

np.save(processedPath+'samples.npy', X)
np.save(processedPath+'labels.npy', y)
np.save(processedPath+'subs.npy', subs)

print(X.shape, y.shape, subs.shape)

"""