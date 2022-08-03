#@title MNE
"""
MODIFIED BY Goragod P. Hosei University
From the work of Francesco Mattioli (https://github.com/Kubasinska/MI-EEG-1D-CNN/blob/9495d00addc95f7045b9031f175cbdcf3abc73d8/dataset_generator/generator.py#L47)
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

"""
"""
THIS IS THE MAIN GENERATOR
"""

from mne.io import BaseRaw, read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import numpy as np
import mne
from typing import List, Tuple

from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from sklearn.model_selection import train_test_split

def rename_ch(raw:BaseRaw)-> BaseRaw:
    old_ch_name = raw.ch_names
    new_ch_name = [ch.replace('.', '').upper() for ch in old_ch_name]
    new_ch = {key:val for key, val in zip(old_ch_name, new_ch_name)}
    raw.rename_channels(new_ch)

    return raw

def load_data(subjects:List, runs:List, max_duration: int = 124, chs: List = None) -> List[List[BaseRaw]]:
    all_subject_list = []
    
    # read each subject and each run
    for subject in subjects:
        single_subject_run = []
        for run in runs:
            #create file name
            f = eegbci.load_data(subject, run)
            raw_run = read_raw_edf(f[0], preload=True, verbose=False)
            #Cleanup channel name
            raw_run = rename_ch(raw_run)
            #Change annotaion name
            if np.sum(raw_run.annotations.duration) > max_duration:
                # print("Run %d of subject %d has duration less than %d, cropped" % (run, subject, max_duration))
                raw_run.crop(tmax = max_duration)

            #change annotation from T0,T1 and T2 to more meaningful
            if run in [3, 7, 11]: #Task 1
                for i, annotation in enumerate(raw_run.annotations.description):
                    if annotation == 'T0':
                        raw_run.annotations.description[i] = 'R'
                    if annotation == 'T1':
                        raw_run.annotations.description[i] = 'ML'
                    if annotation == 'T2':
                        raw_run.annotations.description[i] = 'MR'
            if run in [4, 8, 12]: #Task 2
                for i, annotation in enumerate(raw_run.annotations.description):
                    if annotation == 'T0':
                        raw_run.annotations.description[i] = 'R'
                    if annotation == 'T1':
                        raw_run.annotations.description[i] = 'IL'
                    if annotation == 'T2':
                        raw_run.annotations.description[i] = 'IR'
            if run in [5, 9, 13]: #Task 3
                for i, annotation in enumerate(raw_run.annotations.description):
                    if annotation == 'T0':
                        raw_run.annotations.description[i] = 'R'
                    if annotation == 'T1':
                        raw_run.annotations.description[i] = 'MH'
                    if annotation == 'T2':
                        raw_run.annotations.description[i] = 'MF'
            if run in [6, 10, 14]: #Task 4
                for i, annotation in enumerate(raw_run.annotations.description):
                    if annotation == 'T0':
                        raw_run.annotations.description[i] = 'R'
                    if annotation == 'T1':
                        raw_run.annotations.description[i] = 'IH'
                    if annotation == 'T2':
                        raw_run.annotations.description[i] = 'IF'

            #Select channel
            if chs is not None:
                if len(chs) >= 1:
                    raw_run.pick_channels(chs)

            single_subject_run.append(raw_run)
        all_subject_list.append(single_subject_run)
    return all_subject_list



def epochs(raws: List[BaseRaw], exclude_base: bool = False, normal_by_sub = False,
            tmin :int = 0, tmax: int = 4, verbose=False) -> Tuple[np.ndarray, List]:
    xs = []
    ys = []
    ALL_EVENT_ID = dict(R = 0, ML = 1, MR = 2, IL = 3, IR = 4, MH = 5, MF = 6, IH = 7, IF = 8)

    for i, raw in enumerate(raws):
        # create key denpend on the number of exisiting class
        for run in raw:
            event_annotation = np.unique(run.annotations.description)
            event_id = dict.fromkeys(event_annotation, None)
            event_id = {key:val for key,val in zip(ALL_EVENT_ID.keys(), ALL_EVENT_ID.values()) if key in event_annotation}
            if exclude_base:
                del event_id['R']
            # print(event_id, run)
            events, _ = mne.events_from_annotations(run, event_id = event_id, verbose=verbose)
            picks = mne.pick_types(run.info, meg = False, eeg = True, eog = False, exclude = 'bads')
            ep = Epochs(run, events, event_id, tmin, tmax, proj = True, picks = picks, baseline = None, preload = True, verbose=verbose)
            # print(event_id)
            # print(ep.get_data().shape, ep.events.shape)
            data = ep.get_data()
            if normal_by_sub:
                normalize(data)
            xs.append(eeg_tensor_format(data))
            ys.append(ep.events)            

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis = 0)

def to_one_hot(y, by_sub=False):
    new_array = y.copy()
    total_labels = np.unique(new_array)
    # print(total_labels)
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x
    for x in range(len(new_array)):
        new_array[x] = mapping[new_array[x]]

    return tf.keras.utils.to_categorical(new_array)


def normalize(x:np.ndarray):
    reshape_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    norm = minmax_scale(reshape_x, axis = 1)
    return np.reshape(norm, (x.shape[0], x.shape[1], x.shape[2]))

def eeg_tensor_format(x:np.ndarray)->np.ndarray:
    return np.transpose(x, (0,2,1)).astype(np.float64)

def mne_train_test_split(x, y, perc):
    from numpy.random import default_rng
    rng = default_rng()
    test_x = list()
    train_x = list()
    train_y = list()
    test_y = list()
    for sub_x, sub_y in zip(x, y):
        how_many = int(len(sub_x) * perc)
        indexes = np.arange(0, len(sub_x))
        choices = rng.choice(indexes, how_many, replace=False)
        for sample_x, sample_y, index in zip(sub_x, sub_y, range(len(sub_x))):
            if index in choices:
                test_x.append(sub_x[index])
                test_y.append(sub_y[index])
            else:
                train_x.append(sub_x[index])
                train_y.append(sub_y[index])
    return np.dstack(tuple(train_x)), np.dstack(tuple(test_x)), np.array(train_y), np.array(test_y)


# if __name__ == "__main__":
#     # subject = [1,2]
#     # runs = [3,4,5]
    
#     # # d = load_data(subjects=subject, runs=runs, max_duration=120, chs = ['FP1', 'FP2'])
#     # d = load_data(subjects=subject, runs=runs, max_duration=120)
#     # ch_info = (d[0][0].info['ch_names'])
    
#     # X, y = epochs(d, tmax=1)
#     # y = to_one_hot(y[:,-1])
#     # print(X.shape, y.shape)
#     exclude = [38, 88, 89, 92, 100, 104]
#     subjs = [s for s in range(1,5 + 1) if s not in exclude]
#     #Task 1 and 2, MI vs MM of left and righht fist
#     runs = [3,4, 7,8, 13,14]
#     raws = load_data(subjs, runs, max_duration = 120)
    
#     x,y = epochs(raws)
#     y = to_one_hot(y[:,-1])
#     x = normalize(x)
#     print(x.shape, y.shape)

#     try:
#         x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
#         print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
#     except ValueError as e:
#         print(e)