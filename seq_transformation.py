# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

SET            = "train"
LABELS_INPUT   = 'output/final/train_augmented_labels.txt'
SKEL_PKL_INPUT = 'output/final/train_augmented_skel.pkl'
OUTPUT         = 'output/final/train.npz'

# SET            = "test"
# LABELS_INPUT   = 'output/final/test_unstable_labels.txt'
# SKEL_PKL_INPUT = 'output/final/test_skel.pkl'
# OUTPUT         = 'output/final/test.npz'


def remove_nan_frames(ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('<skename>\t{:^5}\t{}'.format(f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        print(ske_joints.shape)
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def align_frames(skes_joints):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = 91
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 2))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, labels):
    x = skes_joints
    y = one_hot_vector(labels)
    x_shape_zero = (0,) + x.shape[1:]
    y_shape_zero = (0,) + y.shape[1:]

    if SET == "train":
        train_x = x
        train_y = y
        test_x = np.zeros(x_shape_zero)
        test_y = np.zeros(y_shape_zero)
    else:
        train_x = np.zeros(x_shape_zero)
        train_y = np.zeros(y_shape_zero)
        test_x = x
        test_y = y

    np.savez(OUTPUT, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)


if __name__ == '__main__':
    labels = np.loadtxt(LABELS_INPUT, dtype=np.int32) - 1  # action label: 0~1

    with open(SKEL_PKL_INPUT, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)
    skes_joints = align_frames(skes_joints)  # aligned to the same frame length

    print(skes_joints.shape)
    split_dataset(skes_joints, labels)
