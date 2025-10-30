### Randomly splits a PKL list and labels.txt into 
import pickle
import random
# import numpy as np

SKEL_PKL_INPUT = "output/NEW_video_skel.pkl"
SKEL_PKL_TRAIN_OUTPUT = "output/final/train_skel.pkl"
SKEL_PKL_TEST_OUTPUT = "output/final/test_skel.pkl"
STABLE_UNSTABLE_LABELS_INPUT = "output/NEW_unstable_labels.txt"
STABLE_UNSTABLE_LABELS_TRAIN_OUTPUT = "output/final/train_unstable_labels.txt"
STABLE_UNSTABLE_LABELS_TEST_OUTPUT = "output/final/test_unstable_labels.txt"
TRAIN_FRACTION = 0.70 # test fraction is (1 - TRAIN_FRACTION)

# Load
with open(SKEL_PKL_INPUT, 'rb') as fr:
    skel = pickle.load(fr)  # a list
labels = [int(label.strip()) for label in open(STABLE_UNSTABLE_LABELS_INPUT, "r")]

# Merge, shuffle, unmerge
assert len(skel) == len(labels), f"Skeleton count and label count must be equal. {len(skel)} != {len(labels)}"
merged = list(zip(skel, labels))
random.shuffle(merged)
skel, labels = zip(*merged) # <- unzips
skel = list(skel)
labels = list(labels)

# Split skeletons and labels
train_idx = int(len(skel) * TRAIN_FRACTION)
skel_train = skel[:train_idx]
skel_test = skel[train_idx:]
train_labels = labels[:train_idx]
test_labels = labels[train_idx:]

# Save
with open(SKEL_PKL_TRAIN_OUTPUT, 'wb') as f:
    pickle.dump(skel_train, f)
with open(SKEL_PKL_TEST_OUTPUT, 'wb') as f:
    pickle.dump(skel_test, f)

with open(STABLE_UNSTABLE_LABELS_TRAIN_OUTPUT, 'w') as f:
    f.writelines(f"{label}\n" for label in train_labels)
with open(STABLE_UNSTABLE_LABELS_TEST_OUTPUT, 'w') as f:
    f.writelines(f"{label}\n" for label in test_labels)
