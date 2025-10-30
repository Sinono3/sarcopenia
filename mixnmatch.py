import torch
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation
import pickle
import einops

PICKLE_PATH = "output/final/train_skel.pkl"
UNSTABLE_LABELS_PATH = "output/final/train_unstable_labels.txt"
SKEL_PICKLE_OUT_PATH = "output/final/train_augmented_skel.pkl"
UNSTABLE_LABELS_OUTPUT_PATH = "output/final/train_augmented_labels.txt"

SKELETON_CONNECTIONS = [
    # Torso
    [0, 1],
    [1, 20],
    [20, 2],
    [2, 3],
    # Left Arm
    [2, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 21],
    [7, 22],
    # Right Arm
    [2, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 23],
    [11, 24],
    # Left Leg
    [0, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    # Right Leg
    [0, 16],
    [16, 17],
    [17, 18],
    [18, 19],
]
SKELETON_CONNECTIONS = np.array(SKELETON_CONNECTIONS)


with open(PICKLE_PATH, 'rb') as f:
    skels = pickle.load(f)

labels = [int(label.strip()) for label in open(UNSTABLE_LABELS_PATH, "r")]
print(f"before: {len(skels)}")

for i in range(len(skels)):
    skels[i] = einops.rearrange(skels[i], 'frame (joint coord) -> frame joint coord', coord=3)
    skels[i] = torch.tensor(skels[i])


## Mixnmatch
# 1 is lower, 0 is upper
mix = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
MIX = torch.tensor(mix, dtype=torch.bool)

def mixnmatch(skels):
    mixed = []

    for i in range(len(skels)):
        for j in range(len(skels)):
            skel_length = min(skels[i].shape[0], skels[j].shape[0])
            mix = einops.repeat(MIX, "j -> t j 3", t=skel_length)
            skel_i = skels[i][:skel_length, :, :] # crop to shortest
            skel_j = skels[j][:skel_length, :, :]
            # pick from upper body from skel_i, lower body from skel_j
            new_skel = np.where(mix, skel_i, skel_j)
            mixed.append(new_skel)

    return mixed

skels_stable = []
skels_unstable = []

for i, label in enumerate(labels):
    if label == 0:
        skels_stable.append(skels[i])
    elif label == 1:
        skels_unstable.append(skels[i])
    else:
        raise NotImplementedError()
        

print(f"Stable. Before: {len(skels_stable)}")
print(f"Unstable. Before: {len(skels_unstable)}")

skels_stable = mixnmatch(skels_stable)
skels_unstable = mixnmatch(skels_unstable)

print(f"Stable. After: {len(skels_stable)}")
print(f"Unstable. After: {len(skels_unstable)}")

# OUTPUT final mix and match, pick N random
# IMPORTANT: we select the minimum, to cut off the excess skels (and fix a distributional imbalance)
minimum_len = min(len(skels_stable), len(skels_unstable))
skels_stable = random.choices(skels_stable, k=minimum_len)
skels_unstable = random.choices(skels_unstable, k=minimum_len)
print(f"Stable (after random picks). After: {len(skels_stable)}")
print(f"Unstable (after random picks). After: {len(skels_unstable)}")

## concat and save to file
skels = skels_stable + skels_unstable
new_unstable_labels = [0] * len(skels_stable) + [1] * len(skels_unstable)

for i in range(len(skels)):
    skels[i] = einops.rearrange(skels[i], 'frame joint coord -> frame (joint coord)')

with open(SKEL_PICKLE_OUT_PATH, 'wb') as f:
    pickle.dump(skels, f)
with open(UNSTABLE_LABELS_OUTPUT_PATH, 'w') as f:
    f.writelines(f"{label}\n" for label in new_unstable_labels)

for i in range(len(skels)):
    skels[i] = einops.rearrange(skels[i], 'frame (joint coord) -> frame joint coord', coord=3)


















## Visualization ----------------------------------------------------------
# DEBUG: Only visualize first 100
skels = skels[:100]
for i in range(len(skels)):
    # # Feet on y=0 (every frame)
    # ground = (skels[i][:, 14] + skels[i][:, 18]) / 2
    # ground = einops.reduce(ground, "frame coord -> coord", reduction="mean")
    # skels[i] -= ground

    # # Calculate new average direction
    # # vec_up = (skels[i][:, 3] - (skels[i][:, 17] + skels[i][:, 13]) / 2)
    # vec_right = skels[i][-8:, 8] - skels[i][-8:, 4]
    # vec_up = skels[i][:, 2]

    # vec_right = vec_right.sum(axis=0)
    # vec_up = vec_up.sum(axis=0)

    # vec_fwd = np.cross(vec_right, vec_up)
    # rotation = np.column_stack([
    #     vec_right / np.linalg.norm(vec_right),
    #     vec_up / np.linalg.norm(vec_up),
    #     vec_fwd / np.linalg.norm(vec_fwd),
    # ])
    # # Orthogonalize
    # # rotation = np.linalg.qr(rotation)[0]

    # # Matrix multiplication
    # print(len(skels))
    # print(skels[0].shape)
    # skels[i] = einops.einsum(rotation, skels[i], "coord_i coord_j, frame joint coord_j -> frame joint coord_i")

    # Y <-> Z
    skels[i] = skels[i][:, :, [0, 2, 1]]
    skels[i][:, :, 0] *= -1.0
    skels[i][:, :, 2] *= -1.0

anim_idx = 0
base_idx = 0
skels_lines = [skel[:, SKELETON_CONNECTIONS, :] for skel in skels]
fig = plt.figure()
ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
points: Path3DCollection = ax.scatter(*skels[anim_idx][0].T)  # Points

lines = Line3DCollection(skels_lines[anim_idx][0])
ax.add_collection3d(lines)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def animate(i):
    global anim_idx, base_idx, cap, fps, frame_count
    
    frame = i - base_idx
    if frame >= skels[anim_idx].shape[0]:
        base_idx = i
        anim_idx += 1
        frame = 0
        print(f"Next animation: {anim_idx}")

    skel = skels[anim_idx]
    skel_lines = skels_lines[anim_idx]
    lines.set_segments(skel_lines[frame])
    points._offsets3d = (skel[frame, :, 0], skel[frame, :, 1], skel[frame, :, 2])

ani = FuncAnimation(fig, animate, interval=30)
ani.resume()
plt.show()

