from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

CLS_STABLE = 0
CLS_UNSTABLE = 1
CLS_SARCOPENIA = 1
CLS_NORMAL = 2

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/stable_unstable")
STABLE_PATH = VIDEO_PATH / "stable"
UNSTABLE_PATH = VIDEO_PATH / "unstable"
samples_paths = list(STABLE_PATH.iterdir()) + list(UNSTABLE_PATH.iterdir())
samples = []

subjects = defaultdict(list)
stable = defaultdict(list)
unstable = defaultdict(list)
sarcopenia = defaultdict(list)
normal = defaultdict(list)

for idx, path in enumerate(samples_paths):
    parts = path.stem.split("_")
    label_stable_unstable = int(parts[0]) # 0 = stable/1 = unstable
    subject = int(parts[1][:-2]) # there's a trailing "tg" for some reason, we trim it off here (thus the :-2)
    label_sarcopenia_normal = int(parts[2]) # sarcopenia/normal

    samples.append(dict(
                       idx=idx,
                       subject=subject,
                       label_stable_unstable=label_stable_unstable,
                       label_sarcopenia_normal=label_sarcopenia_normal,
                   ))

    subjects[idx].append(idx)

    if label_stable_unstable == CLS_STABLE:
        stable[idx].append(idx)
    if label_stable_unstable == CLS_UNSTABLE:
        unstable[idx].append(idx)

    if label_sarcopenia_normal == CLS_SARCOPENIA:
        sarcopenia[idx].append(idx)
    if label_sarcopenia_normal == CLS_NORMAL:
        normal[idx].append(idx)


print(f"{len(samples)=}")
print(f"{len(subjects)=}")
print(f"{len(stable)=}")
print(f"{len(unstable)=}")
print(f"{len(sarcopenia)=}")
print(f"{len(normal)=}")
