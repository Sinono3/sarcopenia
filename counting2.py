import pandas as pd
from pathlib import Path

CLS_STABLE = 0
CLS_UNSTABLE = 1
CLS_SARCOPENIA = 1
CLS_NORMAL = 2

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/stable_unstable")
STABLE_PATH = VIDEO_PATH / "stable"
UNSTABLE_PATH = VIDEO_PATH / "unstable"

samples_paths = list(STABLE_PATH.iterdir()) + list(UNSTABLE_PATH.iterdir())

data = []
for path in samples_paths:
    parts = path.stem.split("_")
    class_stable_unstable = int(parts[0])
    subject = int(parts[1][:-2])
    class_sarcopenia_normal = int(parts[2])

    if class_stable_unstable == CLS_STABLE:
        label_stable_unstable = "stable"
    if class_stable_unstable == CLS_UNSTABLE:
        label_stable_unstable = "unstable"

    if class_sarcopenia_normal == CLS_SARCOPENIA:
        label_sarcopenia_normal = "sarcopenia"
    if class_sarcopenia_normal == CLS_NORMAL:
        label_sarcopenia_normal = "normal"

    data.append((subject, label_stable_unstable, label_sarcopenia_normal))

df = pd.DataFrame(data, columns=["subject", "unstable", "sarcopenia"])

P_unstable = (df["unstable"] == "unstable").mean()
P_sarcopenia = (df["sarcopenia"] == "sarcopenia").mean()
ct = pd.crosstab(df["unstable"], df["sarcopenia"], normalize="index")
P_sarcopenia_given_unstable = ct.loc["unstable", "sarcopenia"]

print(f"{P_unstable=:.3f}")
print(f"{P_sarcopenia=:.3f}")
print(f"{P_sarcopenia_given_unstable=:.3f}")
