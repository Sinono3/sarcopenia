### downloads 

- `git clone https://github.com/Sinono3/sarco`
- `cd sarco`
- Download [models.zip](https://drive.google.com/file/d/1YvS50TVNRfQ9q5b7hs75Nl05SsF6H5cN/view?usp=sharing) and extract to a folder `models`.
  - directory structure should be like
    - sarco
      - models
        - hrnet
        - mediapipe
        - poseformer
        - yolov3
      - ...

### pose estimation

- You will first need to install dependencies.. Just test the scripts until they run. you will have to
  figure out on your own... sorry. Shouldn't be too difficult though.
- Edit initial constants of `mp4_to_skeleton.py` to reflthe ect current environment.
  - Modify `VIDEO_PATH` to point to dataset folder.
    - This folder on this path should contain two sub-folders: `stable` and `unstable`.
    - `stable` and `unstable`
    - The naming should be consistent with e.g. `1_137tg_2_33.0_to_36.0.mp4`
  - Executing this script will produce four outputs:
    - `SKELETON_OUTPUT_PATH`: the actual skeleton data, a pickle file of a list of each clip as a skeleton sequence stored as a `np.array` of shape (90, 25, 4).
    - `UNSTABLE_LABELS_OUTPUT_PATH`: a .txt file where the ith line has a 0/1 for stable/unstable for the ith skeleton in the Pickle file
    - `SARCOPENIA_LABELS_OUTPUT_PATH`: a .txt file where the ith line has a 0/1 for sarcopenia/healthy for the ith skeleton in the Pickle file
    - `PATHS_OUTPUT_PATH`: a .txt file where the ith line has the video path from which the ith skeleton in the Pickle file was estimated
  - We convert the mediapipe's 17-joint format to 25 to reflect NTU RGB+D's format (which SkateFormer is trained on.)
- Run `python mp4_to_skeleton.py`
- Make sure the paths in `seq_transformation.py` are okay.
- Run `python seq_transformation.py`
  - This script saves the final NPZ file that SkateFormer is trained on.
  - This script creates a 70/30 train/test split. (in this case, the test is more like validation)

### visualization

- Edit `visualize_skeleton.py` and make sure `PICKLE_PATH`, `PATHS_PATH` and `VIDEO_PATH` are okay.
- Run `python visualize_skeleton.py`

### skateformer

- `git clone https://github.com/Sinono3/skateformer`
- `cd skateformer`
- `uv venv`
- `source .venv/bin/activate`
- Install dependencies (this may fail on your machine, you will have to figure out on your own... sorry.). You can look at the skateformer `requirements.yaml` for reference.
  - `pip install "cython<3.0.0" setuptools wheel`
  - `pip install "pyyaml==5.4.1" --no-build-isolation`
  - `uv sync`
- Download SkateFormer pretrained models as instructed in SkateFormer's README.
- Then, edit `newconfig/stableunstable_train.yaml`.
  - Modify line 14 (`train_feeder_args.data_path`) to reflect `SAVE_PATH` (from `seq_transformation.py`).
  - Modify line 27 (`test_feeder_args.data_path`) to reflect `SAVE_PATH` (from `seq_transformation.py`).
- Then, to finetune (and get the validation accuracy after each epoch), run:
  - `python main.py --config newconfig/stableunstable_train.yaml`
