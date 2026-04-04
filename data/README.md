# Data Directory

This folder is intended to store EEG datasets used for training and evaluation.

⚠️ Due to size constraints and licensing restrictions, the dataset is not included in this repository.

## Dataset Source

You can download the dataset from:

* PhysioNet Sleep-EDF Database
  https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/#files-panel

## Instructions

1. Download the dataset from the source above
2. Place the files inside this `data/` folder
3. Ensure the folder structure matches the expected format used in the code

## Notes

* Supported formats: `.edf`, `.npz`, `.npy`
* Preprocessing scripts are available in the `src/` directory
