# Chest X-Ray Pneumonia Classifier

This project develops and evaluates convolutional neural network models to classify chest X-ray images 
as either **NORMAL** or **PNEUMONIA**. It includes data downloading and re-splitting, exploratory analysis,
a baseline CNN, three model improvements (data augmentation, batch normalization + dropout, early stopping),
and a final evaluation on a test set.

## Dependencies

- Python 3.9+
- PyTorch
- torchvision
- pandas, numpy, matplotlib
- Kaggle API (`pip install kaggle`)
- Weights & Biases (`pip install wandb`)

## Data

We use the Kaggle dataset **paultimothymooney/chest-xray-pneumonia**, which is downloaded via the Kaggle API
and then re-split into:

- `train/` (80%)
- `val/`   (10%)
- `test/`  (10%)

Each folder contains two subfolders: `NORMAL/` and `PNEUMONIA/`.

## Usage

1. Place your `kaggle.json` API token under `~/.config/kaggle/kaggle.json`.
2. Run the notebook `COMP0188_Answers.ipynb` to download, preprocess, train, and evaluate.
3. Adjust hyperparameters or transforms as needed.


