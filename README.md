# Assessing Unsupervised Machine Learning solutions for Anomaly Detection in Cloud Gaming Sessions

This repository contains the code of the different unsupervised machine learning algorithms implemented in the paper to detect anomalies in Cloud Gaming Sessions.

The models implemented are :
- PCA
- Isolation Forest
- One Class SVM
- Auto-Encoders
- LSTM-VAE

## Disclaimer

The paper is under review at the CNSM Workshop HiPNet'22. The repository is for reviewers only and will be publicly available upon paper acceptance.

## Datasets

The datasets can be download on [this link](https://filesender.renater.fr/?s=download&token=372ec1a6-baaf-4c7e-a183-7944bbd4bfe7).  
Download and unzip the file in the data folder.

```bash
cd data/
tar -xvf data.tar.gz .
```

## Dependencies

The main dependencies are the following:

- Python 3.8+
- Torch
- Numpy
- Pandas
- Scikit-Learn

## Installation

By assuming that conda is installed, you can run this to install the required dependencies.

```bash
conda create --name [ENV_NAME] python=3.8
conda activate [ENV_NAME]
pip install -r requirements.txt
```

## Usage

From the root of the repository, the usage of the main file can be seen by running:

```bash
python3 -m main --help
```

This will output the following parameters to run the main program.

```bash
usage: main.py [-h] --model_name {pca,iforest,oc_svm,ae,lstm_vae} [--window_size WINDOW_SIZE] [--contamination-ratio CONTAMINATION_RATIO]
               [--seed SEED] [--save-dir SAVE_DIR] [--data-dir DATA_DIR] [--if-mixed-data]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {pca,iforest,oc_svm,ae,lstm_vae}
                        The model to train and test.
  --window_size WINDOW_SIZE
                        The window size. Default is 10.
  --contamination-ratio CONTAMINATION_RATIO
                        The contamination ratio. Default is 0.
  --seed SEED           The random generator seed. Default is 42.
  --save-dir SAVE_DIR   The folder to store the model outputs.
  --data-dir DATA_DIR   The folder where the data are stored.
  --if-mixed-data       If datasets must be mixed or not. Default action is true.
```

The hyper-parameters of each model used in the paper are the default parameters in the code. They can be seen or changed in their respective code file (`src\models\[model_name].py`).

## Example

To train an Auto-Encoder model with the default parameters, with the _mixed-datasets_ splitting strategy, with an anomaly contamination of 5%, run :

```bash
python main.py --model_name ae --contamination-ratio 0.05
```
