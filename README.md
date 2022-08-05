# Assessing Unsupervised Machine Learning solutions for Anomaly Detection in Cloud Gaming Sessions

This repository contains the code of the different unsupervised machine learning algorithms implemented in the paper to detect anomalies in Cloud Gaming Sessions.

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

```bash
conda create --name [ENV_NAME] python=3.8
conda activate [ENV_NAME]
pip install -r requirements.txt
```

## Usage

From the root of the repository, you can see how to use one of the models:

```bash
python3 -m src.models.[model_name] --help
```

This will output:

```bash
usage: model_name.py [-h] [--window_size WINDOW_SIZE] [--contamination-ratio CONTAMINATION_RATIO] [--seed SEED]
                  [--save-dir SAVE_DIR] [--data-dir DATA_DIR] [--if-mixed-data]

optional arguments:
  -h, --help            show this help message and exit
  --window_size WINDOW_SIZE
                        The window size. Default is 10.
  --contamination-ratio CONTAMINATION_RATIO
                        The contamination ratio. Default is 0.
  --seed SEED           The random generator seed. Default is 42.
  --save-dir SAVE_DIR   The folder to store the model outputs.
  --data-dir DATA_DIR   The folder where the data are stored.
  --if-mixed-data       If datasets must be mixed or not. Default action is true.
```

## Example
