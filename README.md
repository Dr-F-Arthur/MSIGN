## This is the GitHub repository for the project: 

#  MSIGN: A deep learning framework based on multi-scale interaction graph neural networks for predicting binding of synthetic cannabinoids to receptors
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Conda](https://img.shields.io/badge/conda-supported-green.svg)](https://docs.conda.io/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

Zhenyong Cheng [1], Dinghao Liu [1], Yuanpeng Fu [2], Kewei Sheng [3], Yan Xing [4],  Yanling Qia[2,3,4], Shangxuan Cai[5,6], Jubo Wang[4], Peng XU[2,3,4], Bin Di[3,4] and Jun Liao[1,3,7]*

Dataset: https://zenodo.org/doi/10.5281/zenodo.16018514)

## Overview 
Deep learning-based models have been extensively applied to the task of protein-ligand binding affinity (PLA) prediction. Current 3D ligand-complex-based GNNs, though advanced, still struggle with accuracy and generalization due to their overreliance on atomic-level physical features and neglect of chemical space dynamics, leading to data memorization rather than robust learning. To address these issues, we propose a deep learning model based on a Multi-Scale Interaction Graph Neural Network (MSIGN). By constructing ligand functional group graphs and protein amino acid graphs, we introduce chemical information features into the model, which are combined with physical features to enhance binding affinity prediction. Especially, we innovatively adopt a pre-training and fine-tuning training approach in the PLA domain to improve the model's generalization capability on downstream tasks (this study focuses on the binding affinity prediction of synthetic cannabinoids), and we validated the MSIGN model predictions with wet experiments such as SPR on three novel synthetic cannabinoids. Furthermore, we analyze the impact of different fine-tuning strategies on the model's generalization ability. Multiple results collectively demonstrate the superiority of our MSIGN model design, providing a novel approach for future PLA prediction. 

## Hardware Requirements
| Task | GPU Memory | System Memory | Explanation |
|---------|--------|---------|------|
| **Pre-training** | ≥8GB | ≥32GB | Recommended: RTX 3080/3090 or higher |
| **Fine-tuning** | ≥6GB | ≥16GB | An RTX 2080Ti will suffice |
| **Prediction** | ≥4GB | ≥8GB | Can run on lower-end GPUs |

Each epoch in the pre-training phase takes more than one minute.Training and fine-tuning can be performed on a single GPU workstation.

## Software Requirements
The code has been tested on the following systems:
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS

**Python Dependencies** <br />
We recommend using `anaconda3` to set up a Python virtual environment. This software has been tested with core libraries:
```
python
cloudpickle
contourpy
cycler
dgl
dgllife
dill
fonttools
future
hyperopt
importlib-resources
joblib
kiwisolver
matplotlib
multiprocess
networkx
numpy
pandas
pillow
py4j
pymol
pyparsing
python-dateutil
pytz
rdkit
scikit-learn
scipy
seaborn
six
threadpoolctl
torch
tqdm
tzdata
zipp
```
For a more detailed list of required libraries, please refer to environment.yaml.

# RUN MSIGN
## Training & Fine-tuning
You can use the train.py and finetune.py files located in the train directory to perform training and fine-tuning.

## Prediction
You can run `predict_single.py` or modify it and then run it.
```
python predict_single.py
```
Of course, you can also use `toy_test.ipynb` to understand the complete process required to predict affinity using this model. This project includes a toy example.

## notice
Please note, you can create new folders and add the corresponding required files to predict the structure you need. Be sure to pre-dock your ligand and receptor files (i.e., 5xr8.pdb); otherwise, arbitrary ligand files cannot be predicted.
