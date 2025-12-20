## This is the GitHub repository for the project: 

#  MSIGN: A deep learning framework based on multi-scale interaction graph neural networks for predicting binding of synthetic cannabinoids to receptors
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Conda](https://img.shields.io/badge/conda-supported-green.svg)](https://docs.conda.io/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

Zhenyong Cheng [1], Dinghao Liu [1], Yuanpeng Fu [2], Kewei Sheng [3], Yan Xing [4],  Yanling Qia[2,3,4], Shangxuan Cai[5,6], Jubo Wang[4], Peng XU[2,3,4], Bin Di[3,4] and Jun Liao[1,3,7]*

Dataset: https://zenodo.org/doi/10.5281/zenodo.16018514)

## Overview 
Deep learning-based models have been extensively applied to the task of protein-ligand binding affinity (PLA) prediction. Current 3D ligand-complex-based GNNs, though advanced, still struggle with accuracy and generalization due to their overreliance on atomic-level physical features and neglect of chemical space dynamics, leading to data memorization rather than robust learning. To address these issues, we propose a deep learning model based on a Multi-Scale Interaction Graph Neural Network (MSIGN). By constructing ligand functional group graphs and protein amino acid graphs, we introduce chemical information features into the model, which are combined with physical features to enhance binding affinity prediction. Especially, we innovatively adopt a pre-training and fine-tuning training approach in the PLA domain to improve the model's generalization capability on downstream tasks (this study focuses on the binding affinity prediction of synthetic cannabinoids), and we validated the MSIGN model predictions with wet experiments such as SPR on three novel synthetic cannabinoids. Furthermore, we analyze the impact of different fine-tuning strategies on the model's generalization ability. Multiple results collectively demonstrate the superiority of our MSIGN model design, providing a novel approach for future PLA prediction. 

## Software Requirements
The code has been tested on the following systems:
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS

**Python Dependencies** <br />
We recommend using `anaconda3` to set up a Python virtual environment. This software has been tested with:
```
python
cloudpickle
contourpy
cycler
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
numpy
pandas
pillow
py4j
pyparsing
python-dateutil
pytz
rdkit
scikit-learn
seaborn
six
threadpoolctl
tqdm
tzdata
zipp
```

# RUN MSIGN
```
python predict_single.py
```
## notice
Please note, you can create new folders and add the corresponding required files to predict the structure you need. Be sure to pre-dock your ligand and receptor files (i.e., 5xr8.pdb); otherwise, arbitrary ligand files cannot be predicted.
