# Sparse-VNN

This repository contains the code for the paper "Sparse Covariance Neural Networks" ([preprint](https://arxiv.org/abs/2410.01669)). The code for VNN is taken from https://github.com/sihags/VNN.

## Description

Sparse coVariance Neural Networks (S-VNNs) apply sparsification techniques to the sample covariance matrix of the data before performing convolutions. This results in stability improvements, especially in high-dimensional low-data settings, increased performance due to a reduced impact of spurious correlations and improved time and memory efficiency.   

## Usage

### Requirements

- Python 3.11.5
- `pip install -r requirements.txt`

### Datasets

- The synthetic datasets `SparseCov`, `LargeCov` and `SmallCov` are available in the `Data` folder.
- `epilepsy` can be downloaded at https://math.bu.edu/people/kolaczyk/datasets.html. We provide the preprocessed version in the `Data/epilepsy` folder.
- `CNI` can be downloaded at http://www.brainconnectivity.net/challenge.html. We provide the preprocessed version in the `Data/CNI` folder. 
- `MHEALTH` can be downloaded at https://archive.ics.uci.edu/dataset/319/mhealth+dataset. The downloaded files should be placed in a folder called `Data/MHEALTHDATASET`. The dataset is automatically processed every time it is loaded.
- `Realdisp` can be downloaded at https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset and preprocessed with the function `preprocess_realdisp()` in `utils.py` (this is automatically done the first time an experiment is run on the dataset). The downloaded files should be placed in a folder called `Data/realdisp+activity+recognition+dataset`.


### Run experiments on real data

The following is an example of training and evaluation of S-VNN on a brain dataset.

```
python main_vnn_brain.py --dimNodeSignals 1,32,32 --dimLayersMLP 32,16,1 --sparse_tensor true --cov_type RCV --p 0.25 --dset epilepsy --nEpochs 50
```

The following, instead, is an example of training and evaluation of S-VNN on a human action recognition dataset.

```
python main_vnn_har.py --dimNodeSignals 1,32,32 --dimLayersMLP 32,16,1 --sparse_tensor true --cov_type ACV --dset mhealth --nEpochs 50
```


Parameters:
- `dimNodeSignals`: the size of the input node signal followed by the size of each layer of VNN (as a csv string)
- `dimLayersMLP`: the size of the MLP layers for the final task followed by the output size for each node (as a csv string)
- `sparse_tensor`: if `True`, use torch.sparse to represent the covariance matrix, otherwise, store it as a dense matrix
- `cov_type`: type of covariance sparsification used for the experiment. Supports `standard` (no sparsification), `RCV` (with additional parameter `p`), `ACV`, `hard_thr` and `soft_thr`.
- `dset`: dataset. Supports `SparseCov`, `LargeCov`, `SmallCov` (synthetic), `epilepsy`, `cni` (brain datasets), `mhealth`, `realdisp` (Human Action Recognition)
- `tau`: parameter for hard and soft thresholding
- `lr`: learning rate
- `iterations`: how many times to repeat an experiment
- `batchSize`: batch size
- `nEpochs`: number of epochs


The hyperparameters used in the experiments for each dataset are reported in the paper and the file `example_run.sh` contains the commands to replicate the experiments on the real datasets with the correct parameters for ACV. The results are saved in csv files in the `out` folder and printed on the terminal.


### Run stability experiments

The following is an example of the stability experiment with hard thresholding on the synthetic dataset `SparseCov`.

```
python main_vnn_sparse.py --dimNodeSignals 1,32,32 --dimLayersMLP 32,16,1 --cov_type hard_thr --tau 0.1 --nEpochs 50
```

The following, instead, is an example of the stability experiment with stochastic sparsification on the synthetic dataset `SmallCov`. Supports also `LargeCov`.

```
python main_vnn_prob.py --dimNodeSignals 1,32,32 --dimLayersMLP 32,16,1 --cov_type ACV --nEpochs 50 --dset SmallCov
```

The training arguments are the same as for the experiments on real data.


### Repository structure

- `Data`: folder containing the datasets
- `Modules`: folder containing model definitions 
- `out`: folder for saving results
- `Utils`: folder with various utility functions
- `main_vnn_sparse.py`: run stability experiments on synthetic data with hard and soft thresholding. Supports `hard_thr` and `soft_thr` for sparsification and synthetic datasets.
- `main_vnn_prob.py`: run stability experiments on synthetic data with stochastic sparsification (both ACV and RCV). Supports `SmallCov` and `LargeCov` as datasets.
- `main_vnn_har.py`: run experiments on Human Action Recognition datasets. Supports `mhealth` and `realdisp`.
- `main_vnn_brain.py`: run experiments on brain datasets. Supports `epilepsy` and `cni`.
- `utils.py`: additional utility functions


## Citation

```
@misc{cavallo2024sparsevnn,
      title={Sparse Covariance Neural Networks}, 
      author={Andrea Cavallo and Zhan Gao and Elvin Isufi},
      year={2024},
      eprint={2410.01669},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01669}, 
}
```