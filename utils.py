import argparse
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import zscore
import sys
sys.path.append('../')


def parse_boolean(value):
    """Parse boolean values passed as argument"""
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False

def parse_hidden_sizes(value):
    """Create list of int from string of csv"""
    
    return list(map(lambda x: int(x), value.split(",")))


def parse_args():
    """ Parse arguments """
    parse = argparse.ArgumentParser()

    ## Run details
    parse.add_argument("--m", help="number of time samples", type=int, default=100)
    parse.add_argument("--pred_step", help="prediction step", type=int, default=1)
    parse.add_argument("--T", help="history length", type=int, default=5)
    parse.add_argument("--update_covariance", help="whether to perform covariance update", type=parse_boolean, default=False)
    parse.add_argument("--gamma", help="covariance update coefficient", type=float, default=0.1)
    parse.add_argument("--tau", help="threshold coefficient tau", type=float, default=1.)
    parse.add_argument("--dimNodeSignals", help="sizes of GNN hidden layers", type=parse_hidden_sizes, default=[1,13,13])
    parse.add_argument("--dimLayersMLP", help="sizes of MLP hidden layers", type=parse_hidden_sizes, default=[1,1])
    parse.add_argument("--filter_taps", help="filter taps", type=int, default=2)
    parse.add_argument("--iterations", help="repetitions of the same experiments", type=int, default=1)
    parse.add_argument("--n_it", help="number of iterations", type=int, default=10)
    parse.add_argument("--online", help="whether to update the model online", type=parse_boolean, default=True)
    parse.add_argument("--sparse_tensor", help="whether to use torch sparse tensors or not", type=parse_boolean, default=False)
    parse.add_argument("--nEpochs", help="epochs", type=int, default=100)    
    parse.add_argument("--perms", help="number of permutations", type=int, default=10)
    parse.add_argument("--batchSize", help="batch size", type=int, default=1000)
    parse.add_argument("--split", help="dataset split", type=int, default=1)
    parse.add_argument("--dimOutputSignals", help="dim recurrent output", type=int, default=8)
    parse.add_argument("--dimHiddenSignals", help="dim recurrent hidden state", type=int, default=8)
    parse.add_argument("--out_file", help="output file", type=str, default=None)
    parse.add_argument("--dset", help="dataset", type=str, default="dense")
    parse.add_argument("--optimizer", help="optimizer", type=str, default="SGD")
    parse.add_argument("--lr", help="learning rate", type=float, default=0.015)
    parse.add_argument("--p", help="mean of edge drop probability", type=float, default=0.5)
    parse.add_argument("--lr_test", help="learning rate for online test", type=float, default=0.001)
    parse.add_argument("--h_size", help="MLP size for TPCA", type=int, default=256)
    parse.add_argument("--suffix", help="suffix for some experiments", type=str, default="")
    parse.add_argument("--cov_type", help="type of covariance estimator", type=str, default="standard")

    args = parse.parse_args()
    return args



def create_xy_realdisp(sub_id_list):
    T = 128
    X, y = [], []


    for sub_id in sub_id_list:
        print("Subject ", sub_id)
        df = pd.read_csv(f"Data/realdisp+activity+recognition+dataset/subject{sub_id}_ideal.log", delim_whitespace=True, header=None)
        labels_to_keep = [10, 11, 29, 31,  9, 32, 33,  3,  2,  1]
        
        df = df[df.iloc[:,-1].isin(labels_to_keep)]

        all_X, all_y = [], []
        labels = df.iloc[:,-1].unique()
        for lab in labels:
            cur_df = df[df.iloc[:,-1] == lab].to_numpy()[:,2:-1]
            if cur_df.shape[0] < T:
                continue
            X_win, y_win = [], []
            for i in range(0, cur_df.shape[0], 64):
                if i + T <= cur_df.shape[0]:
                    window = cur_df[i: i + T]
                    X_win.append(window)
                    y_win.append(labels_to_keep.index(lab))
                else:
                    break
            all_X.append(np.stack(X_win))
            all_y.append(np.array(y_win))
        X.append(np.concatenate(all_X, axis=0))
        y.append(np.concatenate(all_y, axis=0))

    X = np.concatenate(X, axis=0)
    y = torch.LongTensor(np.concatenate(y))
    s, n = X.shape[1], X.shape[2]
    X = torch.FloatTensor(zscore(X.reshape((-1,n)), axis=0).reshape((-1,s,n)))
    return X, y

def preprocess_realdisp():
    X_train, y_train = create_xy_realdisp([2,3,5,10,13,15,16,17])
    X_val, y_val = create_xy_realdisp([4,6,10,11])
    X_test, y_test = create_xy_realdisp([1,7,8,9,12,14])
    torch.save((X_train, X_val, X_test, y_train, y_val, y_test), "Data/realdisp_preprocess")


def load_realdisp():
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = torch.load("Data/realdisp_preprocess")
    except:
        X_train, y_train = create_xy_realdisp([2,3,5,10,13,15,16,17])
        X_val, y_val = create_xy_realdisp([4,6,10,11])
        X_test, y_test = create_xy_realdisp([1,7,8,9,12,14])
        torch.save((X_train, X_val, X_test, y_train, y_val, y_test), "Data/realdisp_preprocess")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_xy_mhealth(sub_id_list):
    T = 128
    X, y = [], []
    col_to_keep = [0,1,2,5,6,7,8,9,10,14,15,16,17,18,19]

    for sub_id in sub_id_list:
        df = pd.read_csv(f"Data/MHEALTHDATASET/mHealth_subject{sub_id}.log", sep='\t', header=None)
        df = df[df.iloc[:,-1] != 0]
        s = df.to_numpy()[:,col_to_keep]
        
        all_X, all_y = [], []
        labels = df.iloc[:,-1].unique()
        for lab in labels:
            cur_df = df[df.iloc[:,-1] == lab].to_numpy()[:,::-1]
            X_win, y_win = [], []
            for i in range(0, cur_df.shape[0], T//2):
                if i + T <= cur_df.shape[0]:
                    window = cur_df[i: i + T]
                    X_win.append(window)
                    y_win.append(lab)
                else:
                    break
            all_X.append(np.stack(X_win))
            all_y.append(np.array(y_win))
        X.append(np.concatenate(all_X, axis=0))
        y.append(np.concatenate(all_y, axis=0))

    X = np.concatenate(X, axis=0)
    y = torch.LongTensor(np.concatenate(y)-1)
    s, n = X.shape[1], X.shape[2]
    X = torch.FloatTensor(zscore(X.reshape((-1,n)), axis=0).reshape((-1,s,n)))
    return X, y

def load_mhealth():
    X_train, y_train = create_xy_mhealth([1,3,4,5,7,8])
    X_val, y_val = create_xy_mhealth([6,10])
    X_test, y_test = create_xy_mhealth([2,9])
    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_accuracy(output, target):
    output_labels = torch.where(output > 0, 1., 0.)
    correct = output_labels.eq(target).double()
    # print(f"Correct 1: {(output_labels[target == 1] == 1).sum()} Correct 0: {(output_labels[target == 0] == 0).sum()}")
    return (correct.sum() / target.shape[0]).item()

def compute_multiclass_accuracy(output, target):
    preds = output.argmax(1).type_as(target)
    correct = preds.eq(target).double()
    correct = correct.sum()
    return correct / len(target)

def load_epilepsy():
    x = np.load("Data/epilepsy/x_epilepsy.npy")
    y = torch.FloatTensor(np.load("Data/epilepsy/y_epilepsy.npy"))
    if len(x.shape) == 3:
        x = torch.FloatTensor(zscore(x, axis=0))
    else:
        x = torch.FloatTensor(zscore(np.load("Data/epilepsy/x_epilepsy.npy"), axis=0))
    return x,y

def load_cni():
    x_train = np.load("Data/CNI/x_train.npy")
    x_test = np.load("Data/CNI/x_test.npy")
    s, n = x_train.shape[1], x_train.shape[2]
    x_train = torch.FloatTensor(zscore(x_train.reshape((-1,n)), axis=0).reshape((-1,s,n)))
    s, n = x_test.shape[1], x_test.shape[2]
    x_test = torch.FloatTensor(zscore(x_test.reshape((-1,n)), axis=0).reshape((-1,s,n)))
    y_train = torch.FloatTensor(np.load("Data/CNI/y_train.npy"))
    y_test = torch.FloatTensor(np.load("Data/CNI/y_test.npy"))
    return x_train,x_test,y_train,y_test

def load_data(dset):
    if dset == "SmallCov": 
        df = pd.read_csv('Data/SmallCov.csv')
    elif dset == "LargeCov":
        df = pd.read_csv('Data/LargeCov.csv', index_col=0)
    elif dset == "SparseCov":
        df = pd.read_csv('Data/SparseCov.csv', index_col=0)

    x_input = df.iloc[:,:-1].to_numpy()
    y_output = df.iloc[:,-1].to_numpy()
    return x_input, y_output


def sparsify_covariance(C, cov_type, thr=0.0, p=0.1, sparse_tensor=False):
    if cov_type == "standard":
        C_sparse = C
    elif cov_type == "RCV": 
        # Generate probability values
        sigma = min((1-p)/3, p/3)
        lim_prob = np.linspace(0,1,1000)
        distr_prob = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((lim_prob-p)/sigma)**2)
        distr_prob = distr_prob / distr_prob.sum()
        prob_values = np.random.choice(lim_prob, p=distr_prob, size=C.shape[0] ** 2)
        prob_values = torch.FloatTensor(np.sort(prob_values))

        # Assign probability values 
        sorted_idx = torch.argsort(C.abs().flatten())
        prob = torch.zeros([C.shape[0] ** 2,]).float().scatter_(0, sorted_idx, prob_values)
        prob = prob.reshape(C.shape)
        prob[torch.eye(prob.shape[0]).long()] = 1 # no removal on the diagonal
        
        # Drop edges symmetrically
        mask = torch.rand(C.shape) <= prob
        triu = torch.triu(torch.ones(C.shape), diagonal=0).bool()
        mask = mask * triu + mask.t() * ~triu # make resulting matrix symmetric
        C_sparse = torch.where(mask, C, 0)

    elif cov_type == "ACV":
        prob = C.abs() / C.abs().max()
        prob[torch.eye(prob.shape[0]).long()] = 1 # no removal on the diagonal
        mask = torch.rand(C.shape) <= prob
        triu = torch.triu(torch.ones(C.shape), diagonal=0).bool()
        mask = mask * triu + mask.t() * ~triu # make resulting matrix symmetric
        C_sparse = torch.where(mask, C, 0)

    elif cov_type == "hard_thr":
        C_sparse = torch.where(C.abs() > thr, C, 0)
    elif cov_type == "soft_thr":
        C_sparse = torch.where(C.abs() > thr, C - (C>0).float()*thr, 0)

    if sparse_tensor:
        return C_sparse.to_sparse()
    
    return C_sparse

def compute_covariance(X, cov_type, thr=0.0, p=0.0):
    C = torch.cov(X)
    C_sparse = sparsify_covariance(C, cov_type, thr, p)

    return C_sparse

