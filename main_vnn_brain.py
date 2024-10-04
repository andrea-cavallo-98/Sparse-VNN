import sys 
sys.path.append('../')
import pandas as pd
import numpy as np
import torch 
from torch import nn, optim
from copy import deepcopy
import Modules.architectures as archit
import Utils.graphML as gml
from utils import *
import time
from sklearn.model_selection import train_test_split

args = parse_args()

dset = args.dset

dimNodeSignals = args.dimNodeSignals
L = len(dimNodeSignals) - 1
nFilterTaps = [args.filter_taps] * L
dimLayersMLP = args.dimLayersMLP 
lr = args.lr
cov_type = args.cov_type
tau = args.tau

nEpochs = args.nEpochs
df_res = pd.DataFrame(columns=['it', 'Shape', 'Time', 'Test_perf_acc'])


if dset == "epilepsy":
    x, y = load_epilepsy()    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)
    dimNodeSignals[0] = 1
elif dset == "cni":
    x_train,x_test,y_train,y_test = load_cni()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    dimNodeSignals[0] = 122
else:
    print("Dataset not supported!")
if len(x_train.shape) == 3:
    m = x_train.shape[2]
    C_dense = torch.cov(x_train.reshape(-1,m).T)
else:
    m = x_train.shape[1]
    C_dense = torch.cov(x_train.T)

dimLayersMLP[-1] = 1 # binary classification
nTrain = x_train.shape[0]
nTest = x_test.shape[0]

for it in range(args.iterations):
    C = sparsify_covariance(C_dense, cov_type, 
                        thr=tau * torch.tensor(np.sqrt(np.log(m) / nTrain)), 
                        p=args.p, sparse_tensor=args.sparse_tensor)

    GNN = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.LeakyReLU, [m]*len(nFilterTaps), 
                                gml.NoPool, [1]*len(nFilterTaps), dimLayersMLP, C, average=True)

    Loss = nn.BCEWithLogitsLoss()
    batchSize = args.batchSize
    nTrainBatches = int(np.ceil(nTrain / batchSize))

    optimizer = optim.Adam(GNN.parameters(), lr=lr, weight_decay=0.001)
    Best_Valid_Loss, Best_Valid_acc = 1e10, 0

    all_times = []
    for epoch in range(nEpochs):
        tot_train_loss = []
        tot_val_mae = 0.
        tot_val_mape = 0.
        
        tot_time = 0
        train_perm_idx = torch.randperm(nTrain)
        for batch in range(nTrainBatches):
            thisBatchIndices = torch.LongTensor(train_perm_idx[batch * batchSize : (batch + 1) * batchSize])
            xTrainBatch = x_train[thisBatchIndices] if len(x_train.shape) == 3 else x_train[thisBatchIndices].unsqueeze(1)
            yTrainBatch = y_train[thisBatchIndices]
            GNN.zero_grad()
            t0 = time.time()
            yHatTrainBatch = GNN(xTrainBatch)
            t1 = time.time()         
            tot_time += t1-t0
            lossValueTrain = Loss(yHatTrainBatch.squeeze(), yTrainBatch)
            accTrain = compute_accuracy(yHatTrainBatch.squeeze(), yTrainBatch)
            lossValueTrain.backward()
            optimizer.step()
            tot_train_loss.append(lossValueTrain.detach())

        with torch.no_grad():
            yHatVal = GNN(x_val) if len(x_val.shape) == 3 else GNN(x_val.unsqueeze(1))
            lossValueVal = Loss(yHatVal.squeeze(), y_val)
            accVal = compute_accuracy(yHatVal.squeeze(), y_val)

            if accVal > Best_Valid_acc:
                Best_GNN = deepcopy(GNN)
                Best_Valid_acc = accVal
                
        print(f"Epoch {epoch} Train loss: {sum(tot_train_loss)} Train acc: {accTrain} Val loss: {lossValueVal.detach().item()} Val acc: {accVal} time: {tot_time}")
        all_times.append(tot_time)


    GNN = deepcopy(Best_GNN)
    yBestTest = GNN(x_test) if len(x_train.shape) == 3 else GNN(x_test.unsqueeze(1))
    lossTest = Loss(yBestTest.squeeze(), y_test)
    acc = compute_accuracy(yBestTest.squeeze(), y_test)

    print("Test accuracy: ", acc, " Test loss: ", lossTest.detach().item())
    df_new_row = pd.DataFrame(data=np.array([[it, dimNodeSignals[-1], np.array(all_times).mean(), acc]]), 
                                    columns=['it', 'Shape', 'Time', 'Test_perf_acc'])
    df_res = pd.concat([df_res,df_new_row], ignore_index=True)

df_res.to_csv(f"out/{dset}_{cov_type}_{args.p}_{L}_vnn_res_brain.csv")

