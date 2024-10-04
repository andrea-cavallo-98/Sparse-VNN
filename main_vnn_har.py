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

args = parse_args()

dset = args.dset

dimNodeSignals = args.dimNodeSignals
L = len(dimNodeSignals) - 1
nFilterTaps = [args.filter_taps] * L
dimLayersMLP = args.dimLayersMLP 
lr = args.lr
cov_type = args.cov_type
tau = args.tau
split = args.split # Predefined dataset split

nEpochs = args.nEpochs
df_res = pd.DataFrame(columns=['it', 'Shape', 'Time', 'Test_perf_acc'])

if dset == "mhealth":
    x_train, y_train, x_val, y_val, x_test, y_test = load_mhealth()
elif dset == "realdisp":
    x_train, y_train, x_val, y_val, x_test, y_test = load_realdisp()
else:
    print("Dataset not available!")

dimLayersMLP[-1] = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
print("Number of classes: ", dimLayersMLP[-1])
dimNodeSignals[0] = x_train.shape[1]
m = x_train.shape[2]
C_dense = torch.cov(x_train.reshape(-1,m).T)

nTrain = x_train.shape[0]
nTest = x_test.shape[0]
print("Training samples: ", nTrain, " Valid samples: ", x_val.shape[0], " Test samples: ", nTest)

for it in range(args.iterations):

    C = sparsify_covariance(C_dense, cov_type, 
                        thr=tau * torch.tensor(np.sqrt(np.log(m) / nTrain)), 
                        p=args.p, sparse_tensor=args.sparse_tensor)

    GNN = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.LeakyReLU, [m]*len(nFilterTaps), 
                                gml.NoPool, [1]*len(nFilterTaps), dimLayersMLP, C, average=True)

    Loss = nn.NLLLoss()
    logSoftmax = nn.LogSoftmax(dim=1)

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
            lossValueTrain = Loss(logSoftmax(yHatTrainBatch).squeeze(), yTrainBatch)
            accTrain = compute_multiclass_accuracy(logSoftmax(yHatTrainBatch).squeeze(), yTrainBatch)
            lossValueTrain.backward()
            optimizer.step()
            tot_train_loss.append(lossValueTrain.detach())

        with torch.no_grad():
            yHatVal = GNN(x_val) if len(x_val.shape) == 3 else GNN(x_val.unsqueeze(1))
            lossValueVal = Loss(logSoftmax(yHatVal).squeeze(), y_val)
            accVal = compute_multiclass_accuracy(logSoftmax(yHatVal).squeeze(), y_val)

            if accVal > Best_Valid_acc:
                Best_GNN = deepcopy(GNN)
                Best_Valid_acc = accVal
                
        print(f"Epoch {epoch} Train loss: {sum(tot_train_loss)} Train acc: {accTrain} Val loss: {lossValueVal.detach().item()} Val acc: {accVal} time: {tot_time}")
        all_times.append(tot_time)


    GNN = deepcopy(Best_GNN)
    yBestTest = GNN(x_test) if len(x_train.shape) == 3 else GNN(x_test.unsqueeze(1))
    lossTest = Loss(logSoftmax(yBestTest).squeeze(), y_test)
    acc = compute_multiclass_accuracy(logSoftmax(yBestTest).squeeze(), y_test).item()

    print("Test accuracy: ", acc, " Test loss: ", lossTest.detach().item())
    df_new_row = pd.DataFrame(data=np.array([[it, dimNodeSignals[-1], np.array(all_times).mean(), acc]]), 
                                    columns=['it', 'Shape', 'Time', 'Test_perf_acc'])
    df_res = pd.concat([df_res,df_new_row], ignore_index=True)

df_res.to_csv(f"out/{dset}_{cov_type}_{args.p}_{L}_vnn_res_har.csv")

