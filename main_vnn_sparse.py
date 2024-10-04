import sys 
sys.path.append('../')
import pandas as pd
import numpy as np
import torch 
from torch import nn, optim
from copy import deepcopy
from tqdm import tqdm
import Modules.architectures as archit
import Utils.graphML as gml
from scipy.stats import zscore
from utils import *

args = parse_args()

dset = "SparseCov"

dimNodeSignals = args.dimNodeSignals
L = len(dimNodeSignals) - 1
nFilterTaps = [args.filter_taps] * L
dimLayersMLP = args.dimLayersMLP 
lr = args.lr
cov_type = args.cov_type
tau = args.tau

x_input, y_output = load_data(dset)

nEpochs = args.nEpochs

train_perc, valid_perc, test_perc = 0.8, 0.1, 0.1
nTotal = y_output.shape[0]
m = x_input.shape[1]

nTrain = np.floor(train_perc*nTotal).astype(int)
idxTotal = np.random.permutation(nTotal)
idxTest = idxTotal[np.floor((train_perc + valid_perc)*nTotal).astype(int):]
idxTrain = idxTotal[0:nTrain]
idxValid = idxTotal[np.floor(train_perc*nTotal).astype(int):np.floor((train_perc+valid_perc)*nTotal).astype(int)]
idx_train = np.concatenate([idxTrain, idxValid])
n_all_train = len(idx_train)

Xtrain = torch.FloatTensor(zscore(x_input[idx_train,:].T,axis=1)) # training set
Xtest = torch.FloatTensor(zscore(x_input[idxTest,:].T,axis=1)) # test set
y = torch.FloatTensor(y_output)

df_smpls_perf = pd.DataFrame(columns=['Perm','samples', 'tau',
                                    'Test_perf_MAE', 'Train_perf_MAE',
                                    'Test_embDiff', 'Train_embDiff'])
df_perf = pd.DataFrame( columns=['Perm', 'Valid_perf_MAE',
                                'Test_perf_MAE', 'Train_perf_MAE'])

C = compute_covariance(Xtrain, cov_type, thr=tau * torch.tensor(np.sqrt(np.log(m) / nTrain)), p=args.p)

iterations = args.iterations
Loss = nn.MSELoss()
MAE = nn.L1Loss()
MSE = nn.MSELoss()
GNN_list = []
pca_list = []

for perm in range(iterations):

    idx_train = np.random.permutation(n_all_train)
    idxTrain = idx_train[0:np.floor(0.88*n_all_train).astype(int)]
    idxValid = idx_train[np.floor(0.88*n_all_train).astype(int)+1:np.floor(n_all_train).astype(int)]
    nTest = int(np.floor(0.1*nTotal))
    nTrain = idxTrain.shape[0]

    xTrain = Xtrain[:,idxTrain]
    xTrain = torch.tensor(np.expand_dims(xTrain, axis=1))
    yTrain = y[idxTrain]
    xTest = Xtest
    xTest = torch.tensor(np.expand_dims(xTest, axis=1))
    yTest = y[idxTest]
    xValid = Xtrain[:,idxValid]
    xValid = torch.tensor(np.expand_dims(xValid, axis=1))
    yValid = y[idxValid]

    GNN = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.ReLU, [m]*len(nFilterTaps), 
                                gml.NoPool, [1]*len(nFilterTaps), dimLayersMLP, C, average=True)

    batchSize = args.batchSize
    nTrainBatches = int(np.ceil(nTrain / batchSize))

    optimizer = optim.Adam(GNN.parameters(), lr=lr, weight_decay=0.001)

    Best_Valid_Loss, Best_Valid_MAPE = 1e10, 1e10
    for epoch in tqdm(range(nEpochs)):
        tot_train_loss = []
        tot_val_mae = 0.
        tot_val_mape = 0.
        train_perm_idx = torch.randperm(nTrainBatches) # shuffle order during training

        for batch in range(nTrainBatches):
            thisBatchIndices = torch.LongTensor(np.arange(nTrain)[batch * batchSize : (batch + 1) * batchSize])
            xTrainBatch = xTrain[:,:,thisBatchIndices].permute((2,1,0))
            yTrainBatch = yTrain[thisBatchIndices].unsqueeze(0)

            GNN.zero_grad()
            yHatTrainBatch = GNN(xTrainBatch[:,:,:])
            lossValueTrain = Loss((yHatTrainBatch) , yTrainBatch.T)
            lossValueTrain.backward()
            optimizer.step()
            tot_train_loss.append(lossValueTrain.detach())
            
        with torch.no_grad():
            yHatValid = GNN(xValid[:,:,:].permute((2,1,0)))

            Valid_Loss = MAE((yHatValid) , yValid.unsqueeze(0).T)

            if Valid_Loss < Best_Valid_Loss:
                Best_Valid_Loss = Valid_Loss
                Best_GNN = deepcopy(GNN)
    
    
    GNN_list.append(Best_GNN)
    yBestValid, embBestValid = Best_GNN.splitForward(xValid[:,:,:].permute((2,1,0)))
    yBestTest, embBestTest = Best_GNN.splitForward(xTest[:,:,:].permute((2,1,0)))
    yBestTrain, embBestTrain = Best_GNN.splitForward(xTrain[:,:,:].permute((2,1,0)))

    df_new_row = pd.DataFrame(data=np.array([[perm,MAE(yBestValid,yValid.unsqueeze(0).T ).detach(),
                                                MAE(yBestTest,yTest.unsqueeze(0).T ).detach(),
                                                MAE(yBestTrain,yTrain.unsqueeze(0).T ).detach()
                                                ]]), 
                                columns=['Perm', 'Valid_perf_MAE',
                                'Test_perf_MAE', 'Train_perf_MAE' ])
    df_perf = pd.concat([df_perf,df_new_row], ignore_index=True)


    ## Stability analysis
    
    for smpls in range(10,nTrain, 10):
        with torch.no_grad():
            smpls_perm = np.random.permutation(smpls)
            C_t =  compute_covariance(Xtrain[:,range(smpls)], cov_type, thr=tau * torch.tensor(np.sqrt(np.log(m) / smpls))) # perturbed covariance matrix
            
            # VNN that imports weights/parameters from the nominal model
            GNN_t = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.ReLU, [m]*len(nFilterTaps), 
                                        gml.NoPool, [1]*len(nFilterTaps), dimLayersMLP, C, average=True)

            GNN_t.GFL.load_state_dict(GNN_list[perm].GFL.state_dict())
            GNN_t.MLP.load_state_dict(GNN_list[perm].MLP.state_dict())
            yHatTest_t, embHatTest_t = GNN_t.splitForward(xTest[:,:,:].permute((2,1,0)))
            ytrain_t, embHatTrain_t = GNN_t.splitForward(xTrain[:,:,:].permute((2,1,0)))
            
            # store VNN, PCA-LR and PCA-rbf performance metrics for stability analysis
            df_new_row_t = pd.DataFrame(data=np.array([[perm,smpls,tau,
                                                    MAE(yHatTest_t,yTest.unsqueeze(0).T ).detach(),
                                                    MAE(ytrain_t,yTrain.unsqueeze(0).T ).detach(),
                                                    torch.sqrt(((embHatTest_t-embBestTest)**2).mean()).detach(),
                                                    torch.sqrt(((embHatTrain_t-embBestTrain)**2).mean()).detach(),
                                                    ]]), 
                                    columns=['Perm','samples','tau',
                                    'Test_perf_MAE', 'Train_perf_MAE',
                                    'Test_embDiff', 'Train_embDiff'])
            df_smpls_perf = pd.concat([df_smpls_perf,df_new_row_t], ignore_index=True)


df_perf.to_csv(f"out/{dset}_{cov_type}_true_vnn_res.csv")
df_smpls_perf.to_csv(f"out/{dset}_{cov_type}_true_vnn_res_stab.csv")
