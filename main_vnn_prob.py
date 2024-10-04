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

dset = args.dset
if dset not in ["SmallCov", "LargeCov"]:
    raise NotImplementedError("Dataset is invalid")

dimNodeSignals = args.dimNodeSignals
L = len(dimNodeSignals) - 1
nFilterTaps = [args.filter_taps] * L
dimLayersMLP = args.dimLayersMLP 
lr = args.lr
m = args.m
cov_type = args.cov_type
n_it = args.n_it 

x_input, y_output = load_data(dset)
x_input = zscore(x_input.T,axis=1)

nEpochs = args.nEpochs

train_perc, valid_perc, test_perc = 0.8, 0.1, 0.1

nTotal = y_output.shape[0]
idxTotal = np.random.permutation(nTotal)
idxTest = idxTotal[np.floor((train_perc + valid_perc)*nTotal).astype(int):]
idxTrain = idxTotal[0:np.floor(train_perc*nTotal).astype(int)]
idxValid = idxTotal[np.floor(train_perc*nTotal).astype(int):np.floor((train_perc+valid_perc)*nTotal).astype(int)]

Xtrain = torch.FloatTensor(x_input[:,idxTrain]) # training set
Xvalid = torch.FloatTensor(x_input[:,idxValid]) # training set
Xtest = torch.FloatTensor(x_input[:,idxTest]) # training set
m = Xtrain.shape[0]
y = torch.FloatTensor(y_output)
C = compute_covariance(Xtrain, "standard") # always train with true covariance

iterations = args.iterations
Loss = nn.MSELoss()
MAE = nn.L1Loss()
MSE = nn.MSELoss()
GNN_list = []
pca_list = []

df_smpls_perf = pd.DataFrame(columns=['Perm','it','prob','prob_type',
                                    'Test_perf_MAE', 'Train_perf_MAE'])
df_perf = pd.DataFrame( columns=['Perm', 'Valid_perf_MAE',
                                'Test_perf_MAE', 'Train_perf_MAE'])


for perm in range(iterations):

    nTest = int(np.floor(0.1*nTotal))
    nTrain = idxTrain.shape[0]

    xTrain = Xtrain
    xTrain = torch.tensor(np.expand_dims(xTrain, axis=1))
    yTrain = y[idxTrain]
    xTest = Xtest
    xTest = torch.tensor(np.expand_dims(xTest, axis=1))
    yTest = y[idxTest]
    xValid = Xvalid
    xValid = torch.tensor(np.expand_dims(xValid, axis=1))
    yValid = y[idxValid]

    GNN = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.LeakyReLU, [m]*len(nFilterTaps), 
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
            Valid_Loss = MAE(yHatValid, yValid.unsqueeze(0).T)

            if Valid_Loss < Best_Valid_Loss:
                Best_Valid_Loss = Valid_Loss
                Best_GNN = deepcopy(GNN)
    
    
    GNN_list.append(deepcopy(Best_GNN))
    yBestValid = Best_GNN(xValid[:,:,:].permute((2,1,0)))
    yBestTest = Best_GNN(xTest[:,:,:].permute((2,1,0)))
    yBestTrain = Best_GNN(xTrain[:,:,:].permute((2,1,0)))

    df_new_row = pd.DataFrame(data=np.array([[perm,MAE(yBestValid,yValid.unsqueeze(0).T ).detach().item(),
                                                MAE(yBestTest,yTest.unsqueeze(0).T ).detach().item(),
                                                MAE(yBestTrain,yTrain.unsqueeze(0).T ).detach().item()
                                                ]]), 
                                columns=['Perm', 'Valid_perf_MAE',
                                'Test_perf_MAE', 'Train_perf_MAE' ])
    df_perf = pd.concat([df_perf,df_new_row], ignore_index=True)

    print(df_new_row)

    # Add results with true covariance
    df_smpls_new_row = pd.DataFrame(data=np.array([[perm,0,'true',0,
                                                MAE(yBestTest,yTest.unsqueeze(0).T ).detach().item(),
                                                MAE(yBestTrain,yTrain.unsqueeze(0).T ).detach().item()
                                                ]]), 
                                columns=['Perm','it', 'prob_type', 'prob',
                                'Test_perf_MAE', 'Train_perf_MAE' ])
    df_smpls_perf = pd.concat([df_smpls_perf,df_smpls_new_row], ignore_index=True)

    print("Stochastic stability test")
    for it in tqdm(range(n_it)): # iterations for stochasticity
        ## Stability analysis

        # Begin with prob = cov
        C_t =  compute_covariance(Xtrain, "ACV") # perturbed covariance matrix
        
        # VNN that imports weights/parameters from the nominal model
        GNN_t = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.LeakyReLU, [m]*len(nFilterTaps), 
                                gml.NoPool, [1]*len(nFilterTaps), dimLayersMLP, C_t, average=True)

        GNN_t.GFL.load_state_dict(GNN_list[perm].GFL.state_dict())
        GNN_t.MLP.load_state_dict(GNN_list[perm].MLP.state_dict())
        yHatTest_t = GNN_t(xTest[:,:,:].permute((2,1,0)))
        ytrain_t = GNN_t(xTrain[:,:,:].permute((2,1,0)))
        
        # store VNN, PCA-LR and PCA-rbf performance metrics for stability analysis
        df_new_row_t = pd.DataFrame(data=np.array([[perm,it,'prob_cov',0,
                                                MAE(yHatTest_t,yTest.unsqueeze(0).T ).detach().item(),
                                                MAE(ytrain_t,yTrain.unsqueeze(0).T ).detach().item(),
                                                ]]), 
                                columns=['Perm','it', 'prob_type','prob',
                                'Test_perf_MAE', 'Train_perf_MAE' ])
        df_smpls_perf = pd.concat([df_smpls_perf,df_new_row_t], ignore_index=True)

        # Now percentile probabilities
        for prob in np.linspace(0.01,0.99,20):
            with torch.no_grad():
                C_t =  compute_covariance(Xtrain, 'RCV', p=prob) # perturbed covariance matrix
                
                # VNN that imports weights/parameters from the nominal model
                GNN_t = archit.SelectionGNN(dimNodeSignals, nFilterTaps, True, nn.LeakyReLU, [m]*len(nFilterTaps), 
                                gml.NoPool, [1]*len(nFilterTaps), dimLayersMLP, C_t, average=True)

                GNN_t.GFL.load_state_dict(GNN_list[perm].GFL.state_dict())
                GNN_t.MLP.load_state_dict(GNN_list[perm].MLP.state_dict())
                yHatTest_t = GNN_t(xTest[:,:,:].permute((2,1,0)))
                ytrain_t = GNN_t(xTrain[:,:,:].permute((2,1,0)))
                
                # store VNN, PCA-LR and PCA-rbf performance metrics for stability analysis
                df_new_row_t = pd.DataFrame(data=np.array([[perm,it, 'RCV',prob,
                                                        MAE(yHatTest_t,yTest.unsqueeze(0).T ).detach().item(),
                                                        MAE(ytrain_t,yTrain.unsqueeze(0).T ).detach().item(),
                                                        ]]), 
                                        columns=['Perm','it','prob_type','prob',
                                        'Test_perf_MAE', 'Train_perf_MAE',])
                df_smpls_perf = pd.concat([df_smpls_perf,df_new_row_t], ignore_index=True)


df_perf.to_csv(f"out/{dset}_prob_vnn_res.csv")
df_smpls_perf.to_csv(f"out/{dset}_prob_vnn_res_stab.csv")

