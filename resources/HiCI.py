import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, mean_squared_error
from torch import Tensor


class HiCI():
    def __init__(self, X_train, X_test, y_train, y_test, treatments_columns):
        super(HiCI, self).__init__()

        #Selecting only rows where one treatment was observed
        self.treatments_columns = treatments_columns
        self.covariates_columns = [col for col in list(range(X_train.shape[1])) if col not in treatments_columns]
        if len(treatments_columns)>1:
            T_train = X_train[:, treatments_columns]
            T_test = X_test[:, treatments_columns]
            X_train = np.delete(X_train, treatments_columns, 1)
            X_test = np.delete(X_test, treatments_columns, 1)
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

            rowSumTrain = T_train.sum(1)
            rowSumTest = T_test.sum(1)

            X_train = X_train[rowSumTrain==1,:]
            X_test = X_test[rowSumTest==1,:]
            y_train = y_train[rowSumTrain==1]
            y_test = y_test[rowSumTest==1]
            T_train = T_train[rowSumTrain==1,:]
            T_test = T_test[rowSumTest==1,:]
        else:
            T_train = X_train[:, treatments_columns]
            T_test = X_test[:, treatments_columns]
            X_train = np.delete(X_train, treatments_columns, 1)
            X_test = np.delete(X_test, treatments_columns, 1)
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)


        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = None
        self.ate = None
        self.f1_test = None
        self.test_output_list = []
        self.train_output_list = []
        self.T_train = T_train
        self.T_test = T_test
        print('Running HiCI')

    def loader(self, shuffle=True, batch=250, seed=1):
        X_train, X_val, y_train, y_val, T_train, T_val = train_test_split(self.X_train, self.y_train, self.T_train,
                                                                          test_size=0.15, random_state=seed)
        ''' Creating TensorDataset to use in the DataLoader '''
        dataset_train = TensorDataset(Tensor(X_train), Tensor(y_train), Tensor(T_train))
        dataset_test = TensorDataset(Tensor(self.X_test), Tensor(self.y_test), Tensor(self.T_test))
        dataset_val = TensorDataset(Tensor(X_val), Tensor(y_val), Tensor(T_val))

        ''' Required: Create DataLoader for training the models '''
        loader_train = DataLoader(dataset_train, shuffle=shuffle, batch_size=batch)
        loader_val = DataLoader(dataset_val, shuffle=shuffle, batch_size=X_val.shape[0])
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=self.X_test.shape[0])

        return loader_train, loader_val, loader_test

    def fit(self, total_epochs=100, lr=0.1,
            hidden1=64, hidden2=32, print_=True, weight_decay=0.9, loss_weight=[5,0.05,2],
            gamma=0.8, batch=250, type_target='binary'):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device:", device)
        loader_train, loader_val, loader_test = self.loader(batch=batch)
        model = HiCI_DNN(len(self.treatments_columns), len(self.covariates_columns), hidden1, hidden2)
        if torch.cuda.is_available():
            model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)#0.05
        if type_target == 'binary':
            criteria = [torch.nn.MSELoss(reduction='mean'), loss_ce, torch.nn.BCEWithLogitsLoss()]
        else:
            criteria = [torch.nn.MSELoss(reduction='mean'), loss_ce, torch.nn.MSELoss()]

        train_loss_values = []
        val_loss_values = []

        train_ce_loss = []
        train_ae_loss = []
        train_pred_loss = []

        for epoch in range(total_epochs):
            # ===== training =====
            train_running_loss = 0.0
            train_running_loss_ce = 0.0
            train_running_loss_ae = 0.0
            train_running_loss_pred = 0.0

            # model.train() incorrect call
            for i, data in enumerate(loader_train, 0):
                X, y_obs, T = data[0].to(device), data[1].to(device), data[2].to(device)
                optimizer.zero_grad()
                decoded, y_pred, ptk = model(T, X)
                l_ae = criteria[0](X, decoded)*loss_weight[0]
                l_ce = criteria[1](T, ptk)*loss_weight[1]
                l_pred = criteria[2](input=y_obs, target=y_pred)*loss_weight[2]
                loss = l_ae + l_ce + l_pred

                loss.backward()
                optimizer.step()

                train_running_loss += loss.item()
                train_running_loss_ce += l_ce.item()
                train_running_loss_ae += l_ae.item()
                train_running_loss_pred += l_pred.item()


            #print(criteria[1](T, ptk, True))

            scheduler.step()
            train_loss_values.append(train_running_loss / (i + 1))
            train_ce_loss.append(train_running_loss_ce / (i + 1))
            train_ae_loss.append(train_running_loss_ae / (i + 1))
            train_pred_loss.append(train_running_loss_pred / (i + 1))

            # ===== validation =====
            val_running_loss = 0.0
            for i, data in enumerate(loader_val, 0):
                X, y_obs, T = data[0].to(device), data[1].to(device), data[2].to(device)
                # print('target',y_pred)
                decoded, y_pred, ptk = model(T, X)
                l_ae = criteria[0](X, decoded)
                l_ce = criteria[1](T, ptk)
                l_pred = criteria[2](y_obs, y_pred)
                loss = l_ae + l_ce + l_pred
                val_running_loss += loss.item()

            val_loss_values.append(val_running_loss / (i + 1))

        print('losses - final batch ')
        print(l_ae.item(), l_ce.item(), l_pred.item())
        print("Finishing Training")
        y_train_pred = model(Tensor(self.T_train).to(device), Tensor(self.X_train).to(device), test=True).cpu().detach().numpy()
        y_test_pred = model(Tensor(self.T_test).to(device), Tensor(self.X_test).to(device), test=True).cpu().detach().numpy()
        self.model = model

        if len(np.unique(self.y_train)) == 2:
            # print('observed', self.y_train.shape)
            # print('pred', y_test_pred.shape)
            thhold = self.Find_Optimal_Cutoff(self.y_train, y_train_pred)
            y_train_pred01 = [0 if item < thhold else 1 for item in y_train_pred]
            y_test_pred01 = [0 if item < thhold else 1 for item in y_test_pred]
            if print_:
                print('... Evaluation:')
                print('... Training set: F1 - ', f1_score(self.y_train, y_train_pred01))
                print('...... confusion matrix: ', confusion_matrix(self.y_train, y_train_pred01).ravel())

                print('... Testing set: F1 - ', f1_score(self.y_test, y_test_pred01))
                print('...... confusion matrix: ', confusion_matrix(self.y_test, y_test_pred01).ravel())
            f1_test = f1_score(self.y_test, y_test_pred01)
        else:
            print('... Evaluation:')
            print('... Training set: MSE - ', mean_squared_error(self.y_train, y_train_pred))

            print('... Testing set: MSE - ', mean_squared_error(self.y_test, y_test_pred))
            f1_test = mean_squared_error(self.y_test, y_test_pred)
        self.f1_test = f1_test

    def cate(self, dataset='all'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset == 'train':
            X = self.X_train
            T = self.T_train
        if dataset == 'test':
            X = self.X_test
            T = self.T_test
        if dataset == 'all':
            X = np.concatenate([self.X_train, self.X_test], axis=0)
            T = np.concatenate([self.T_train, self.T_test], axis=0)
        outcome = self.model(Tensor(T).to(device), Tensor(X).to(device), test=True).cpu().detach().numpy()
        #print('outcome', outcome)
        cate = []
        if len(self.treatments_columns)>1:
            for i in range(len(self.treatments_columns)):
                q_t1 = np.mean(outcome[np.where(T[:, i] == 1)[0]])
                q_t0 = np.mean(outcome[np.where(T[:, i] == 0)[0]])
                cate.append(q_t1 - q_t0)
        else:
            #print('IM HERE')
            q_t1 = np.mean(outcome[np.where(T == 1)[0]])
            q_t0 = np.mean(outcome[np.where(T == 0)[0]])
            #print(q_t1,q_t0)
            cate.append(q_t1 - q_t0)
        self.cate = cate

        return self.cate

    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations
        predicted : Matrix with predicted data, where rows are observations
        Returns
        -------
        list type, with optimal cutoff value
        https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t['threshold'])


class autoencoder(nn.Module):
    # https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    def __init__(self, input, hidden1=64, hidden2=8):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input, out_features=hidden1
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden1, out_features=hidden2
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=hidden2, out_features=hidden1
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden1, out_features=input
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return code, reconstructed


class HiCI_DNN(nn.Module):
    def __init__(self, num_treatments, num_features,
                 hidden1, hidden2):
        super(HiCI_DNN, self).__init__()

        self.k = num_treatments
        self.num_features = num_features
        self.theta_len = hidden2 + num_treatments
        self.decorrelation_net = autoencoder(num_features, hidden1=hidden1, hidden2=hidden2)
        self.theta = nn.ParameterList(
            [
                nn.Parameter(torch.rand(size=(hidden2, 1)))
                for i in range(self.k)
            ]
        )
        self.output = nn.Linear(self.theta_len, 1)

    def forward(self, treatments, covariates, test=False):
        encoded, decoded = self.decorrelation_net(covariates)
        embedding = torch.cat([encoded, treatments], dim=1)
        # cross-entropy per treatment
        for k in range(self.k):
            if k == 0:
                ptk = torch.mm(encoded, self.theta[k])
            else:
                ptk_ = torch.mm(encoded, self.theta[k])
                ptk = torch.cat([ptk, ptk_], 1)
        y_pred = self.output(embedding)
        if test:
            return y_pred
        else:
            return decoded, y_pred, ptk


def loss_ce(ptk_obs, ptk_pred, print_=False):
    prior = torch.sum(ptk_obs, 0)
    prior = prior / ptk_obs.shape[0]
    if print_:
        print('obs ',ptk_obs,'/nPred', ptk_pred)
    total_sum = torch.sum(torch.exp(ptk_pred))
    ce = 0
    for k in range(ptk_obs.shape[1]):
        ce += -prior[k] * ptk_pred[k] - prior[k] * torch.log(total_sum)
    return torch.mean(ce)
