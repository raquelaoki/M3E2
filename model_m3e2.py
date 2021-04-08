import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class data_nn(object):
    def __init__(self, X_train, X_test, y_train, y_test, treatments_columns, treatment_effec=None,
                 X1_cols=[], X2_cols=None):
        super(data_nn, self).__init__()
        self.X_train = np.delete(X_train, treatments_columns, 1)
        self.X_test = np.delete(X_test, treatments_columns, 1)
        self.y_train = y_train
        self.y_test = y_test
        self.treat_col = treatments_columns
        self.treatment_effec = treatment_effec
        self.T_train = X_train[:, treatments_columns]
        self.T_test = X_test[:, treatments_columns]
        self.X1_cols = X1_cols
        if X2_cols is None:
            X2_cols = range(X_train.shape[1])
        self.X2_cols = X2_cols

        print('M3E2: Train Shape ', self.X_train.shape, self.T_train.shape)

    def loader(self, shuffle=True, batch=250, seed=1):
        X_val, X_test, y_val, y_test, T_val, T_test = train_test_split(self.X_test, self.y_test, self.T_test,
                                                                       test_size=0.5, random_state=seed)
        ''' Creating TensorDataset to use in the DataLoader '''
        dataset_train = TensorDataset(Tensor(self.X_train), Tensor(self.y_train), Tensor(self.T_train))
        dataset_test = TensorDataset(Tensor(X_test), Tensor(y_test), Tensor(T_test))
        dataset_val = TensorDataset(Tensor(X_val), Tensor(y_val), Tensor(T_val))

        ''' Required: Create DataLoader for training the models '''
        loader_train = DataLoader(dataset_train, shuffle=shuffle, batch_size=batch)
        loader_val = DataLoader(dataset_val, shuffle=shuffle, batch_size=X_val.shape[0])
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=X_test.shape[0])

        return loader_train, loader_val, loader_test, len(self.X2_cols) + len(self.X1_cols)


def fit_nn(loader_train, loader_val, loader_test, params, treatement_columns, num_features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # X, y, T = next(iter(loader_train))
    model = M3E2(data='gwas', num_treat=len(treatement_columns), num_exp=params['num_exp'],
                 num_features=num_features)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params["cw_census"][i]).to(device)) for i in
    #             range(num_tasks)]
    # print('lr', params['lr'], type(params['lr']))

    if torch.cuda.is_available():
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params["gamma"])

    loss_train, loss_val = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
    auc_train, auc_val = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])

    loss_av = 0
    best_val_AUC = 0
    best_epoch = 0

    for e in range(params['max_epochs']):
        torch.cuda.empty_cache()
        # for i, batch in enumerate(tqdm(train_loader)):
        loss_av = 0
        auc_av = 0
        # Train
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            ty_train_pred = model(batch[0].to(device))
            ty_train_obs = np.concatenate([batch[2], batch[1]], 1)
            loss = criterion(ty_train_pred.reshape(1, -1), ty_train_obs.reshape(1, -1))
            loss_av += loss
            auc_av += roc_auc_score(ty_train_pred[model.num_treat + 1].cpu().detach().numpy(),
                                    batch[1])
            loss.backward()
            optimizer.step()

        loss_train[e] = loss_av.cpu().detach().numpy() / i
        auc_train[e] = auc_av / i

        # Validation
        X_val, y_val, T_val = next(iter(loader_val))
        ty_val_pred = model(X_val.to(device))
        ty_val_obs = np.concatenate([T_val, y_val], 1)
        loss_val[e] = criterion(ty_val_pred.reshape(1, -1), ty_val_obs.reshape(1, -1)).cpu().detach().numpy()
        auc_val[e] = roc_auc_score(ty_val_pred[model.num_treat + 1].cpu().detach().numpy(), y_val)

        # Best model saved
        if params["best_validation_test"]:
            if auc_val[e] > best_val_AUC:
                best_val_AUC = auc_val[e]
                path = 'm3e2_' + id + 'best.pth'
                torch.save(model.state_dict(), path)

        # Printing
        if e % params['print'] == 0:
            print('...... Train: loss ', loss_train[e], ' and auc ', auc_train[e])
            print('...... Val: loss ', loss_val[e], ' and auc ', auc_val[e])

    if params['best_validation_test']:
        print('... Loading Best validation epoch')
        model.load_state_dict(torch.load(path))

    print('... Final Metrics')
    data_ = [loader_train, loader_val, loader_test]
    data_name = ['Train', 'Validation', 'Test']
    for i, data in enumerate(data_):
        X, y, T = next(iter(data))
        ty_pred = model(X)
        auc = roc_auc_score(ty_pred[model.num_treat + 1].cpu().detach().numpy(), y)
        print('...', data_name[i], ': ', auc)

    return model.outcomeY


class M3E2(nn.Module):
    # https://github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py
    def __init__(self, data, num_treat, num_exp, num_features, expert=None, units_exp=4, use_bias_exp=False,
                 use_bias_gate=False,
                 use_autoencoder=False, y_continuous=False):
        super().__init__()
        self.data = data
        self.num_treat = num_treat
        self.num_exp = num_exp
        self.expert = expert
        self.units_exp = units_exp
        self.units_tower = units_exp
        self.use_bias_exp = use_bias_exp
        self.use_bias_gate = use_bias_gate
        self.use_autoencoder = use_autoencoder
        self.y_continuous = y_continuous
        self.num_features = num_features

        '''Defining experts - number is defined by the user'''
        if self.expert is None:
            self.expert_kernels = nn.ParameterList(
                [
                    nn.Parameter(torch.rand(size=(self.num_features, self.units_exp)).float())
                    for i in range(self.num_exp)
                ]
            )
            self.expert_output = nn.ModuleList(
                [nn.Linear(self.units_exp, 1).float() for i in range(self.num_treat * self.num_exp)]
            )
        '''Defining gates - one per treatment'''
        gate_kernels = torch.rand(self.num_treat, self.num_features, self.num_exp)
        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)

        '''Defining biases - treatments, gates and experts'''
        if self.use_bias_exp:
            self.bias_exp = nn.Parameter(torch.zeros(self.num_exp), requires_grad=True)
        if self.use_bias_gate:
            self.bias_gate = nn.Parameter(torch.zeros(self.num_treat, 1, self.num_exp), requires_grad=True)
        self.bias_treat = nn.Parameter(torch.zeros(self.num_treat), requires_grad=True)
        self.bias_y = nn.Parameter(torch.zeros(1), requires_grad=True)

        '''Defining towers - per treatment'''
        self.HT_list = nn.ModuleList([nn.Linear(self.num_exp, self.units_tower) for i in range(self.num_treat)])
        # self.HY = nn.Linear(self.num_exp, self.units_tower)
        self.propensityT = nn.ModuleList([nn.Linear(self.units_tower, 1) for i in range(self.num_treat)])
        self.outcomeY = nn.Parameter(torch.rand(size=(self.num_treat + self.units_tower, 1)))

        '''TODO: Defining'''
        # autoencoder
        if self.use_autoencoder:
            print('missing')
        # propensity score

        '''Defining activation functions and others'''
        self.dropout = nn.Dropout(0.25)
        self.tahn = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        print('...model initialization done!')

    def forward(self, inputs, treat_assignment=None):
        n = inputs.shape[0]
        # MISSING AUTOENCODER

        ''' Calculating Experts'''
        for i in range(self.num_exp):
            aux = torch.mm(inputs, self.expert_kernels[i]).reshape((n, self.expert_kernels[i].shape[1]))
            if i == 0:
                expert_outputs = self.expert_output[i](aux)
            else:
                expert_outputs = torch.cat((expert_outputs, self.expert_output[i](aux)), dim=1)

        if self.use_bias_exp:
            for i in range(self.num_exp):
                expert_outputs[i] = expert_outputs[i].add(self.bias_exp[i])
        expert_outputs = F.relu(expert_outputs)

        '''Calculating Gates'''
        for i in range(self.num_treat):
            if i == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[i]).reshape(1, n, self.num_exp)
            else:
                gate_outputs = torch.cat(
                    (gate_outputs, torch.mm(inputs, self.gate_kernels[i]).reshape(1, n, self.num_exp)),
                    dim=0,
                )

        if self.use_bias_gate:
            gate_outputs = gate_outputs.add(self.bias_gate)

        gate_outputs = F.softmax(gate_outputs, dim=2)
        ''' Multiplying gates x experts  + propensity score output - T1, ..., Tk'''
        output = []
        for treatment in range(self.num_treat):
            gate = gate_outputs[treatment]
            out_t = torch.mul(gate, expert_outputs).reshape(1, gate.shape[0], gate.shape[1])
            out_t = out_t.add(self.bias_treat[treatment])
            out_t = self.HT_list[treatment](out_t)
            if treatment == 0:
                HY = out_t
            else:
                HY = HY + out_t
            out_t = self.tahn(out_t)
            out_t = self.propensityT[treatment](out_t)
            out_t = self.sigmoid(out_t)
            output.append(out_t)

        '''Outcome output - Y'''
        HY = HY / self.num_treat
        if treat_assignment is not None:
            print('TODO')
        else:
            aux = np.concatenate([output, HY], axis=1)
            out = torch.mul(aux, self.outcomeY)
            out = out.add(self.bias_y)
            # if y_continuous:
            #     out = self.tahn(out)
            # else:
            out = self.sigmoid(out)
            output.append(out)

        return output
