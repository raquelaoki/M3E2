import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix


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
        self.treat_weights = 1 / (self.T_train.sum(0) / self.T_train.shape[0])
        print('Weights', self.treat_weights)

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


def roc_auc_batch(pred, obs, auc=None, count=0, errorm=''):
    sigmoid = nn.Sigmoid()
    y01_pred = sigmoid(Tensor(pred)).numpy()
    y01_pred = [1 if item > 0.5 else 0 for item in y01_pred]
    if auc is None:
        # try:
        # roc = roc_auc_score(obs, y01_pred)
        # if roc == 0.5:
        #    print(confusion_matrix(obs, y01_pred))
        return roc_auc_score(obs, y01_pred)
        # except ValueError:
        #    print(errorm)
        #    if errorm == 'Treat - Val Full':
        #        print('Positive Examples', obs.sum())
        #    return np.nan

    else:
        try:
            auc += roc_auc_score(obs, y01_pred)
            count += 1
            return auc, count
        except ValueError:
            return auc, count


def fit_nn(loader_train, loader_val, loader_test, params, treatement_columns, num_features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # X, y, T = next(iter(loader_train))
    model = M3E2(data='gwas', num_treat=len(treatement_columns), num_exp=params['num_exp'],
                 num_features=num_features, dropoutp=params['dropoutp'])
    # criterion = nn.BCEWithLogitsLoss()
    if params['type_treatment'] == 'binary':
        criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params['pos_weights'][i])) for i in
                     range(model.num_treat)]
    else:
        criterion = [nn.MSELoss() for i in range(model.num_treat)]
    if params['type_target'] == 'binary':
        # Target
        criterion.append(nn.BCEWithLogitsLoss())
    else:
        criterion.append(nn.MSELoss())

    if torch.cuda.is_available():
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params["gamma"])

    loss_train, loss_val = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
    auc_train, auc_val = [], []  # np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])

    best_val_AUC = 0
    best_epoch = 0
    print('... Training')

    for e in range(params['max_epochs']):
        auc_train_, auc_val_ = np.zeros(model.num_treat + 1), np.zeros(model.num_treat + 1)
        torch.cuda.empty_cache()
        # for i, batch in enumerate(tqdm(train_loader)):
        loss_av = 0
        auc_count_ = np.zeros(model.num_treat + 1)
        # Train alternating
        # TODO: Task balance
        for i, batch in enumerate(loader_train):

            ty_train_pred = model(batch[0].to(device))
            # Doing by task / target

            # For treat
            loss_batch_treats, loss_batch_target = 0, 0
            optimizer.zero_grad()
            for j in range(model.num_treat):
                loss_batch_treats += criterion[j](ty_train_pred[:, j].reshape(-1), batch[2][:, j].reshape(-1).to(device))
                auc_train_[j], auc_count_[j] = roc_auc_batch(ty_train_pred[:, j].cpu().detach().numpy(),
                                                             batch[2][:, j].reshape(-1),
                                                             auc_train_[j], auc_count_[j])
            #loss_av += loss_batch_treats.cpu().detach().numpy()

            # For target
            j = model.num_treat
            loss_batch_target = criterion[j](ty_train_pred[:, j].reshape(-1),
                                      batch[1].reshape(-1).to(device))
            auc_train_[j], auc_count_[j] = roc_auc_batch(ty_train_pred[:, j].cpu().detach().numpy(),
                                                         batch[1].reshape(-1),
                                                         auc_train_[j], auc_count_[j])
            #print(loss_batch_treats.cpu().detach().numpy(), loss_batch_target.cpu().detach().numpy())
            alpha = loss_batch_treats.cpu().detach().numpy()/loss_batch_target.cpu().detach().numpy()
            loss_batch = loss_batch_treats + loss_batch_target*alpha
            loss_batch.backward()
            loss_av += loss_batch.cpu().detach().numpy()
            optimizer.step()

        loss_train[e] = loss_av / i
        auc_count_ = [np.max([i, 1]) for i in auc_count_]
        print('line 140',auc_train_,auc_count_)
        auc_train.append(np.divide(auc_train_, auc_count_))

        # Validation
        X_val, y_val, T_val = next(iter(loader_val))
        ty_val_pred = model(X_val.to(device))
        for j in range(ty_val_pred.shape[1]):
            if j == ty_val_pred.shape[1] - 1:
                loss_val[e] += criterion[j](ty_val_pred[:, j].reshape(-1),
                                            y_val.reshape(-1).to(device)).cpu().detach().numpy() * model.num_treat * 2
                errorm = 'Target - Val full'
                auc_val_[j] = roc_auc_batch(ty_val_pred[:, j].cpu().detach().numpy(),
                                            y_val.reshape(-1), errorm=errorm)
            else:
                loss_val[e] += criterion[j](ty_val_pred[:, j].reshape(-1),
                                            T_val[:, j].reshape(-1).to(device)).cpu().detach().numpy()
                errorm = 'Treat - Val Full'
                auc_val_[j] = roc_auc_batch(ty_val_pred[:, j].cpu().detach().numpy(),
                                            T_val[:, j].reshape(-1), errorm=errorm)

        # Best model saved
        auc_val.append(auc_val_)
        if params["best_validation_test"]:
            # TODO: check sum with np.nan values
            if np.sum(auc_val[e]) > best_val_AUC:
                best_epoch = e
                best_val_AUC = np.sum(auc_val[e])
                path = 'm3e2_' + params['id'] + 'best.pth'
                torch.save(model.state_dict(), path)

        # Printing
        if e % params['print'] == 0:
            opt_scheduler.step()
            print('...... ', e, ' \nTrain: loss ', round(loss_train[e], 2), 'auc ', auc_train[e],
                  '\nVal: loss ', round(loss_val[e], 2), 'auc ', auc_val[e])

    if params['best_validation_test'] and best_epoch > 0:
        print('... Loading Best validation (epoch ', best_epoch, ')')
        model.load_state_dict(torch.load(path))

    print('... Final Metrics - Target')
    data_ = [loader_train, loader_val, loader_test]
    data_name = ['Train', 'Val', 'Test']
    sigmoid = nn.Sigmoid()
    for i, data in enumerate(data_):
        X, y, T = next(iter(data))
        ty_pred = model(X.to(device))
        y01_pred = sigmoid(ty_pred[:, model.num_treat]).cpu().detach().numpy()
        y01_pred = [1 if item > 0.5 else 0 for item in y01_pred]
        try:
            auc = roc_auc_score(y, y01_pred)
            print(confusion_matrix(y, y01_pred))
        except ValueError:
            aux = np.nan()
        print('......', data_name[i], ': ', round(auc, 3))

    return model.outcomeY[0:model.num_treat].cpu().detach().numpy().reshape(-1)


class M3E2(nn.Module):
    # https://github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py
    def __init__(self, data, num_treat, num_exp, num_features, expert=None, units_exp=4, use_bias_exp=False,
                 use_bias_gate=False, use_autoencoder=False, y_continuous=False, dropoutp=0.25):
        super().__init__()
        self.data = data  # name only
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
        self.dropout = nn.Dropout(dropoutp)
        self.tahn = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        print('... Model initialization done!')

    def forward(self, inputs, treat_assignment=None, device=None):
        n = inputs.shape[0]
        # MISSING AUTOENCODER
        '''Dropout'''
        inputs = self.dropout(inputs)

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
        # output = []
        for i, treatment in enumerate(range(self.num_treat)):
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
            # out_t = self.sigmoid(out_t)
            if i == 0:
                output = out_t.reshape(-1, 1)
            else:
                output = torch.cat((output, out_t.reshape(-1, 1)), 1)

        # output = torch.transpose(output, 0, 1)
        '''Outcome output - Y'''
        HY = HY / self.num_treat
        HY = torch.reshape(HY, (HY.shape[1], HY.shape[2]))

        if treat_assignment is not None:
            print('TODO')
        else:
            aux = torch.cat((output, HY), 1)
            out = torch.matmul(aux, self.outcomeY)
            out = out.add(self.bias_y)
            # if y_continuous:
            #     out = self.tahn(out)
            # else:
            # out = self.sigmoid(out)
            # because BCEwithLogitsLoss combines sigmoid and BCELoss
            output = torch.cat((output, out.reshape(-1, 1)), 1)

        return output
