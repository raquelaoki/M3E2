import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, f1_score, accuracy_score
from sklearn.metrics import precision_score

logger = logging.getLogger(__name__)


class data_nn(object):
    def __init__(self, X_train, X_test, y_train, y_test, treatments_columns, treatment_effec=None,
                 Xlow_cols=[], Xhigh_cols=None):
        super(data_nn, self).__init__()
        self.X_train = np.delete(X_train, treatments_columns, 1)
        self.X_test = np.delete(X_test, treatments_columns, 1)
        self.y_train = y_train
        self.y_test = y_test
        self.treat_col = treatments_columns
        self.treatment_effec = treatment_effec
        self.T_train = X_train[:, treatments_columns]
        self.T_test = X_test[:, treatments_columns]
        self.Xlow_cols = Xlow_cols
        if Xhigh_cols is None:
            Xhigh_cols = range(X_train.shape[1])
        self.Xhigh_cols = Xhigh_cols
        logger.debug('M3E2: Train Shape ', self.X_train.shape, self.T_train.shape)

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

        return loader_train, loader_val, loader_test, len(self.Xhigh_cols) + len(self.Xlow_cols)


def metric_batch(pred, obs, auc=None, count=0, is_binary=True):
    if is_binary:
        sigmoid = nn.Sigmoid()
        y01_pred = sigmoid(Tensor(pred)).numpy()
        y01_pred = [1 if item > 0.5 else 0 for item in y01_pred]
        if auc is None:
            try:
                return accuracy_score(obs, y01_pred)
            except ValueError:
                return np.nan
        else:
            try:
                auc += accuracy_score(obs, y01_pred)
                count += 1
                return auc, count
            except ValueError:
                return auc, count
    else:
        if auc is None:
            return mean_squared_error(obs, pred)
        else:
            return mean_squared_error(obs, pred), 1


def metric_precision(pred, obs, is_binary=True):
    if is_binary:
        sigmoid = nn.Sigmoid()
        y01_pred = sigmoid(Tensor(pred)).numpy()
        y01_pred = [1 if item > 0.5 else 0 for item in y01_pred]
        return precision_score(obs, y01_pred, labels=[0, 1])
    else:
        return 999


def fit_nn(loader_train, loader_val, loader_test,
           params,
           treatement_columns,
           Xlow_cols,
           Xhigh_cols=None,
           use_bias_y=False,
           path_save_model='savedmodels/'
           ):
    """Fit M3E2 = creates model, loss function, and training.

    Parameters
    ----------
    loader_train: TensorDataset obj.
    loader_val: TensorDataset obj.
    loader_test: TensorDataset obj.
    params: dictionary.
    treatement_columns: [int, int...].
    Xlow_cols: [].
    Xhigh_cols: [].
    use_bias_y: bool.
    path_save_model: str.

    Returns
    -------
    y_pred, score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating model.
    model = M3E2(num_treat=len(treatement_columns),
                 num_exp=params['num_exp'],
                 dropoutp=params['dropoutp'],
                 Xlow_cols=Xlow_cols,
                 Xhigh_cols=Xhigh_cols,
                 hidden1=params['hidden1'],
                 hidden2=params['hidden2'],
                 use_bias_y=use_bias_y,
                 units_exp=params['units_exp'])

    # Adding criterion.
    if params['is_binary_treatment']:
        criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params['pos_weights'][i])) for i in
                     range(model.num_treat)]
    else:
        criterion = [nn.MSELoss() for i in range(model.num_treat)]

    if params['is_binary_target']:
        criterion.append(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params['pos_weight_y'])))
    else:
        criterion.append(nn.MSELoss())

    if Xhigh_cols is not None:
        ae_criterion = nn.MSELoss()

    # Moving model to device and creating optimizer.
    if torch.cuda.is_available():
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params["gamma"])

    loss_train, loss_val = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
    metric_train, metric_val = [], []  # np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
    metric_train_p, metric_val_p = [], []  # np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
    loss_train_ae, loss_val_ae = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])

    if params['is_binary_target']:
        best_val_metric = 0
    else:
        best_val_metric = 999
    best_epoch = 0
    logger.debug('... Training')

    for e in range(params['max_epochs']):
        metric_train_, metric_val_ = np.zeros(model.num_treat + 1), np.zeros(model.num_treat + 1)
        metric_train_p_, metric_val_p_ = np.zeros(model.num_treat), np.zeros(model.num_treat)
        torch.cuda.empty_cache()
        loss_av, loss_ae_av = 0, 0
        metric_count_ = np.zeros(model.num_treat + 1)

        for i, batch in enumerate(loader_train):
            ty_train_pred, X2_reconstruct = model(batch[0].to(device), batch[2].to(device))
            # Doing by task / target
            # For treat
            loss_batch_treats, loss_batch_target = 0, 0
            optimizer.zero_grad()
            for j in range(model.num_treat):
                loss_batch_treats += criterion[j](ty_train_pred[:, j].reshape(-1),
                                                  batch[2][:, j].reshape(-1).to(device))
                metric_train_[j], metric_count_[j] = metric_batch(pred=ty_train_pred[:, j].cpu().detach().numpy(),
                                                                  obs=batch[2][:, j].reshape(-1),
                                                                  auc=metric_train_[j],
                                                                  count=metric_count_[j],
                                                                  is_binary=params['is_binary_treatment'])
                metric_train_p_[j] = metric_precision(pred=ty_train_pred[:, j].cpu().detach().numpy(),
                                                      obs=batch[2][:, j].reshape(-1),
                                                      is_binary=params['is_binary_treatment'])
            # For target
            j = model.num_treat
            loss_batch_target = criterion[j](ty_train_pred[:, j].reshape(-1),
                                             batch[1].reshape(-1).to(device))
            metric_train_[j], metric_count_[j] = metric_batch(pred=ty_train_pred[:, j].cpu().detach().numpy(),
                                                              obs=batch[1].reshape(-1),
                                                              auc=metric_train_[j],
                                                              count=metric_count_[j],
                                                              is_binary=params['is_binary_target'])
            if Xhigh_cols is not None:
                loss_ae = ae_criterion(X2_reconstruct, batch[0][:, Xhigh_cols].to(device))
                loss_ae_av += loss_ae.cpu().detach().numpy()
                loss_batch = loss_batch_treats * params['loss_treat'] + loss_batch_target * params[
                    'loss_target'] + loss_ae * params['loss_da']
                print_losses_noweights = [loss_batch_treats.cpu().detach().numpy(),
                                          loss_batch_target.cpu().detach().numpy(),
                                          loss_ae.cpu().detach().numpy()]
                print_losses_withweights = [loss_batch_treats.cpu().detach().numpy() * params['loss_treat'],
                                            loss_batch_target.cpu().detach().numpy() * params['loss_target'],
                                            loss_ae.cpu().detach().numpy() * params['loss_da']]
            else:
                loss_batch = loss_batch_treats * params['loss_treat'] + loss_batch_target * params[
                    'loss_target']
            loss_batch.backward()
            loss_av += loss_batch.cpu().detach().numpy()
            optimizer.step()

        loss_train[e] = loss_av / i
        if Xhigh_cols is not None:
            loss_train_ae[e] = loss_ae_av / i
        metric_count_ = [np.max([i, 1]) for i in metric_count_]
        metric_train.append(np.divide(metric_train_, metric_count_))
        metric_train_p.append(metric_train_p_)

        # Validation
        X_val, y_val, T_val = next(iter(loader_val))
        ty_val_pred, X2_reconstruct_val = model(X_val.to(device), T_val.to(device))
        for j in range(ty_val_pred.shape[1]):
            if j == ty_val_pred.shape[1] - 1:
                loss_val[e] += criterion[j](ty_val_pred[:, j].reshape(-1),
                                            y_val.reshape(-1).to(device)).cpu().detach().numpy() * params['loss_target']
                metric_val_[j] = metric_batch(pred=ty_val_pred[:, j].cpu().detach().numpy(),
                                              obs=y_val.reshape(-1),
                                              is_binary=params['is_binary_target'])
            else:
                loss_val[e] += criterion[j](ty_val_pred[:, j].reshape(-1),
                                            T_val[:, j].reshape(-1).to(device)).cpu().detach().numpy() * params[
                                   'loss_treat']
                metric_val_[j] = metric_batch(pred=ty_val_pred[:, j].cpu().detach().numpy(),
                                              obs=T_val[:, j].reshape(-1),
                                              is_binary=params['is_binary_treatment'])
                metric_val_p_[j] = metric_precision(pred=ty_val_pred[:, j].cpu().detach().numpy(),
                                                    obs=T_val[:, j].reshape(-1),
                                                    is_binary=params['is_binary_treatment'])
        if Xhigh_cols is not None:
            loss_val_ae[e] = ae_criterion(X2_reconstruct_val, X_val[:, Xhigh_cols].to(device)).cpu().detach().numpy() * \
                             params['loss_da']

        # Saving best model.
        metric_val.append(metric_val_)
        metric_val_p.append(metric_val_p_)
        if params["best_validation_test"]:
            # Metric based on target.
            if params['is_binary_target']:
                # largest f1 score or precision or accuracy.
                if np.sum(metric_val[e][-1]) > best_val_metric:
                    best_epoch = e
                    best_val_metric = np.sum(metric_val[e][-1])
                    path = path_save_model + 'm3e2_' + params['id'] + 'best.pth'
                    torch.save(model.state_dict(), path)
            else:
                # smallest RMSE.
                if np.sum(metric_val[e][-1]) < best_val_metric:
                    best_epoch = e
                    best_val_metric = np.sum(metric_val[e][-1])
                    path = path_save_model + 'm3e2_' + params['id'] + 'best.pth'
                    torch.save(model.state_dict(), path)

        # Logging.
        if e % params['print'] == 0:
            if Xhigh_cols is not None:
                logger.debug('...... ', e, ' \n... Train: loss ', round(loss_train[e], 2), 'metric', metric_train[e])
                logger.debug('... Val: loss ', round(loss_val[e], 2), 'metric', metric_val[e])
                logger.debug('... Best Epoch', metric_val[e][-1])
                logger.debug('No Weights', print_losses_noweights)
                logger.debug('Weights', print_losses_withweights)
            else:
                logger.debug('...... ', e, ' \n... Train: loss ', round(loss_train[e], 2), 'metric ', metric_train[e],
                             '\n... Val: loss ', round(loss_val[e], 2), 'metric ', metric_val[e])
        # Decay
        if e % params['decay'] == 0:
            opt_scheduler.step()

    # Loading best model.
    if params['best_validation_test'] and best_epoch > 0:
        logger.debug('... Loading Best validation (epoch ', best_epoch, ')')
        model.load_state_dict(torch.load(path))
    else:
        print('... BEST MODEL WAS AT EPOCH 0')

    logger.debug('... Final Metrics - Target')
    data_ = [loader_train, loader_val, loader_test]
    data_name = ['Train', 'Val', 'Test']
    sigmoid = nn.Sigmoid()
    for i, data in enumerate(data_):
        X, y, T = next(iter(data))
        ty_pred, _ = model(X.to(device), T.to(device))
        if params['is_binary_target']:
            y01_pred = sigmoid(ty_pred[:, model.num_treat]).cpu().detach().numpy()
            y01_pred = [1 if item > 0.5 else 0 for item in y01_pred]
        else:
            y01_pred = ty_pred[:, model.num_treat].cpu().detach().numpy()

        try:
            metric = metric_batch(obs=y, pred=y01_pred, is_binary=params['is_binary_target'])
        except ValueError:
            metric = np.nan()

        if params['is_binary_target']:
            logger.debug('......', data_name[i], ': ', round(metric, 3), ' and precision:',
                         precision_score(y01_pred, y))
            if data_name[i] == 'Test':
                f1 = f1_score(y, y01_pred)
        else:
            logger.debug('......', data_name[i], ': ', round(metric, 3), ' and RMSE:', mean_squared_error(y01_pred, y))
            logger.debug(y01_pred[0:6], '\n', y[0:6])
            if data_name[i] == 'Test':
                logger.debug('mean_squared_error')
                f1 = mean_squared_error(y, y01_pred)
                logger.debug('obs ', y[0:10], '\npred ', y01_pred[0:10], f1)
    return model.outcomeY[0:model.num_treat].cpu().detach().numpy().reshape(-1), f1


class M3E2(nn.Module):
    """ Creates the M3E2 model.
    Reference to create a pytorch model: github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py
    """
    def __init__(self, num_treat, num_exp, expert=None, units_exp=4, use_bias_exp=False,
                 use_bias_gate=False, y_continuous=False, dropoutp=0.25, Xlow_cols=None, Xhigh_cols=None,
                 hidden1=None, hidden2=None, use_bias_y=False):
        super().__init__()
        self.num_treat = num_treat
        self.num_exp = num_exp
        self.expert = expert
        self.units_exp = units_exp
        self.units_tower = units_exp
        self.use_bias_exp = use_bias_exp
        self.use_bias_gate = use_bias_gate
        self.use_bias_y = use_bias_y
        self.y_continuous = y_continuous

        if Xhigh_cols is not None:
            self.num_features = hidden2 + len(Xlow_cols)
        else:
            self.num_features = len(Xlow_cols)
        self.Xlow_cols = Xlow_cols
        self.Xhigh_cols = Xhigh_cols

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
        if self.use_bias_y:
            self.bias_y = nn.Parameter(torch.zeros(1), requires_grad=True)

        '''Defining towers - per treatment'''
        self.HT_list = nn.ModuleList([nn.Linear(self.num_exp, self.units_tower) for i in range(self.num_treat)])
        # self.HY = nn.Linear(self.num_exp, self.units_tower)
        self.propensityT = nn.ModuleList([nn.Linear(self.units_tower, 1) for i in range(self.num_treat)])
        self.outcomeY = nn.Parameter(torch.rand(size=(self.num_treat + self.units_tower, 1)))

        '''Autoenconder'''
        if self.Xhigh_cols is not None:
            self.ae = AE(input=len(self.Xhigh_cols), hidden1=hidden1, hidden2=hidden2)
        # propensity score

        '''Defining activation functions and others'''
        self.dropout = nn.Dropout(dropoutp)
        self.tahn = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        logger.debug('... Model initialization done!')

    def forward(self, inputs, treat_assignment=None):

        n = inputs.shape[0]

        '''Dropout'''
        inputs = self.dropout(inputs)

        if self.Xhigh_cols is not None:
            inputsX2 = inputs[:, self.Xhigh_cols]
            inputsX1 = inputs[:, self.Xlow_cols]
            inputsX2, L = self.ae(inputsX2)
            inputs = torch.cat((inputsX1, L), 1)

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
            if i == 0:
                output = out_t.reshape(-1, 1)
            else:
                output = torch.cat((output, out_t.reshape(-1, 1)), 1)

        '''Outcome output - Y'''
        HY = HY / self.num_treat
        HY = torch.reshape(HY, (HY.shape[1], HY.shape[2]))

        if treat_assignment is None:
            # Testing set.
            logger.debug('No Treatment')
            aux = torch.cat((output, HY), 1)
            aux = self.relu(aux)
            out = torch.matmul(aux, self.outcomeY)
            if self.use_bias_y:
                out = out.add(self.bias_y)
            output = torch.cat((output, out.reshape(-1, 1)), 1)
        else:
            # Training set.
            HY = self.relu(HY)
            aux = torch.cat((treat_assignment, HY), 1)
            if self.num_treat > 1:
                aux = self.relu(aux)
            out = torch.matmul(aux, self.outcomeY)
            if self.use_bias_y:
                out = out.add(self.bias_y)
            if self.num_treat > 1:
                out = self.tahn(out)
            output = torch.cat((output, out.reshape(-1, 1)), 1)

        if self.Xhigh_cols is not None:
            return output, inputsX2
        else:
            return output, None


class AE(nn.Module):
    """
    Reference: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
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
        return reconstructed, code
