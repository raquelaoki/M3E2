"""Dragonnet.

Reference: https://github.com/claudiashi57/dragonnet
Adapted to multiple-treatments (run multiple independent dragonnets)

"""

import logging
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, mean_squared_error
from tensorflow.keras.optimizers import SGD, Adam

# Local Imports
from models_dragonnet import make_dragonnet
from models_dragonnet import binary_classification_loss, regression_loss, treatment_accuracy, track_epsilon
from models_dragonnet import dragonnet_loss_binarycross, make_tarreg_loss
from semi_parametric_estimation.ate import psi_naive, psi_tmle_cont_outcome

logger = logging.getLogger(__name__)


class dragonnet():
    def __init__(self, X_train, X_test, y_train, y_test,
                 treatments_columns):
        super(dragonnet, self).__init__()
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)
        self.treatments_columns = treatments_columns
        self.covariates_columns = [col for col in list(range(X_train.shape[1])) if col not in treatments_columns]
        self.models = []
        self.simple_ate = []
        self.tmle_ate = []
        self.y_scaler = None
        self.f1_test = None
        self.test_output_list = []
        self.train_output_list = []
        self.all_output_list = []
        #print('Running dragonnet')

    def fit_all(self, is_targeted_regularization, ratio=1., val_split=0.2,
            batch_size=64, epochs_adam=100, epochs_sgd=300,
            print_=True, u1=200, u2=100, u3=1):
        knob_loss = dragonnet_loss_binarycross
        x_train = self.X_train[:, self.covariates_columns]
        x_test = self.X_test[:, self.covariates_columns]
        y_train = self.y_train
        y_test = self.y_test
        f1_test = []

        for t_col in self.treatments_columns:
            #print('Col treat', t_col)
            t_train = self.X_train[:, [t_col]]
            t_test = self.X_test[:, [t_col]]

            test_output, train_output, all_output = self.train_and_predict_dragons(t_train, y_train, x_train,
                                                                                   t_test, y_test, x_test,
                                                                                   targeted_regularization=is_targeted_regularization,
                                                                                   knob_loss=knob_loss, ratio=ratio,
                                                                                   val_split=val_split,
                                                                                   batch_size=batch_size,
                                                                                   epochs_adam=epochs_adam,
                                                                                   epochs_sgd=epochs_sgd,
                                                                                   u1=u1, u2=u2,u3=u3)
            self.test_output_list.append(test_output)
            self.train_output_list.append(train_output)
            self.all_output_list.append(all_output)

            t_train = t_train.reshape(-1)
            y_train_pred = train_output[:, 0] * (t_train == 0) + train_output[:, 1] * (t_train == 1)
            y_train_pred = self.y_scaler.inverse_transform(y_train_pred.reshape(-1,1))

            t_test = t_test.reshape(-1)
            y_test_pred = test_output[:, 0] * (t_test == 0) + test_output[:, 1] * (t_test == 1)
            y_test_pred = self.y_scaler.inverse_transform(y_test_pred.reshape(-1,1))
            if len(np.unique(self.y_train)) ==2:
                thhold = self.Find_Optimal_Cutoff(self.y_train, y_train_pred)
                y_train_pred01 = [0 if item < thhold else 1 for item in y_train_pred]
                y_test_pred01 = [0 if item < thhold else 1 for item in y_test_pred]
                if print_:
                    logger.debug('... Evaluation:')
                    logger.debug('... Training set: F1 - ', f1_score(self.y_train, y_train_pred01))
                    logger.debug('...... confusion matrix: ', confusion_matrix(self.y_train, y_train_pred01).ravel())

                    logger.debug('... Testing set: F1 - ', f1_score(self.y_test, y_test_pred01))
                    logger.debug('...... confusion matrix: ', confusion_matrix(self.y_test, y_test_pred01).ravel())
                f1_test.append(f1_score(self.y_test, y_test_pred01))
            else:
                logger.debug('... Evaluation:')
                logger.debug('... Training set: MSE - ', mean_squared_error(self.y_train, y_train_pred))

                logger.debug('... Testing set: MSE - ', mean_squared_error(self.y_test, y_test_pred))
                f1_test.append(mean_squared_error(self.y_test, y_test_pred))
        self.f1_test = np.mean(f1_test)

    def train_and_predict_dragons(self, t_train, y_train, x_train, t_test, y_test, x_test,
                                  targeted_regularization=True, knob_loss=dragonnet_loss_binarycross,
                                  ratio=1., val_split=0.2, batch_size=64, epochs_adam=100, epochs_sgd=300,
                                  u1=200, u2=100, u3=1):
        # reference: https://github.com/claudiashi57/dragonnet
        verbose = 0  # Don't output log into the standard output stream
        self.y_scaler = StandardScaler().fit(y_train)
        y_train = self.y_scaler.transform(y_train)
        y_test = self.y_scaler.transform(y_test)

        # print("I am here making dragonnet")
        dragonnet = make_dragonnet(input_dim=x_train.shape[1], reg_l2=0.01,
                                   u1 = u1, u2=u2, u3=u3)

        metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

        if targeted_regularization:
            loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss

        i = 0
        # tf.random.set_random_seed(i)'
        tf.random.set_seed(i)
        np.random.seed(i)

        yt_train = np.concatenate([y_train, t_train], 1)

        # import time
        # start_time = time.time()

        dragonnet.compile(
            optimizer=Adam(lr=1e-3),
            loss=loss, metrics=metrics)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)
        ]

        dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                      validation_split=val_split,
                      epochs=epochs_adam,
                      batch_size=batch_size, verbose=verbose)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        sgd_lr = 1e-5
        momentum = 0.9
        dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                          metrics=metrics)
        dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                      validation_split=val_split,
                      epochs=epochs_sgd,
                      batch_size=batch_size, verbose=verbose)

        # elapsed_time = time.time() - start_time
        # print("***************************** elapsed_time is: ", elapsed_time)

        self.models.append(dragonnet)

        yt_hat_test = dragonnet.predict(x_test)
        yt_hat_train = dragonnet.predict(x_train)
        yt_hat_all = dragonnet.predict(np.concatenate([x_train, x_test], axis=0))

        K.clear_session()

        return yt_hat_test, yt_hat_train, yt_hat_all

    def get_estimate(self, q_t0, q_t1, g, t, y_dragon, truncate_level=0.01):
        """
        getting the back door adjustment & TMLE estimation
        """

        psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
        psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                                  y_dragon,
                                                                                                  truncate_level=truncate_level)
        return psi_n, psi_tmle, initial_loss, final_loss, g_loss

    def split_output(self, yt_hat, t, y, y_scaler, x):
        q_t0 = self.y_scaler.inverse_transform(yt_hat[:, 0].copy().reshape(-1,1))
        q_t1 = self.y_scaler.inverse_transform(yt_hat[:, 1].copy().reshape(-1,1))
        g = yt_hat[:, 2].copy()

        if yt_hat.shape[1] == 4:
            eps = yt_hat[:, 3][0]
        else:
            eps = np.zeros_like(yt_hat[:, 2])

        y = y_scaler.inverse_transform(y.copy())
        # var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
        #                                                                    g[t.squeeze() == 0.].mean())
        # print(var)

        # return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}
        return q_t0, q_t1, g, t, y, x, eps

    def ate(self, dataset='all'):
        if dataset == 'train':
            X = self.X_train[:, self.covariates_columns]
            y = self.y_train
        if dataset == 'test':
            X = self.X_test[:, self.covariates_columns]
            y = self.y_test
        if dataset == 'all':
            X = np.concatenate([self.X_train[:, self.covariates_columns], self.X_test[:, self.covariates_columns]],
                               axis=0)
            y = np.concatenate([self.y_train, self.y_test], axis=0)

        y = self.y_scaler.transform(y)

        # print('CAlculating ate')
        for i in range(len(self.treatments_columns)):
            t_col = self.treatments_columns[i]
            if dataset == 'train':
                t = self.X_train[:, [t_col]]
            if dataset == 'test':
                t = self.X_test[:, [t_col]]
            if dataset == 'all':
                t = np.concatenate([self.X_train[:, [t_col]], self.X_test[:, [t_col]]], axis=0)
            yt_hat = self.all_output_list[i]

            q_t0, q_t1, g, t, y_dragon, x, eps = self.split_output(yt_hat, t, y, self.y_scaler, X)

            psi_n, psi_tmle, initial_loss, final_loss, g_loss = self.get_estimate(q_t0, q_t1, g, t, y_dragon,
                                                                                  truncate_level=0.01)
            # print('TCOL',psi_n,q_t0, q_t1)
            self.simple_ate.append(psi_n)
            self.tmle_ate.append(psi_tmle)

        return self.simple_ate, self.tmle_ate

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
        #print('Im HERE')
        #print('Target:', target)
        #print('predicted',predicted)
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t['threshold'])
