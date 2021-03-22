import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class M3E2(nn.Module):
    # https://github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py
    def __init__(self,data, num_treat, num_exp, expert = None, units_exp = 4,
    use_bias_exp = False, use_bias_gate = False, use_autoencoder = False, y_continuous = False):
        super(M3E2, self).__init__()
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

    '''Defining experts - number is defined by the user'''
    if self.expert is None:
        self.expert_kernels = nn.ParameterList(
                [
                    nn.Parameter(torch.rand(size=(self.num_features, self.units_exp)).float())
                    for i in range(self.num_exp)
                ]
            )
            self.expert_output = nn.ModuleList(
                [nn.Linear(self.units_exp, 1).float() for i in range(self.num_treat*self.num_exp)]
            )
    '''Defining gates - one per treatment'''
    gate_kernels = torch.rand(self.num_treatments, self.num_features,self.num_exp)
    self.gate_kernels = nn.Parameter(gate_kernels,requires_grad = True)

    '''Defining biases - treatments, gates and experts'''
    if self.use_bias_exp:
        self.bias_exp = nn.Parameter(torch.zeros(self.num_exp), requires_grad=True)
    if self.use_bias_gate:
        self.bias_gate = nn.Parameter(torch.zeros(self.num_treat, 1, self.num_exp), requires_grad=True)
    self.bias_treat = nn.Parameter(torch.zeros(self.num_treat), requires_grad=True)
    self.bias_y = nn.Parameter(torch.zeros(1),requires_grad=True)

    '''Defining towers - per treatment'''
    self.HT_list = nn.ModuleList([nn.Linear(self.num_exp, self.units_tower) for i in range(self.num_treat)])
    #self.HY = nn.Linear(self.num_exp, self.units_tower)
    self.propensityT = nn.ModuleList([nn.Linear(self.units_tower, 1) for i in range(self.num_treat)])
    self.outcomeY = nn.Parameter(torch.rand(size = (self.num_treat+self.units_tower)))

    '''TODO: Defining'''
    #autoencoder
    if self.use_autoencoder:
        print('missing')
    #propensity score

    '''Defining activation functions and others'''
    self.dropout = nn.Dropout(0.25)
    self.tahn = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

    def foward(self,inputs, treat_assignment = None ):
        n = inputs.shape[0]
        #MISSING AUTOENCODER

        ''' Calculating Experts'''
        for i in range(self.num_exp):
            aux = torch.mm(inputs,self.expert_kernels[i]).reshape((n, self.expert_kernels[i].shape[1]))
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
            if if == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[i]).reshape(1, n, self.num_exp)
            else:
                gate_outputs = torch.cat(
                    (gate_outputs,torch.mm(inputs, self.gate_kernels[index]).reshape(1, n, self.num_exp)),
                    dim=0,
                )

        if self.use_bias_gate:
            gate_outputs = gate_outputs.add(self.bias_gate)

        gate_outputs = F.softmax(gate_outputs, dim = 2)
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
        HY = HY/self.num_treat
        if treat_assignment is not None:
            print('TODO')
        else:
            print('Checking shapes: ', output.shape, HY.shape)
            aux = np.concatenate([output,HY],axis = 1)
            out = torch.mul(aux, self.outcomeY)
            out = out.add(self.bias_y)
            if y_continuous:
                out = self.tahn(out)
            else:
                out = self.sigmoid(out)
            output.append(out)

        return output
