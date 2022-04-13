
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ihdp_data import *
import tensorflow as tf
import numpy as np

# random normal initilization on dense layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        nn.init.zeros_(m.bias)


def init_weights_t(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class Gragonnet(nn.Module):

        def __init__(self, input_dim):
            super(Gragonnet, self).__init__()
            self.flatten = nn.Flatten()
            self.first_time = True
            self.linear_ELU_stack = nn.Sequential(
                

                nn.Linear(input_dim, 200),
                nn.ELU(),
                nn.Linear(200, 200),
                nn.ELU(),
                nn.Linear(200, 200),
                nn.ELU(),

            )
            self.t_prediction = nn.Sequential(
                nn.Linear(200, 1),
                nn.Sigmoid(),
            )

            # HYPOTHESIS y0
            self.y_hypothesis = nn.Sequential(
                nn.Linear(200, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
                nn.Linear(100, 1),
            )

            # HYPOTHESIS y1
            self.y_hypothesis1 = nn.Sequential(
                nn.Linear(200, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
                nn.Linear(100, 1),
            )

            #epsilon
            self.epsilon = nn.Sequential(
                nn.Linear(1,1),
            )


        def forward(self, X):
            X = self.flatten(X)
            x = self.linear_ELU_stack

            epsilon = self.epsilon
            t_predictions = self.t_prediction

            if self.first_time:
                x.apply(init_weights)
                epsilon.apply(init_weights)
                t_predictions.apply(init_weights_t)
                self.first_time = False

            x = x(X)

            t_predictions = t_predictions(x)
            y0_prediction = self.y_hypothesis(x)
            y1_prediction = self.y_hypothesis1(x)
            epsilon = epsilon(torch.ones_like(torch.empty(1,1)))
            epsilon = torch.ones_like(y1_prediction) * epsilon

            return torch.cat((y1_prediction, y0_prediction, t_predictions, epsilon), 1)




def dragonnet_loss_binarycross(concat_true, concat_pred):

    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def regression_loss(concat_true, concat_pred):

    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))

    return loss0 + loss1

def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = nn.functional.binary_cross_entropy(t_pred,t_true, reduction='sum')

    return losst

def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred > 0.5).float()

    return (t_true == t_pred).float().sum()


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)
        
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]

        t_pred = (t_pred + 0.01) / 1.02

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))
        loss = vanilla_loss + ratio * targeted_regularization

        return loss

    return tarreg_ATE_unbounded_domain_loss