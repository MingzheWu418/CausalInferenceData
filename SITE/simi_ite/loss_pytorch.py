from torch import nn
import torch
from .site_net_pytorch import pddmTransformation


class lossCalc:
    def __init__(self, r_lambda, r_mid_point_mini, r_pddm, FLAGS):
        self.r_lambda = r_lambda
        self.r_mid_point_mini = r_mid_point_mini
        self.r_pddm = r_pddm
        self.FLAGS = FLAGS
        self.mid_distance = 0
        self.pddm_loss = 0
        self.tot_loss = 0
        self.pred_loss = 0
        # self.x_i, self.x_j, self.x_k, self.x_l, self.x_m, self.x_n = 0, 0, 0, 0, 0, 0

    def calc_loss(self, t, p_t, y_, y, pddm_loss, mid_loss):
        # print(y)
        y = torch.Tensor(y)
        wd_loss = 0

        ''' Compute sample reweighting '''
        if self.FLAGS['reweight_sample']:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * (1 - p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0
        sample_weight = torch.from_numpy(sample_weight).float()
        # print(sample_weight)
        # self.sample_weight = sample_weight.float()

        if self.FLAGS['loss'] == 'l1':
            risk = torch.mean(sample_weight * torch.abs(y_ - y))
            # TODO: what is this "res"
            pred_error = -torch.mean(res)

        elif self.FLAGS['loss'] == 'log':
            y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025
            res = y_ * torch.log(y) + (1.0 - y_) * torch.log(1.0 - y)
            risk = -torch.mean(sample_weight * res)
            pred_error = -torch.mean(res)
        else:
            risk = torch.mean(sample_weight * torch.square(y_ - y))
            # print risk
            # print sample_weight
            pred_error = torch.sqrt(torch.mean(torch.square(y_ - y)))
            # print(y)

        self.pred_loss = pred_error

        ''' Calculate the total error '''
        tot_error = risk
        if self.FLAGS['p_lambda'] > 0:
            tot_error = tot_error + self.r_lambda * wd_loss

        if self.FLAGS['p_mid_point_mini'] > 0:
            tot_error = tot_error + self.r_mid_point_mini * mid_loss

        if self.FLAGS['p_pddm'] > 0:
            tot_error = tot_error + self.r_pddm * pddm_loss
        self.tot_loss = tot_error

        return tot_error

