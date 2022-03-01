import torch
from torch import nn
import numpy as np
from SITE.simi_ite.util_pytorch import dynamic_stitch, safe_sqrt


class SiteNet(nn.Module):
    """ The model to be trained """

    def __init__(self, dims, dropout_in, dropout_out, FLAGS):
        super(SiteNet, self).__init__()

        dim_in = dims[1]
        dim_out = dims[2]
        self.FLAGS = FLAGS

        ''' InputLayers returns a representation given the input '''
        self.inLayers = InputLayers(dims, dropout_in, FLAGS)

        ''' Output layers calculates the final prediction given the representation '''
        if self.FLAGS['split_output']:
            self.outLayers1 = OutputLayers(dim_in, dim_out, dropout_out, FLAGS)
            self.outLayers2 = OutputLayers(dim_in, dim_out, dropout_out, FLAGS)
        else:
            self.outLayers = OutputLayers(dim_in + 1, dim_out, dropout_out, FLAGS)

        ''' Layers required to calculate pddm losses '''
        self.pddm1 = pddmTransformation(dims, FLAGS)
        self.pddm2 = pddmTransformation(dims, FLAGS)
        self.pddm3 = pddmTransformation(dims, FLAGS)
        self.pddm4 = pddmTransformation(dims, FLAGS)
        self.pddm5 = pddmTransformation(dims, FLAGS)

    def forward(self, x, t):
        x = torch.FloatTensor(x)
        t = torch.FloatTensor(t)  # Convert to float tensor
        rep = self.inLayers(x)  # Input layers convert input to representation

        # torch.set_printoptions(profile="full")
        # print(rep)
        # torch.set_printoptions(profile="default")
        if self.FLAGS['split_output']:
            i0 = torch.where(t < 1)[0].to(torch.int64)
            i1 = torch.where(t > 0)[0].to(torch.int64)
            rep0 = rep[i0]
            rep1 = rep[i1]
            y0 = self.outLayers1(rep0)
            y1 = self.outLayers2(rep1)
            y = dynamic_stitch(torch.cat((i0, i1)), torch.cat((y0, y1)))
        else:
            h_input = torch.cat((rep, t), 1)
            y = self.outLayers(h_input)

        # Returning the prediction
        return y.view((-1, 1)), rep

    def __calc_pddm(self, h_in_batch, three_pairs_simi):
        """ Calculate pddm losses """
        # print("----")
        # print(h_in_batch)
        h_rep_norm_batch = self.inLayers(h_in_batch)
        # print(h_rep_norm_batch)
        '''PDDM unit'''
        x_i = h_rep_norm_batch[0:1]
        x_j = h_rep_norm_batch[1:2]
        x_k = h_rep_norm_batch[2:3]
        x_l = h_rep_norm_batch[3:4]
        x_m = h_rep_norm_batch[4:5]
        x_n = h_rep_norm_batch[5:6]

        s_kl = self.pddm1(x_k, x_l)
        s_mn = self.pddm2(x_m, x_n)
        s_km = self.pddm3(x_k, x_m)
        s_ik = self.pddm4(x_i, x_k)
        s_jm = self.pddm5(x_j, x_m)

        simi_kl = three_pairs_simi[0:1, 0:1]
        simi_mn = three_pairs_simi[1:2, 0:1]
        simi_km = three_pairs_simi[2:3, 0:1]
        simi_ik = three_pairs_simi[3:4, 0:1]
        simi_jm = three_pairs_simi[4:5, 0:1]
        # print("----")
        # print(x_i)
        # print(s_kl)
        # print(simi_kl)
        '''pddm loss'''
        pddm_loss = torch.sum(torch.square(simi_kl - s_kl) + torch.square(simi_mn - s_mn)
                              + torch.square(simi_km - s_km) + torch.square(simi_ik - s_ik)
                              + torch.square(simi_jm - s_jm))
        return pddm_loss, x_i, x_j, x_k, x_l, x_m, x_n

    def pddm_mid_loss(self, h_in_batch, three_pairs_simi):
        """ A wrapper that calculates pddm loss and mid point loss at the same time """
        self.pddm_loss, x_i, x_j, x_k, x_l, x_m, x_n = self.__calc_pddm(h_in_batch,
                                                                        three_pairs_simi)
        # print(x_j)
        mid_jk = (x_j + x_k) / 2.0
        mid_im = (x_i + x_m) / 2.0
        '''mid_point distance minimization'''
        self.mid_distance = torch.sum(torch.square(mid_jk - mid_im), dim=(0, 1))
        # print(self.mid_distance)
        return self.pddm_loss, self.mid_distance


class RescalingLayer(nn.Module):
    """
    A custom layer for rescaling
    """

    def __init__(self, size_in, size_out):
        super(RescalingLayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = 1.0 / size_in * torch.ones([size_in])
        self.weights = nn.Parameter(weights)
        self.bias.requires_grad = False

    def forward(self, x):
        return torch.mm(x, self.weights.t())


class MyLinearLayer(nn.Module):
    """
    A custom layer with specified weights and bias
    """

    def __init__(self, size_in, size_out, weight_init):
        super(MyLinearLayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)
        bias = torch.zeros(1, size_out)
        self.bias = nn.Parameter(bias)
        nn.init.normal_(self.weights, std=weight_init / np.sqrt(size_in))

    def forward(self, x):
        w_mul_x = torch.mm(x, self.weights.t())
        return torch.add(w_mul_x, self.bias)


class InputLayers(nn.Module):
    """ Returns a representation for the given input """
    def __init__(self, dims, dropout_in, FLAGS):
        self.FLAGS = FLAGS
        super(InputLayers, self).__init__()
        if FLAGS['nonlin'].lower() == 'elu':
            nonlin = torch.nn.ELU()
        else:
            nonlin = torch.nn.ReLU()
        # self.num_layers = num_layers

        dim_input = dims[0]
        dim_in = dims[1]

        if FLAGS['varsel']:
            ''' If using variable selection, first layer is just rescaling'''
            self.linear0 = RescalingLayer(dim_input, dim_in)
        else:
            self.linear0 = MyLinearLayer(dim_input, dim_in, FLAGS['weight_init'])
            # self.linear0 = nn.Linear(dim_input, dim_in)
        self.nonlin0 = nonlin
        self.dropout0 = nn.Dropout(dropout_in)
        self.layers = nn.ModuleList()
        for i in range(FLAGS['n_in'] - 1):
            self.layers.append(MyLinearLayer(dim_in, dim_in, FLAGS['weight_init']))
            # self.layers.append(nn.Linear(dim_in, dim_in))
            if FLAGS['batch_norm']:
                if FLAGS['normalization'] == 'bn_fixed':
                    self.layers.append(nn.BatchNorm1d(dim_in, eps=1e-3, affine=False))
                else:
                    self.layers.append(nn.BatchNorm1d(dim_in, eps=1e-3))
            self.layers.append(nonlin)
            self.layers.append(nn.Dropout(dropout_in))

    def forward(self, x):
        x = x.float()
        # print(x)
        x = self.linear0(x)
        x = self.nonlin0(x)
        x = self.dropout0(x)
        # print("------")
        # print(self.linear0.weights)
        # print(self.linear0.bias)
        for layer in self.layers:
            x = layer(x)

        # print("----")
        # print(x)
        # torch.set_printoptions(profile="full")
        # print(rep)
        # torch.set_printoptions(profile="default")

        ''' Normalization '''
        if self.FLAGS['normalization'] == 'divide':
            y = safe_sqrt(torch.sum(torch.square(x), dim=1, keepdim=True))
            x = x / y
        else:
            x = 1.0 * x
        return x


class OutputLayers(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_out, FLAGS):
        super(OutputLayers, self).__init__()

        # Initialization
        self.FLAGS = FLAGS
        self.weights_out = []
        dims = [dim_in] + ([dim_out] * FLAGS['n_out'])
        if FLAGS['nonlin'].lower() == 'elu':
            nonlin = torch.nn.ELU()
        else:
            nonlin = torch.nn.ReLU()

        # Add a number of layers based on the parameter FLAGS.n_out
        self.layers = nn.ModuleList()
        for i in range(FLAGS['n_out']):
            my_layer = MyLinearLayer(dims[i], dims[i + 1], FLAGS['weight_init'])
            # my_layer = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(my_layer)
            self.layers.append(nonlin)
            self.layers.append(nn.Dropout(dropout_out))

        self.out = MyLinearLayer(dim_out, 1, FLAGS['weight_init'])
        # self.out = nn.Linear(dim_out, 1)
        # self.weights_out = torch.FloatTensor(self.weights_out)

    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            # print(layer)
            x = layer(x)
        x = self.out(x)
        return x


class pddmTransformation(nn.Module):
    """ Calculate pddm"""
    def __init__(self, dims, FLAGS):
        super(pddmTransformation, self).__init__()
        if FLAGS['nonlin'].lower() == 'elu':
            nonlin = torch.nn.ELU()
        else:
            nonlin = torch.nn.ReLU()
        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        dim_pddm = dims[3]
        dim_c = dims[4]
        dim_s = dims[5]

        self.concate_dim = dim_c + dim_c
        self.dim_c = dim_c
        self.linear_u = MyLinearLayer(dim_in, dim_pddm, FLAGS['weight_init'])
        # self.linear_u = nn.Linear(dim_in, dim_pddm)
        self.nonlin_u = nonlin
        self.linear_v = MyLinearLayer(dim_in, dim_pddm, FLAGS['weight_init'])
        # self.linear_v = nn.Linear(dim_in, dim_pddm)
        self.nonlin_v = nonlin
        self.linear1 = MyLinearLayer(self.concate_dim, dim_c, FLAGS['weight_init'])
        # self.linear1 = nn.Linear(self.concate_dim, dim_c)
        self.nonlin1 = nonlin
        self.linear2 = MyLinearLayer(dim_c, dim_s, FLAGS['weight_init'])
        # self.linear2 = nn.Linear(dim_c, dim_s)

    def forward(self, x_i, x_j):
        x_i = x_i.float()
        x_j = x_j.float()

        u = torch.abs(x_i - x_j)
        v = (x_i + x_j) / 2.0
        u = self.linear_u(u)
        u = self.nonlin_u(u)

        u = nn.functional.normalize(u, p=2, dim=0)
        v = self.linear_v(v)
        v = self.nonlin_v(v)
        v = nn.functional.normalize(v, p=2, dim=0)
        c = torch.cat((u, v), 1)
        c = c.reshape((1, self.concate_dim))
        c = self.linear1(c)
        c = self.nonlin1(c)
        c = c.reshape((1, self.dim_c))
        s = self.linear2(c)

        return s
