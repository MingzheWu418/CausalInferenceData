from torch import nn
import torch
import numpy as np

SQRT_CONST = 1e-3


def l2_loss(x):
    return torch.sum(torch.square(x)) / 2

class lossCalc:
    def __init__(self, dims, r_alpha, r_lambda, FLAGS):
        self.r_lambda = r_lambda
        self.r_alpha = r_alpha
        self.FLAGS = FLAGS
        self.dim_input = dims[0]
        self.dim_in = dims[1]
        self.dim_out = dims[2]

        self.tot_loss = 0
        self.pred_loss = 0
        self.imb_loss = 0

    def calc_loss(self, t, p, y_, y, h_rep, weights_pred, weights_in, weights_out):
        y = torch.Tensor(y)
        t = torch.Tensor(t)

        sq_error = torch.mean(torch.square(y_ - y))
        pred_error = torch.sqrt(sq_error)
        wd_loss = 0
        sig = self.FLAGS['rbf_sigma']

        # ''' Compute sample reweighting '''
        # if self.FLAGS['reweight_sample']:
        #     w_t = t / (2 * p_t)
        #     w_c = (1 - t) / (2 * (1 - p_t))
        #     sample_weight = w_t + w_c
        # else:
        #     sample_weight = 1.0
        # sample_weight = torch.from_numpy(sample_weight).float()
        # print(sample_weight)
        # self.sample_weight = sample_weight.float()

        if self.FLAGS['loss'] == 'l1':
            risk = torch.mean(torch.abs(y_ - y))

        elif self.FLAGS['loss'] == 'log':
            y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025
            risk = -torch.mean(y_ * torch.log(y) + (1.0 - y_) * torch.log(1.0 - y))
            pred_error = risk
        else:
            risk = sq_error

        ''' Regularization '''
        if self.FLAGS['p_lambda'] > 0:
            if self.FLAGS['varsel'] or self.FLAGS['n_out'] == 0:
                regularization = l2_loss(
                    weights_pred[:self.dim_out - 1, :1])  # don't penalize treatment coefficient
            else:
                regularization = l2_loss(weights_pred)

            for i in range(0, self.FLAGS['n_out']):
                regularization = regularization + l2_loss(weights_out[i])

            for i in range(0, self.FLAGS['n_in']):
                if not (self.FLAGS['varsel'] and i == 0):  # No penalty on W in variable selection
                    regularization = regularization + l2_loss(weights_in[i])

        ''' Imbalance error '''
        if self.FLAGS['imb_fun'] == 'mmd2_rbf':
            imb_error = self.r_alpha * mmd2_rbf(h_rep, t, p, sig)
        elif self.FLAGS['imb_fun'] == 'mmd2_lin':
            imb_error = self.r_alpha * mmd2_lin(h_rep, t, p)
        elif self.FLAGS['imb_fun'] == 'mmd_rbf':
            imb_error = torch.sqrt(SQRT_CONST + torch.square(self.r_alpha) * torch.abs(mmd2_rbf(h_rep, t, p, sig)))
        elif self.FLAGS['imb_fun'] == 'mmd_lin':
            imb_error = torch.sqrt(SQRT_CONST + torch.square(self.r_alpha) * mmd2_lin(h_rep, t, p))
        elif self.FLAGS['imb_fun'] == 'wass':
            imb_error = self.r_alpha * wasserstein(h_rep, t, p, lam=self.FLAGS['wass_lambda'],
                                                   its=self.FLAGS['wass_iterations'], sq=False,
                                                   backpropT=self.FLAGS['wass_bpt'])
        elif self.FLAGS['imb_fun'] == 'wass2':
            imb_error = self.r_alpha * wasserstein(h_rep, t, p, lam=self.FLAGS['wass_lambda'],
                                                   its=self.FLAGS['wass_iterations'], sq=True,
                                                   backpropT=self.FLAGS['wass_bpt'])
        else:
            imb_error = self.r_alpha * lindisc(h_rep, p, t)

        tot_error = risk

        if self.FLAGS['p_alpha'] > 0:
            tot_error = tot_error + imb_error

        if self.FLAGS['p_lambda'] > 0:
            tot_error = tot_error + self.r_lambda * regularization

        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.pred_loss = pred_error
        return tot_error


def lindisc(X, p, t):
    ''' Linear MMD '''

    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.mean(Xc, dim=0)
    mean_treated = torch.mean(Xt, dim=0)

    c = torch.square(2 * p - 1) * 0.25
    f = torch.sign(p - 0.5)

    mmd = torch.sum(torch.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + torch.sqrt(c + mmd + SQRT_CONST)

    return mmd


def mmd2_lin(X, t, p):
    ''' Linear MMD '''

    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.mean(Xc, dim=0)
    mean_treated = torch.mean(Xt, dim=0)

    mmd = torch.sum(torch.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))

    return mmd


def mmd2_rbf(X, t, p, sig):
    """ Computes the l2-RBF MMD for X given t """

    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]

    Xc = X[ic]
    Xt = X[it]

    Kcc = torch.exp(-pdist2sq(Xc, Xc) / torch.square(sig))
    Kct = torch.exp(-pdist2sq(Xc, Xt) / torch.square(sig))
    Ktt = torch.exp(-pdist2sq(Xt, Xt) / torch.square(sig))

    m = Xc.shape[0].type(torch.FloatTensor)
    n = Xt.shape[0].type(torch.FloatTensor)

    mmd = torch.square(1.0 - p) / (m * (m - 1.0)) * (torch.sum(Kcc) - m)
    mmd = mmd + torch.square(p) / (n * (n - 1.0)) * (torch.sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * torch.sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * torch.mm(X, torch.t(Y))
    nx = torch.sum(torch.square(X), 1, True)
    ny = torch.sum(torch.square(Y), 1, True)
    D = (C + torch.t(ny)) + nx
    return D


def pdist2(X, Y):
    """ Returns the tensorflow pairwise distance matrix """
    return torch.sqrt(SQRT_CONST + pdist2sq(X, Y))


def pop_dist(X, t):
    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = Xc.shape[0].type(torch.FloatTensor)
    nt = Xt.shape[0].type(torch.FloatTensor)

    ''' Compute distance matrix'''
    M = pdist2(Xt, Xc)
    return M


def wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """
    # print(t)
    # print(torch.where(t > 0))
    it = torch.where(t > 0)[0]
    # print(it)
    ic = torch.where(t < 1)[0]
    Xc = X[ic]
    # print(Xc.shape)
    Xt = X[it]
    nc = Xc.shape[0]
    nt = Xt.shape[0]

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = torch.sqrt(1e-2 + pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = torch.mean(M)
    # M_drop = tf.nn.dropout(M, 10 / (nc * nt))
    delta = torch.max(M).detach()
    eff_lam = (lam / M_mean).detach()

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    # print(M[0:1, :].shape)
    # print(torch.ones(M[0:1, :].shape).shape)
    # print(torch.zeros(1, 1).shape)
    col = torch.cat([delta * torch.ones(M[:,0:1].shape), torch.zeros(1, 1)], 0)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    ''' Compute marginal vectors '''
    # print(torch.where(t > 0))
    # print(p * torch.ones(torch.where(t > 0)[0].shape) / nt)
    a = torch.cat([p * torch.ones(torch.where(t > 0)[0].reshape(-1,1).shape) / nt, (1 - p) * torch.ones(1, 1)], 0)
    b = torch.cat([(1 - p) * torch.ones(torch.where(t < 1)[0].reshape(-1,1).shape) / nc, p * torch.ones(1, 1)], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam)
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (torch.mm(ainvK, (b / torch.t(torch.mm(torch.t(u), K)))))
    v = b / (torch.t(torch.mm(torch.t(u), K)))

    T = u * (torch.t(v) * K)

    if not backpropT:
        T = T.detach()

    D = 2 * torch.sum(T * Mt)

    return D


def simplex_project(x, k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x, axis=0)[::-1]
    nu = (np.cumsum(mu) - k) / range(1, d + 1)
    I = [i for i in range(0, d) if mu[i] > nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x - theta, 0)
    return w
