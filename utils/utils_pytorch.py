"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

Note: Utility functions for GANITE.

(1) xavier_init: Xavier initialization function
(2) batch_generator: generate mini-batch with x, t, and y
"""

# Necessary packages
import torch
import numpy as np


def xavier_init(size):
    """Xavier initialization function.
  
    Args:
      - size: input data dimension
    """
    in_dim = size[0]
    xavier_stddev = 1. / torch.sqrt(in_dim / 2.)
    return torch.empty(size).normal_(std=xavier_stddev)


# Mini-batch generation
def batch_generator(x, t, y, size):
    """ Generate mini-batch with x, t, and y.
  
    Args:
      - x: features
      - t: treatments
      - y: observed labels
      - size: mini batch size

    Returns:
      - X_mb: mini-batch features
      - T_mb: mini-batch treatments
      - Y_mb: mini-batch observed labels
    """
    # print(x.shape)
    batch_idx = np.random.randint(0, x.shape[0], size)

    X_mb = x[batch_idx, :]
    T_mb = np.reshape(t[batch_idx], [size, 1])
    Y_mb = np.reshape(y[batch_idx], [size, 1])
    return X_mb, T_mb, Y_mb


def comb_potential_outcome(yf, ycf, t):
    y0 = yf.reshape(-1,) * (1 - t.reshape(-1,)) + ycf.reshape(-1,) * t.reshape(-1,)
    # print(yf.shape)
    # print((1 - t).shape)
    # print((yf * (1 - t)).shape)
    y1 = yf.reshape(-1,) * t.reshape(-1,) + ycf.reshape(-1,) * (1 - t.reshape(-1,))
    return y0.reshape(-1, 1), y1.reshape(-1, 1)
    #return np.concatenate((y0.reshape(-1, 1), y1.reshape(-1, 1)), 1)
