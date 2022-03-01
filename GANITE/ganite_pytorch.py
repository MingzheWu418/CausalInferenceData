"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

ganite.py

Note: GANITE module.
"""

# Necessary packages
import torch
from torch import nn


class XavierLinear(nn.Module):
    """ Linear layer with Xavier initialization"""
    def __init__(self, input_dim, output_dim):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


# 3. Definitions of generator, discriminator and inference networks
# 3.1 Generator
class Generator(nn.Module):
    """Generator class.

    Args:
      - x: features
      - t: treatments
      - y: observed labels

    Returns:
      - G_logit: estimated potential outcomes
    """

    def __init__(self, dim, h_dim):
        super(Generator, self).__init__()
        self.h1 = XavierLinear(dim + 2, h_dim)
        self.h2 = XavierLinear(h_dim, h_dim)
        self.h31 = XavierLinear(h_dim, h_dim)
        self.h32 = XavierLinear(h_dim, 1)
        self.h41 = XavierLinear(h_dim, h_dim)
        self.h42 = XavierLinear(h_dim, 1)

    def forward(self, x, t, y):
        # Concatenate feature, treatments, and observed labels as input
        inputs = torch.cat((x.float(), t.float(), y.float()), 1)
        G_h1 = nn.functional.relu(self.h1(inputs))
        G_h2 = nn.functional.relu(self.h2(G_h1))

        # Estimated outcome if t = 0
        G_h31 = nn.functional.relu(self.h31(G_h2))
        G_logit1 = self.h32(G_h31)

        # Estimated outcome if t = 1
        G_h41 = nn.functional.relu(self.h41(G_h2))
        G_logit2 = self.h42(G_h41)
        G_logit = torch.cat((G_logit1, G_logit2), 1)
        return G_logit


# 3.2. Discriminator
class Discriminator(nn.Module):
    """Discriminator class."""

    def __init__(self, dim, h_dim):
        super(Discriminator, self).__init__()
        self.h1 = XavierLinear(dim + 2, h_dim)
        self.h2 = XavierLinear(h_dim, h_dim)
        self.h3 = XavierLinear(h_dim, 1)

    def forward(self, x, t, y, hat_y):
        """
        Args:
        - x: features
        - t: treatments
        - y: observed labels
        - hat_y: estimated counterfactuals

        Returns:
        - D_logit: estimated potential outcomes
        """
        ## Concatenate factual & counterfactual outcomes
        x = x.float()
        t = t.float()
        y = y.float()
        hat_y = hat_y.float()
        input0 = (1. - t) * y + t * torch.reshape(hat_y[:, 0], (-1, 1))  # if t = 0
        input1 = t * y + (1. - t) * torch.reshape(hat_y[:, 1], (-1, 1))  # if t = 1

        inputs = torch.cat((x, input0, input1), 1)

        D_h1 = nn.functional.relu(self.h1(inputs))
        D_h2 = nn.functional.relu(self.h2(D_h1))
        D_logit = self.h3(D_h2)
        return D_logit


# 3.3. Inference Nets
class Inference(nn.Module):
    """Inference class."""

    def __init__(self, dim, h_dim):
        super(Inference, self).__init__()
        self.h1 = XavierLinear(dim, h_dim)
        self.h2 = XavierLinear(h_dim, h_dim)
        self.h31 = XavierLinear(h_dim, h_dim)
        self.h32 = XavierLinear(h_dim, 1)
        self.h41 = XavierLinear(h_dim, h_dim)
        self.h42 = XavierLinear(h_dim, 1)

    def forward(self, x):
        """
        Args:
          - x: features

        Returns:
          - I_logit: estimated potential outcomes
        """
        I_h1 = nn.functional.relu(self.h1(x.float()))
        I_h2 = nn.functional.relu(self.h2(I_h1))
        I_h31 = nn.functional.relu(self.h31(I_h2))
        I_logit1 = self.h32(I_h31)

        I_h41 = nn.functional.relu(self.h41(I_h2))
        I_logit2 = self.h42(I_h41)

        I_logit = torch.cat((I_logit1, I_logit2), 1)
        return I_logit