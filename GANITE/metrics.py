"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

metrics.py

Note: Metric functions for GANITE.
Reference: Jennifer L Hill, "Bayesian nonparametric modeling for causal inference", Journal of Computational and Graphical Statistics, 2011.

(1) PEHE: Precision in Estimation of Heterogeneous Effect
(2) ATE: Average Treatment Effect
"""

# Necessary packages
import numpy as np


# ITE and ITE prediction
def PEHE(y0, y1, y0_hat, y1_hat):
    """Compute Precision in Estimation of Heterogeneous Effect.
  
    Args:
      - y: potential outcomes
      - y_hat: estimated potential outcomes

    Returns:
      - PEHE_val: computed PEHE
    """
    # PEHE_val = np.mean(np.abs((y[:, 1] - y[:, 0]) - (y_hat[:, 1] - y_hat[:, 0])))
    PEHE_val = np.sqrt(np.mean(np.square((y1 - y0) - (y1_hat - y0_hat))))
    return PEHE_val


def ATE(y0, y1, y0_hat, y1_hat):
    """Compute Average Treatment Effect.
    Args:
      - y: potential outcomes
      - y_hat: estimated potential outcomes

    Returns:
      - ATE_val: computed ATE
    """
    ATE_val = np.abs(np.mean(y1 - y0) - np.mean(y1_hat - y0_hat))
    return ATE_val


def policy_val(t, yf, y0_pred, y1_pred):
    """ Computes the value of the policy defined by predicted effect """

    y_pred_f = t * y1_pred + (1 - t) * y0_pred
    y_pred_cf = (1 - t) * y1_pred + t * y0_pred
    eff_pred = y_pred_f - y_pred_cf
    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred > 0
    treat_overlap = (policy == t) * (t > 0)
    control_overlap = (policy == t) * (t < 1)

    if np.sum(treat_overlap) == 0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap) == 0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit * treat_value + (1 - pit) * control_value

    # 1- policy_value is policy risk
    return policy_value
