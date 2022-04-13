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


class Evaluator:
    def __init__(self):
        pass

    # ITE and ITE prediction
    def PEHE(self, y0, y1, yf_hat, ycf_hat, t):
        """Compute Precision in Estimation of Heterogeneous Effect.

        Args:
          - y: potential outcomes
          - y_hat: estimated potential outcomes

        Returns:
          - PEHE_val: computed PEHE
        """
        # PEHE_val = np.mean(np.abs((y[:, 1] - y[:, 0]) - (y_hat[:, 1] - y_hat[:, 0])))
        eff_pred = ycf_hat - yf_hat
        eff_pred[t > 0] = -eff_pred[t > 0]
        # mu0 = y0 * (1 - t) + y1 * t
        # mu1 = y1 * (1 - t) + y0 * t

        # print("Sanity Check")
        # print(np.mean((y0)), np.std((y0)), y0.shape)
        # print(np.mean((y1)), np.std((y1)), y1.shape)
        # print(np.mean(yf_hat), np.std(yf_hat), yf_hat.shape)
        # print(np.mean(ycf_hat), np.std(ycf_hat), ycf_hat.shape)
        # print(np.mean((mu1)), np.std((mu1)), mu1.shape)
        # print(np.mean((mu0)), np.std((mu0)), mu0.shape)
        # print(np.mean(eff_pred), np.std(eff_pred), eff_pred.shape)
        PEHE_val = np.sqrt(np.mean(np.square((y1 - y0) - eff_pred)))
        # PEHE_val = np.sqrt(np.mean(np.square((y1 - y0) - (y1_hat - y0_hat))))
        return PEHE_val

    def ATE(self, y0, y1, y0_hat, y1_hat):
        """Compute Average Treatment Effect.
        Args:
          - y: potential outcomes
          - y_hat: estimated potential outcomes

        Returns:
          - ATE_val: computed ATE
        """
        ATE_val = np.abs(np.mean(y1 - y0) - np.mean(y1_hat - y0_hat))
        return ATE_val

    def policy_val(self, t, yf, y0_pred, y1_pred):
        """ Computes the value of the policy defined by predicted effect """
        t = t.reshape(-1,)
        y_pred_f = t * y1_pred.reshape(-1,) + (1 - t) * y0_pred.reshape(-1,)
        y_pred_cf = (1 - t) * y1_pred.reshape(-1,) + t * y0_pred.reshape(-1,)
        # print(t.shape)
        # print(y1_pred.shape)
        # print(y_pred_cf.shape)
        # print((t * y1_pred).shape)
        # print(y_pred_f.shape)
        eff_pred = y_pred_f - y_pred_cf
        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0
        treat_overlap = (policy == t) * (t > 0)
        # print(treat_overlap.shape)
        # print(yf.shape)
        control_overlap = (policy == t) * (t < 1)
        # print(eff_pred.shape)
        # print(policy.shape)
        # print(t.shape)
        # print((policy == t).shape)
        # print((t < 1).shape)
        # print(treat_overlap.shape)
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
        # print(policy_value)
        return policy_value
