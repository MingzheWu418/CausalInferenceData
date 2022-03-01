"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_ganite.py

(1) Import data
(2) Train GANITE & Estimate potential outcomes
(3) Evaluate the performances
  - PEHE
  - ATE
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random

import numpy as np
import warnings

import torch


warnings.filterwarnings("ignore")

# 1. GANITE model
# from GANITE.train_pytorch import ganite
from GANITE.ganite import ganite
# 2. data loading
from data.data_loader import load_data, split, cross_val_index
# 3. Metrics
from GANITE.metrics import PEHE, ATE, policy_val
from utils.utils_pytorch import comb_potential_outcome


def main(args):
    """Main function for GANITE experiments.
  
    Args:
        - data_name: twin
        - train_rate: ratio of training data
        - Network parameters (should be optimized for different datasets)
            - h_dim: hidden dimensions
            - iteration: number of training iterations
            - batch_size: the number of samples in each batch
            - alpha: hyper-parameter to adjust the loss importance

    Returns:
        - test_y_hat: estimated potential outcomes
        - metric_results: performance on testing data
    """
    npzfile = args.outdir + 'result'
    npzfile_test = args.outdir + 'result.test'
    ## data loading
    # train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = \
    #     data_loading_twin(args.train_rate)
    if args.data_name == "twin":
        data_path = "../datasets/twins.npz"
    if args.data_name == "ihdp":
        data_path = "../datasets/ihdp_npci_1-100.npz"
    if args.data_name == "jobs":
        data_path = "../datasets/jobs_DW_bin.npz"
    d = load_data(data_path)

    print(args.data_name + ' dataset is ready.')

    # Set network parameters
    parameters = dict()
    parameters['h_dim'] = args.h_dim
    parameters['iteration'] = args.iteration
    parameters['batch_size'] = args.batch_size
    parameters['alpha'] = args.alpha

    # print(d)
    train_dataset, test_dataset = split(d, args.train_rate)

    test_x = test_dataset['x']
    test_yf = test_dataset['yf']
    test_t = test_dataset['t']
    try:
        test_ycf = test_dataset['ycf']
        test_potential_y = comb_potential_outcome(test_yf, test_ycf, test_t)
    except:
        # print("123")
        print("No Counterfactual data Available")

    # Output initialization
    metric_results = dict()
    metric_results['PEHE_TEST'] = []
    metric_results['PEHE_TRAIN'] = []
    metric_results['PEHE_VAL'] = []
    metric_results['ATE_TEST'] = []
    metric_results['ATE_TRAIN'] = []
    metric_results['ATE_VAL'] = []
    metric_results['P_RISK_TEST'] = []
    metric_results['P_RISK_TRAIN'] = []
    metric_results['P_RISK_VAL'] = []

    indexes = cross_val_index(train_dataset, args.folds)
    for fold, I_val in enumerate(indexes):
        I_train = np.setdiff1d(np.arange(train_dataset['t'].shape[0]), I_val)
        train_x = train_dataset['x'][I_train, :]
        train_t = train_dataset['t'][I_train]
        train_yf = train_dataset['yf'][I_train]
        val_x = train_dataset['x'][I_val, :]
        val_t = train_dataset['t'][I_val]
        val_yf = train_dataset['yf'][I_val]

        try:
            train_ycf = train_dataset['ycf'][I_train]
            train_potential_y = comb_potential_outcome(train_yf, train_ycf, train_t)
            val_ycf = train_dataset['ycf'][I_val]
            val_potential_y = comb_potential_outcome(val_yf, val_ycf, val_t)
        except:
            print("No Counterfactual data Available")

        # TODO: incorporate validation in training
        ## Potential outcome estimations by GANITE
        train_y_hat, val_y_hat, test_y_hat = ganite(train_x, train_t, train_yf, val_x, test_x, parameters)
        print('Finish GANITE training and potential outcome estimations')

        ## Performance metrics

        try:
            # 1. PEHE
            test_PEHE = PEHE(test_potential_y, test_y_hat)
            metric_results['PEHE_TEST'].append(np.round(test_PEHE, 4))

            train_PEHE = PEHE(train_potential_y, train_y_hat)
            metric_results['PEHE_TRAIN'].append(np.round(train_PEHE, 4))

            val_PEHE = PEHE(val_potential_y, val_y_hat)
            metric_results['PEHE_VAL'].append(np.round(val_PEHE, 4))
            # 2. ATE
            test_ATE = ATE(test_potential_y, test_y_hat)
            metric_results['ATE_TEST'].append(np.round(test_ATE, 4))

            train_ATE = ATE(train_potential_y, train_y_hat)
            metric_results['ATE_TRAIN'].append(np.round(train_ATE, 4))

            val_ATE = ATE(val_potential_y, val_y_hat)
            metric_results['ATE_VAL'].append(np.round(val_ATE, 4))
        except UnboundLocalError:
            test_p_value, test_policy_curve = policy_val(test_t, test_yf, test_y_hat)
            metric_results['P_RISK_TEST'].append(np.round(1-test_p_value, 4))

            train_p_value, train_policy_curve = policy_val(train_t, train_yf, train_y_hat)
            metric_results['P_RISK_TRAIN'].append(np.round(1-train_p_value, 4))

            val_p_value, val_policy_curve = policy_val(val_t, val_yf, val_y_hat)
            metric_results['P_RISK_VAL'].append(np.round(1-val_p_value, 4))

        ## Print performance metrics on testing data
    print(metric_results)
    for key, item in metric_results.items():
        # print(key, item)
        try:
            # print(key)
            # print(np.mean(item), np.std(item))
            metric_results[key] = (np.round(np.mean(item), 4), np.round(np.std(item),4))
        except:
            pass

        # np.savez(npzfile, pred=train_y_hat, loss=out_losses, val=I_val)
        # np.savez(npzfile_test, pred=test_y_hat)
    print(metric_results)
    return test_y_hat, metric_results


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['twin', 'ihdp', 'jobs'],
        default='ihdp',
        type=str)
    parser.add_argument(
        '--outdir',
        default='./output/',
        type=str)
    parser.add_argument(
        '--folds',
        help='Number of folds for cross-validation',
        default=5,
        type=int)
    parser.add_argument(
        '--train_rate',
        help='the ratio of training data',
        default=0.8,
        type=float)
    parser.add_argument(
        '--h_dim',
        help='hidden state dimensions (should be optimized)',
        default=30,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=10000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=256,
        type=int)
    parser.add_argument(
        '--alpha',
        help='hyper-parameter to adjust the loss importance (should be optimized)',
        default=1,
        type=int)
    parser.add_argument(
        '--seed',
        help='random seed',
        default=1,
        type=int)

    args = parser.parse_args()

    # Calls main function

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_y_hat, metrics = main(args)
