import random

import numpy as np
import torch
import argparse

from GANITE.metrics import PEHE, ATE, policy_val
from GANITE.train_pytorch import ganite, ganite_predict
from SITE.site_net_train_pytorch import site, site_predict
from utils.utils_pytorch import comb_potential_outcome
from data.data_loader import split, cross_val_index, load_data
import yaml

def main(args):
    npzfile = args.outdir + 'result'
    npzfile_test = args.outdir + 'result.test'
    ## data loading
    # train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = \
    #     data_loading_twin(args.train_rate)
    if args.data_name == "twins":
        data_path = "./datasets/twins.npz"
    if args.data_name == "ihdp":
        data_path = "./datasets/ihdp_npci_1-100.npz"
    if args.data_name == "jobs":
        data_path = "./datasets/jobs_DW_bin.npz"
    d = load_data(data_path)
    # d['x'], yf, t, ycf ...

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

    print(args.data_name + ' dataset is ready.')

    # print(d)
    train_dataset, test_dataset = split(d, args.train_rate) # 0.8 train 0.2 test

    test_x = test_dataset['x']
    test_yf = test_dataset['yf']
    test_t = test_dataset['t']
    try:
        test_ycf = test_dataset['ycf']
        test_y0, test_y1 = comb_potential_outcome(test_yf, test_ycf, test_t)
    except:
        # print("123")
        print("No Counterfactual data Available")

    indexes = cross_val_index(train_dataset, args.folds)
    # [[1,2,3,4,74, ...],[],[],[],[]]

    with open("./configs/" + args.model + ".yaml", "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parameters["propensity_dir"] = './SITE/propensity_score/' + args.data_name + '_propensity_model.sav'
    # print(parameters)

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
            train_y0, train_y1 = comb_potential_outcome(train_yf, train_ycf, train_t)
            val_ycf = train_dataset['ycf'][I_val]
            val_y0, val_y1 = comb_potential_outcome(val_yf, val_ycf, val_t)
        except:
            print("No Counterfactual data Available")

        if args.model == "ganite":
            # print(train_x)
            model = ganite(train_x, train_t, train_yf, parameters)
            train_y0_pred, train_y1_pred = ganite_predict(model, train_x)
            test_y0_pred, test_y1_pred = ganite_predict(model, test_x)
            val_y0_pred, val_y1_pred = ganite_predict(model, val_x)
            # either yf ycf. outcome of shape n*1
            print('Finish GANITE training and potential outcome estimations')
        elif args.model == "site":
            model = site(train_x, train_t, train_yf, d['dim'], parameters)
            # site_predict(model, x, t)
            train_y0_pred, train_y1_pred = site_predict(model, train_x, train_t)
            test_y0_pred, test_y1_pred = site_predict(model, test_x, test_t)
            val_y0_pred, val_y1_pred = site_predict(model, val_x, val_t)
            # print(model)
            print('Finish SITE training and potential outcome estimations')

            ## Performance metrics

            # class, evaluator
        try:
            # 1. PEHE
            test_PEHE = PEHE(test_y0, test_y1, test_y0_pred, test_y1_pred)
            metric_results['PEHE_TEST'].append(np.round(test_PEHE, 4))

            train_PEHE = PEHE(train_y0, train_y1, train_y0_pred, train_y1_pred)
            metric_results['PEHE_TRAIN'].append(np.round(train_PEHE, 4))

            val_PEHE = PEHE(val_y0, val_y1, val_y0_pred, val_y1_pred)
            metric_results['PEHE_VAL'].append(np.round(val_PEHE, 4))
            # 2. ATE
            test_ATE = ATE(test_y0, test_y1, test_y0_pred, test_y1_pred)
            metric_results['ATE_TEST'].append(np.round(test_ATE, 4))

            train_ATE = ATE(train_y0, train_y1, train_y0_pred, train_y1_pred)
            metric_results['ATE_TRAIN'].append(np.round(train_ATE, 4))

            val_ATE = ATE(val_y0, val_y1, val_y0_pred, val_y1_pred)
            metric_results['ATE_VAL'].append(np.round(val_ATE, 4))
        except UnboundLocalError:
            test_p_value, test_policy_curve = policy_val(test_t, test_yf, test_y0_pred, test_y1_pred)
            metric_results['P_RISK_TEST'].append(np.round(1-test_p_value, 4))

            train_p_value, train_policy_curve = policy_val(train_t, train_yf, train_y0_pred, train_y1_pred)
            metric_results['P_RISK_TRAIN'].append(np.round(1-train_p_value, 4))

            val_p_value, val_policy_curve = policy_val(val_t, val_yf, val_y0_pred, val_y1_pred)
            metric_results['P_RISK_VAL'].append(np.round(1-val_p_value, 4))

        print(metric_results)
    for key, item in metric_results.items():
        try:
            metric_results[key] = (np.round(np.mean(item), 4), np.round(np.std(item),4))
        except:
            pass
    print(metric_results)

    # # Set network parameters
    # parameters = dict()
    # parameters['h_dim'] = args.h_dim
    # parameters['iteration'] = args.iteration
    # parameters['batch_size'] = args.batch_size
    # parameters['alpha'] = args.alpha


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['ganite', 'site'],
        default='ganite',
        type=str
    )
    parser.add_argument(
        '--data_name',
        choices=['twins', 'ihdp', 'jobs'],
        default='twins',
        type=str)
    parser.add_argument(
        '--outdir',
        default='./output/',
        type=str)
    parser.add_argument(
        '--folds',
        help='Number of folds for cross-validation',
        default=10,
        type=int)
    parser.add_argument(
        '--train_rate',
        help='the ratio of training data',
        default=0.8,
        type=float)
    # parser.add_argument(
    #     '--h_dim',
    #     help='hidden state dimensions (should be optimized)',
    #     default=30,
    #     type=int)
    # parser.add_argument(
    #     '--iteration',
    #     help='Training iterations (should be optimized)',
    #     default=10000,
    #     type=int)
    # parser.add_argument(
    #     '--batch_size',
    #     help='the number of samples in mini-batch (should be optimized)',
    #     default=256,
    #     type=int)
    # parser.add_argument(
    #     '--alpha',
    #     help='hyper-parameter to adjust the loss importance (should be optimized)',
    #     default=1,
    #     type=int)
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
    main(args)