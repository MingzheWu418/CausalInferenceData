import random

import numpy as np
import torch
import argparse

from GANITE.metrics import PEHE, ATE, policy_val
from GANITE.train_pytorch import ganite, ganite_predict
from SITE.site_net_train_pytorch import site, site_predict
from eval.evaluator import Evaluator
from utils.utils_pytorch import comb_potential_outcome
from data.data_loader import split, cross_val_index, load_data, load_batch
from bartpy.sklearnmodel import SklearnModel
from bartpy.extensions.baseestimator import ResidualBART
from econml.grf import CausalForest, CausalIVForest, MultiOutputGRF
from econml.dml import CausalForestDML

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
    print(d)
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
    train_dataset, test_dataset = split(d, args.train_rate)  # 0.8 train 0.2 test

    if args.data_name == "ihdp": # Currently only loading ihdp into batchs because it is too large
        train_dataset = load_batch(train_dataset, args.batch_size)

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
    # print

    # print(parameters)

    for fold, I_val in enumerate(indexes):
        if args.data_name == 'ihdp':
            batch = np.random.randint(low=0, high=args.batch_size, size=1)
            I_train = np.setdiff1d(np.arange(train_dataset['t'].shape[0]), I_val)
            train_x = train_dataset['x'][I_train, :, batch]
            # print(train_x.shape)
            train_t = train_dataset['t'][I_train, batch]
            train_yf = train_dataset['yf'][I_train, batch]
            val_x = train_dataset['x'][I_val, :, batch]
            val_t = train_dataset['t'][I_val, batch]
            val_yf = train_dataset['yf'][I_val, batch]

            train_ycf = train_dataset['ycf'][I_train, batch]
            train_y0, train_y1 = comb_potential_outcome(train_yf, train_ycf, train_t)
            val_ycf = train_dataset['ycf'][I_val, batch]
            val_y0, val_y1 = comb_potential_outcome(val_yf, val_ycf, val_t)
        else:
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
            parameters["propensity_dir"] = './SITE/propensity_score/' + args.data_name + '_propensity_model.sav'
            model = site(train_x, train_t, train_yf, d['dim'], parameters)
            # site_predict(model, x, t)
            train_y0_pred, train_y1_pred = site_predict(model, train_x, train_t)
            test_y0_pred, test_y1_pred = site_predict(model, test_x, test_t)
            val_y0_pred, val_y1_pred = site_predict(model, val_x, val_t)
            # print(model)
            print('Finish SITE training and potential outcome estimations')
        elif args.model == "bart":
            # model = SklearnModel()  # Use default parameters
            model = ResidualBART()
            train_xt = np.concatenate((train_x, train_t.reshape(-1, 1)), axis=1)
            test_xt = np.concatenate((test_x, test_t.reshape(-1, 1)), axis=1)
            val_xt = np.concatenate((val_x, val_t.reshape(-1, 1)), axis=1)
            model.fit(train_xt, train_yf)  # Fit the model
            train_yf_pred = model.predict(train_xt)  # Make predictions on the train set
            test_yf_pred = model.predict(test_xt)  # Make predictions on new data
            val_yf_pred = model.predict(val_xt)  # Make predictions on new data

            # model = SklearnModel()  # Use default parameters
            model = ResidualBART()
            model.fit(train_xt, train_ycf)  # Fit the model
            train_ycf_pred = model.predict(train_xt)  # Make predictions on the train set
            test_ycf_pred = model.predict(test_xt)  # Make predictions on new data
            val_ycf_pred = model.predict(val_xt)  # Make predictions on new data

            train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
            val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
            test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)

        elif args.model == "grf":

            # print(train_y0)
            # print(np.concatenate((train_y0, train_y1), axis=1))
            try:
                # Base model
                forest = MultiOutputGRF(CausalForest())
                forest.fit(train_x, train_t, np.concatenate((train_y0, train_y1), axis=1))
                train_y_pred = forest.predict(train_x)
                val_y_pred = forest.predict(val_x)
                test_y_pred = forest.predict(test_x)
                train_y0_pred = train_y_pred[:, 0]
                train_y1_pred = train_y_pred[:, 1]
                val_y0_pred = val_y_pred[:, 0]
                val_y1_pred = val_y_pred[:, 1]
                test_y0_pred = test_y_pred[:, 0]
                test_y1_pred = test_y_pred[:, 1]
            except UnboundLocalError:
                forest = CausalForest()
                forest.fit(train_x, train_t, train_yf)
                train_yf_pred = forest.predict(train_x)
                val_yf_pred = forest.predict(val_x)
                test_yf_pred = forest.predict(test_x)

                forest.fit(train_x, 1-train_t, train_yf)
                train_ycf_pred = forest.predict(train_x)
                val_ycf_pred = forest.predict(val_x)
                test_ycf_pred = forest.predict(test_x)

                train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
                val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
                test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)

            # TODO: Double Machine Learning
            # forest = CausalForestDML()
            # train_xt = np.concatenate((train_x, train_t.reshape(-1, 1)), axis=1)
            # test_xt = np.concatenate((test_x, test_t.reshape(-1, 1)), axis=1)
            # val_xt = np.concatenate((val_x, val_t.reshape(-1, 1)), axis=1)
            # forest.fit(train_xt, train_yf)  # Fit the model
            # train_yf_pred = forest.predict(train_xt)  # Make predictions on the train set
            # test_yf_pred = forest.predict(test_xt)  # Make predictions on new data
            # val_yf_pred = forest.predict(val_xt)  # Make predictions on new data
            #
            # # model = SklearnModel()  # Use default parameters
            # forest = CausalForestDML()
            # forest.fit(train_xt, train_ycf)  # Fit the model
            # train_ycf_pred = forest.predict(train_t)  # Make predictions on the train set
            # test_ycf_pred = forest.predict(test_xt)  # Make predictions on new data
            # val_ycf_pred = forest.predict(val_xt)  # Make predictions on new data
            #
            # train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
            # val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
            # test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)

        eval = Evaluator()
        ## Performance metrics
        try:
            # 1. PEHE
            test_PEHE = eval.PEHE(test_y0, test_y1, test_y0_pred, test_y1_pred)
            metric_results['PEHE_TEST'].append(np.round(test_PEHE, 4))

            train_PEHE = eval.PEHE(train_y0, train_y1, train_y0_pred, train_y1_pred)
            metric_results['PEHE_TRAIN'].append(np.round(train_PEHE, 4))

            val_PEHE = eval.PEHE(val_y0, val_y1, val_y0_pred, val_y1_pred)
            metric_results['PEHE_VAL'].append(np.round(val_PEHE, 4))
            # 2. ATE
            test_ATE = eval.ATE(test_y0, test_y1, test_y0_pred, test_y1_pred)
            metric_results['ATE_TEST'].append(np.round(test_ATE, 4))

            train_ATE = eval.ATE(train_y0, train_y1, train_y0_pred, train_y1_pred)
            metric_results['ATE_TRAIN'].append(np.round(train_ATE, 4))

            val_ATE = eval.ATE(val_y0, val_y1, val_y0_pred, val_y1_pred)
            metric_results['ATE_VAL'].append(np.round(val_ATE, 4))
        except UnboundLocalError:
            test_p_value, test_policy_curve = eval.policy_val(test_t, test_yf, test_y0_pred, test_y1_pred)
            metric_results['P_RISK_TEST'].append(np.round(1 - test_p_value, 4))

            train_p_value, train_policy_curve = eval.policy_val(train_t, train_yf, train_y0_pred, train_y1_pred)
            metric_results['P_RISK_TRAIN'].append(np.round(1 - train_p_value, 4))

            val_p_value, val_policy_curve = eval.policy_val(val_t, val_yf, val_y0_pred, val_y1_pred)
            metric_results['P_RISK_VAL'].append(np.round(1 - val_p_value, 4))

        print(metric_results)
    for key, item in metric_results.items():
        try:
            metric_results[key] = (np.round(np.mean(item), 4), np.round(np.std(item), 4))
        except:
            pass
    print(metric_results)
    # torch.dump()
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
        choices=['ganite', 'site', 'bart', 'grf'],
        default='grf',
        type=str
    )
    parser.add_argument(
        '--data_name',
        choices=['twins', 'ihdp', 'jobs'],
        default='jobs',
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
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=100,
        type=int)
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
