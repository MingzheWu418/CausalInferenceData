import random

import numpy as np
import pandas as pd
import torch
import argparse

from GANITE.metrics import PEHE, ATE, policy_val
from GANITE.train_pytorch import ganite, ganite_predict
from SITE.site_net_train_pytorch import site, site_predict
from CFR.train import cfrnet, cfrnet_predict
from eval.evaluator import Evaluator
from utils.utils_pytorch import comb_potential_outcome
from data.data_loader import split, cross_val_index, load_data, load_batch
# from bartpy.sklearnmodel import SklearnModel
# from bartpy.extensions.baseestimator import ResidualBART
from econml.grf import CausalForest, CausalIVForest, MultiOutputGRF
from econml.dml import CausalForestDML
from subprocess import call
from causality import Xlearner, Slearner, Tlearner, Metric, PropensityScoreMatching, InverseProbabilityWeighting, DoublyRobustEstimation

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
        train_data_path = "./datasets/ihdp_npci_1-100.train.npz"
        test_data_path = "./datasets/ihdp_npci_1-100.test.npz"
    if args.data_name == "jobs":
        data_path = "./datasets/jobs_DW_bin.npz"
    if args.data_name == "acic":
        data_path = "./datasets/acic.npz"
    if args.data_name == "lalonde":
        data_path = "./datasets/lalonde.npz"
    d = load_data(data_path)
    # print(d)
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

    if args.data_name == "ihdp":  # Currently only loading ihdp into batchs because it is too large
        train_dataset = load_data(train_data_path) # 672 samples
        test_dataset = load_data(test_data_path) # 75 samples

    # batch = np.random.randint(low=0, high=args.batch_size, size=1)
    test_x = test_dataset['x']
    test_yf = test_dataset['yf']
    test_t = test_dataset['t']
    try:
        test_ycf = test_dataset['ycf']
        test_y0, test_y1 = comb_potential_outcome(test_yf, test_ycf, test_t)
    except:
        print("No Counterfactual data Available")

    indexes = cross_val_index(train_dataset, args.folds)
    # [[1,2,3,4,74, ...],[],[],[],[]]

    if args.model not in ['xlearner', 'slearner', 'tlearner', 'ipw', 'doublyrobust']:
        with open("./configs/" + args.model + ".yaml", "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    # print(parameters)

    for fold, I_val in enumerate(indexes):

        # Handle ihdp separately because
        if args.data_name == 'ihdp':
            # print(train_dataset['x'].shape)
            # print(test_dataset['x'].shape)
            batch = np.random.randint(low=0, high=args.batch_size, size=1)
            batch = 0
            I_train = np.setdiff1d(np.arange(train_dataset['t'].shape[0]), I_val)

            train_x = train_dataset['x'][I_train, :, batch]
            print(train_x)
            print(train_x.shape)
            train_t = train_dataset['t'][I_train, batch]
            train_yf = train_dataset['yf'][I_train, batch]
            val_x = train_dataset['x'][I_val, :, batch]
            # print(val_x)
            # print(val_x.shape)
            val_t = train_dataset['t'][I_val, batch]
            val_yf = train_dataset['yf'][I_val, batch]
            # print(test_t.shape)
            test_x = np.squeeze(test_dataset['x'][:, :, batch])
            test_t = np.squeeze(test_dataset['t'][:, batch])
            # print(test_t.shape)

            train_ycf = train_dataset['ycf'][I_train, batch]
            train_y0, train_y1 = comb_potential_outcome(train_yf, train_ycf, train_t)
            val_ycf = train_dataset['ycf'][I_val, batch]
            val_y0, val_y1 = comb_potential_outcome(val_yf, val_ycf, val_t)
            # print(test_yf.shape)
            test_yf = np.squeeze(test_dataset['yf'][:, batch])
            # print(test_yf.shape)
            test_ycf = np.squeeze(test_dataset['ycf'][:, batch])
            test_y0, test_y1 = comb_potential_outcome(test_yf, test_ycf, test_t)
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

        # train models based on configuration
        if args.model == "ganite":
            model = ganite(train_x, train_t, train_yf, parameters)
            train_y0_pred, train_y1_pred = ganite_predict(model, train_x)
            test_y0_pred, test_y1_pred = ganite_predict(model, test_x)
            val_y0_pred, val_y1_pred = ganite_predict(model, val_x)
            # either yf ycf. outcome of shape n*1
            print('Finish GANITE training and potential outcome estimations')
        elif args.model == "cfrnet":
            # (x_train, t_train, yf_train, dim, parameters)
            # train_dataset = load_data("./datasets/ihdp_train_1.npz")
            # # # print(train_dataset)
            # train_x = train_dataset['x']
            # train_t = train_dataset['t']
            # # print(train_t, train_t.shape)
            # train_yf = train_dataset['yf']
            # train_ycf = train_dataset['ycf']
            # mu0 = train_dataset['mu0']
            # mu1 = train_dataset['mu1']
            # train_y0, train_y1 = comb_potential_outcome(train_yf, train_ycf, train_t)
            data = np.loadtxt(open("./datasets/ihdp_sample.csv","rb"),delimiter=",")
            train_x = data[:,5:]
            train_t = data[:,0:1]
            train_yf = data[:,1:2]
            train_ycf = data[:,2:3]

            train_t_cfr = train_t.reshape(-1, 1)
            train_y0, train_y1 = comb_potential_outcome(train_yf, train_ycf, train_t_cfr)
            val_t_cfr = val_t.reshape(-1, 1)
            test_t_cfr = test_t.reshape(-1, 1)
            model = cfrnet(train_x, train_t_cfr, train_yf, d['dim'], parameters)
            # site_predict(model, x, t)
            # print(train_x)
            # print(train_x.shape)
            train_yf_pred = cfrnet_predict(model, train_x, train_t_cfr)
            # print(test_x)
            # print(test_x.shape)
            test_yf_pred = cfrnet_predict(model, test_x, test_t_cfr)
            val_yf_pred = cfrnet_predict(model, val_x, val_t_cfr)

            model = cfrnet(train_x, 1 - train_t_cfr, train_yf, d['dim'], parameters)
            train_ycf_pred = cfrnet_predict(model, train_x, 1 - train_t_cfr)
            test_ycf_pred = cfrnet_predict(model, test_x, 1 - test_t_cfr)
            val_ycf_pred = cfrnet_predict(model, val_x, 1 - val_t_cfr)

            train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t_cfr)
            val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t_cfr)
            test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t_cfr)

            # print(model)
            print('Finish CFR training and potential outcome estimations')
        elif args.model == "site":
            # propensity_dir = "./propensity_score/" + args.data_name + "_propensity_model.sav"
            # if args.data_name == 'ihdp':
            #     call('python ./SITE/propensity_score_calculation.py %s %s' % (train_data_path, propensity_dir), shell=True)
            # else:
            #     call('python ./SITE/propensity_score_calculation.py %s %s' % (data_path, propensity_dir), shell=True)
            parameters["propensity_dir"] = './SITE/propensity_score/' + args.data_name + '_propensity_model.sav'

            # test if ihdp is correct
            # train_dataset = load_data("./datasets/ihdp_train_1.npz")
            # train_x = train_dataset['x']
            # train_t = train_dataset['t']
            # train_yf = train_dataset['yf']
            # train_ycf = train_dataset['ycf']
            # mu0 = train_dataset['mu0']
            # mu1 = train_dataset['mu1']
            # train_y0, train_y1 = comb_potential_outcome(train_yf, train_ycf, train_t)

            model = site(train_x, train_t, train_yf, d['dim'], parameters)
            for name, param in model.named_parameters():
                print(name, param.data)
            print(model)
            train_yf_pred = site_predict(model, train_x, train_t)
            test_yf_pred = site_predict(model, test_x, test_t)
            val_yf_pred = site_predict(model, val_x, val_t)
            print(np.mean(train_yf_pred), np.std(train_yf_pred), train_yf_pred.shape)

            # model = site(train_x, 1 - train_t, train_yf, d['dim'], parameters)
            train_ycf_pred = site_predict(model, train_x, 1 - train_t)
            test_ycf_pred = site_predict(model, test_x, 1 - test_t)
            val_ycf_pred = site_predict(model, val_x, 1 - val_t)
            print(np.mean(train_ycf_pred), np.std(train_ycf_pred), train_ycf_pred.shape)
            # print(np.mean(train_yf_pred), np.std(train_yf_pred))
            # print(np.mean(train_ycf_pred), np.std(train_ycf_pred))

            train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
            val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
            test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)

            # print(model)
            print('Finish SITE training and potential outcome estimations')
        # elif args.model == "bart":
        #     try:
        #         model = SklearnModel()  # Use default parameters
        #         # model = ResidualBART()
        #         train_xt = np.concatenate((train_x, train_t.reshape(-1, 1)), axis=1)
        #         test_xt = np.concatenate((test_x, test_t.reshape(-1, 1)), axis=1)
        #         val_xt = np.concatenate((val_x, val_t.reshape(-1, 1)), axis=1)
        #
        #         model.fit(train_xt, train_yf.reshape(-1,))  # Fit the model
        #         train_yf_pred = model.predict(train_xt)  # Make predictions on the train set
        #         test_yf_pred = model.predict(test_xt)  # Make predictions on new data
        #         val_yf_pred = model.predict(val_xt)  # Make predictions on new data
        #
        #         model = SklearnModel()  # Use default parameters
        #         # model = ResidualBART()
        #         model.fit(train_xt, train_ycf.reshape(-1,))  # Fit the model
        #
        #         train_ycf_pred = model.predict(train_xt)  # Make predictions on the train set
        #         test_ycf_pred = model.predict(test_xt)  # Make predictions on new data
        #         val_ycf_pred = model.predict(val_xt)  # Make predictions on new data
        #
        #         train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
        #         val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
        #         test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)
        #     except:
        #         model = SklearnModel()  # Use default parameters
        #         # model = ResidualBART()
        #         train_xt = np.concatenate((train_x, train_t.reshape(-1, 1)), axis=1)
        #         test_xt = np.concatenate((test_x, test_t.reshape(-1, 1)), axis=1)
        #         val_xt = np.concatenate((val_x, val_t.reshape(-1, 1)), axis=1)
        #
        #         model.fit(train_xt, train_yf.reshape(-1,))  # Fit the model
        #         train_yf_pred = model.predict(train_xt)  # Make predictions on the train set
        #         test_yf_pred = model.predict(test_xt)  # Make predictions on new data
        #         val_yf_pred = model.predict(val_xt)  # Make predictions on new data
        #
        #         model = SklearnModel()  # Use default parameters
        #         # model = ResidualBART()
        #         train_xt_cf = np.concatenate((train_x, 1 - train_t.reshape(-1, 1)), axis=1)
        #         test_xt_cf = np.concatenate((test_x, 1 - test_t.reshape(-1, 1)), axis=1)
        #         val_xt_cf = np.concatenate((val_x, 1 - val_t.reshape(-1, 1)), axis=1)
        #         model.fit(train_xt_cf, train_yf.reshape(-1,))  # Fit the model
        #         train_ycf_pred = model.predict(train_xt_cf)  # Make predictions on the train set
        #         test_ycf_pred = model.predict(test_xt_cf)  # Make predictions on new data
        #         val_ycf_pred = model.predict(val_xt_cf)  # Make predictions on new data
        #
        #         train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
        #         val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
        #         test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)

        elif args.model == "grf":

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
                # If counter factual is missing
                forest = CausalForest()
                forest.fit(train_x, train_t, train_yf)
                train_yf_pred = forest.predict(train_x)
                val_yf_pred = forest.predict(val_x)
                test_yf_pred = forest.predict(test_x)

                forest.fit(train_x, 1 - train_t, train_yf)
                train_ycf_pred = forest.predict(train_x)
                val_ycf_pred = forest.predict(val_x)
                test_ycf_pred = forest.predict(test_x)

                train_y0_pred, train_y1_pred = comb_potential_outcome(train_yf_pred, train_ycf_pred, train_t)
                val_y0_pred, val_y1_pred = comb_potential_outcome(val_yf_pred, val_ycf_pred, val_t)
                test_y0_pred, test_y1_pred = comb_potential_outcome(test_yf_pred, test_ycf_pred, test_t)
                # print(test_yf_pred.shape)
                # print(test_ycf_pred.shape)
                # print(test_y0_pred.shape)

        elif args.model in ['xlearner', 'slearner', 'tlearner', 'ipw', 'doublyrobust']:
            covariate_names = ['covariate'+str(n) for n in range(train_x.shape[1])]
            cov = pd.DataFrame(train_x, columns = covariate_names)

            df = pd.DataFrame(
                {'treatment': train_t,
                 'y0': train_y0.flatten(),
                 'y1': train_y1.flatten()
                 })

            df = pd.concat([df, cov], axis=1)

            # testing psm model
            # psm_model = PropensityScoreMatching()
            # psm_model.calculateScore(df, covariate_names)
            # psm_model.match_control_with_treated()
            # psm_model.match_treated_with_control()
            # metric = Metric()
            # pehe = metric.get_pehe(*psm_model.get_ite())
            # epsilon_ate = metric.get_epsilon_ate(*psm_model.get_ite())
            # print('Results for the psm model:')
            # print('     pehe: {}, ate: {}'.format(pehe, epsilon_ate))
            # # compare with other psm model
            # matcher = ref_psm()
            # df.loc[:, 'y obs'] = np.where(df['treatment'], df['y1'], df['y0'])
            # cov = {}
            # for c in covariate_names:
            #     cov[c] = 'c'
            # print(cov)
            # print('reference result:', matcher.estimate_ATE(df, 'treatment', 'y obs', cov))

            # print('reference result:', matcher.estimate_ATE(df, 'treatment', 'y obs', {'covariate1':'c'}))

            # testing ipw model
            ipw_model = InverseProbabilityWeighting()
            ipw_model.calculateScore(df, covariate_names)
            metric = Metric()
            epsilon_ate = metric.get_epsilon_ate(*ipw_model.get_ate())
            print('Results for the ipw model:')
            print('     ipw ate:',epsilon_ate)

            # testing doubly robust estimation
            dr_model = DoublyRobustEstimation()
            dr_model.calculateScore(df, covariate_names)
            metric = Metric()
            epsilon_ate = metric.get_epsilon_ate(*dr_model.get_ate())
            print('Results for the dr model:')
            print('     dr ate:',epsilon_ate)

            # testing t-learner
            # tlearner_model = Tlearner(base_model = 'Linear Regression')
            tlearner_model = Tlearner(base_model = 'svm')

            tlearner_model.calculateScore(df, covariate_names)
            metric = Metric()
            pehe = metric.get_pehe(*tlearner_model.get_ite())
            epsilon_ate = metric.get_epsilon_ate(*tlearner_model.get_ite())
            print('Results for the t-learner model:')
            print('     pehe: {}, ate: {}'.format(pehe, epsilon_ate))

            # testing x-learner
            # xlearner_model = Tlearner(base_model = 'Linear Regression')
            xlearner_model = Xlearner(base_model = 'Linear Regression')
            xlearner_model.calculateScore(df, covariate_names)
            metric = Metric()
            pehe = metric.get_pehe(*xlearner_model.get_ite())
            epsilon_ate = metric.get_epsilon_ate(*xlearner_model.get_ite())
            print('Results for the x-learner model:')
            print('     pehe: {}, ate: {}'.format(pehe, epsilon_ate))


            # testing s-learner
            # slearner_model = Slearner(base_model = 'Linear Regression')
            slearner_model = Slearner(base_model = 'svm')
            slearner_model.calculateScore(df, covariate_names)
            metric = Metric()
            pehe = metric.get_pehe(*slearner_model.get_ite())
            epsilon_ate = metric.get_epsilon_ate(*slearner_model.get_ite())
            print('Results for the s-learner model:')
            print('     pehe: {}, ate: {}'.format(pehe, epsilon_ate))



        eval = Evaluator()
        ## Performance metrics
        try:
            # 1. PEHE
            test_PEHE = eval.PEHE(test_yf, test_ycf, test_yf_pred, test_ycf_pred, test_t)
            metric_results['PEHE_TEST'].append(np.round(test_PEHE, 4))
            train_PEHE = eval.PEHE(train_y0, train_y1, train_yf_pred, train_ycf_pred, train_t)
            metric_results['PEHE_TRAIN'].append(np.round(train_PEHE, 4))
            val_PEHE = eval.PEHE(val_yf, val_ycf, val_yf_pred, val_ycf_pred, val_t)
            metric_results['PEHE_VAL'].append(np.round(val_PEHE, 4))

            # 2. ATE
            test_ATE = eval.ATE(test_y0, test_y1, test_y0_pred, test_y1_pred)
            metric_results['ATE_TEST'].append(np.round(test_ATE, 4))

            train_ATE = eval.ATE(train_y0, train_y1, train_y0_pred, train_y1_pred)
            metric_results['ATE_TRAIN'].append(np.round(train_ATE, 4))

            val_ATE = eval.ATE(val_y0, val_y1, val_y0_pred, val_y1_pred)
            metric_results['ATE_VAL'].append(np.round(val_ATE, 4))
        except UnboundLocalError:
            # print(test_t.shape)
            # print(test_yf.shape)
            # print(test_y0_pred.shape)
            test_p_value = eval.policy_val(test_t, test_yf, test_y0_pred, test_y1_pred)
            metric_results['P_RISK_TEST'].append(np.round(1 - test_p_value, 4))

            train_p_value = eval.policy_val(train_t, train_yf, train_y0_pred, train_y1_pred)
            metric_results['P_RISK_TRAIN'].append(np.round(1 - train_p_value, 4))

            val_p_value = eval.policy_val(val_t, val_yf, val_y0_pred, val_y1_pred)
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
        choices=['ganite', 'site', 'bart', 'grf', 'cfrnet', 'xlearner', 'slearner', 'tlearner', 'ipw', 'doublyrobust'],
        default='site',
        type=str
    )
    parser.add_argument(
        '--data_name',
        choices=['twins', 'ihdp', 'jobs', 'acic', 'lalonde'],
        default='ihdp',
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
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=100,
        type=int)
    parser.add_argument(
        '--seed',
        help='random seed',
        default=42,
        type=int)

    args = parser.parse_args()

    # Calls main function

    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    main(args)
