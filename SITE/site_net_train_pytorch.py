import os
import sys
import datetime
import time

import torch
import argparse

# from SITE.simi_ite.evaluation import pehe_nn
from SITE.simi_ite.util_pytorch import *
from SITE.simi_ite.site_net_pytorch import SiteNet, pddmTransformation
from SITE.simi_ite.loss_pytorch import lossCalc
from torch.optim.lr_scheduler import StepLR
from data.data_loader import load_data, load_batch

''' Define parameter flags '''

FLAGS = argparse.ArgumentParser(description="")
FLAGS.add_argument('--loss', type=str, default='l2', help='Which loss function to use (l1/l2/log)')
FLAGS.add_argument('--n_in', type=int, default=3, help='Number of representation layers. ')
FLAGS.add_argument('--n_out', type=int, default=3, help='Number of regression layers. ')
FLAGS.add_argument('--p_lambda', type=float, default=0.0, help='Weight decay regularization parameter. ')
FLAGS.add_argument('--rep_weight_decay', type=int, default=0,
                   help='Whether to penalize representation layers with weight decay')
FLAGS.add_argument('--dropout_in', type=float, default=0.1, help="""Input layers dropout rate. """)
FLAGS.add_argument('--dropout_out', type=float, default=0.1, help="""Output layers dropout rate. """)
FLAGS.add_argument('--nonlin', type=str, default='elu', help="""Kind of non-linearity. Default relu. """)
FLAGS.add_argument('--lrate', type=float, default=0.001, help="""Learning rate. """)
FLAGS.add_argument('--decay', type=float, default=0.5, help="""RMSProp decay. """)
FLAGS.add_argument('--batch_size', type=int, default=100, help="""Batch size. """)
FLAGS.add_argument('--dim_in', type=int, default=200, help="""Pre-representation layer dimensions. """)
FLAGS.add_argument('--dim_out', type=int, default=100, help="""Post-representation layer dimensions. """)
FLAGS.add_argument('--batch_norm', type=int, default=0, help="""Whether to use batch normalization. """)
FLAGS.add_argument('--normalization', type=str, default='divide',
                   help="""How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
FLAGS.add_argument('--experiments', type=int, default=1, help="""Number of experiments. """)
FLAGS.add_argument('--iterations', type=int, default=2000, help="""Number of iterations. """)
FLAGS.add_argument('--weight_init', type=float, default=0.01, help="""Weight initialization scale. """)
FLAGS.add_argument('--lrate_decay', type=float, default=0.97, help="""Decay of learning rate every 100 iterations """)
FLAGS.add_argument('--varsel', type=int, default=0, help="""Whether the first layer performs variable selection. """)
FLAGS.add_argument('--outdir', type=str, default='./results/ihdp', help="""Output directory. """)
FLAGS.add_argument('--datadir', type=str, default='./data/', help="""data directory. """)
FLAGS.add_argument('--dataform', type=str, default='ihdp_npci_1-100.train.npz',
                   help="""Training data filename form. (ihdp_npci_1-100.train.npz/twins_train_preprocessed.npz/jobs_DW_bin.train.npz)""")
FLAGS.add_argument('--data_test', type=str, default='ihdp_npci_1-100.test.npz', help="""Test data filename form. """)
FLAGS.add_argument('--sparse', type=int, default=0, help="""Whether data is stored in sparse format (.x, .y). """)
FLAGS.add_argument('--seed', type=int, default=42, help="""Seed. """)
FLAGS.add_argument('--repetitions', type=int, default=1, help="""Repetitions with different seed.""")
FLAGS.add_argument('--use_p_correction', type=int, default=1,
                   help="""Whether to use population size p(t) in mmd/disc/wass.""")
FLAGS.add_argument('--optimizer', type=str, default='Adam',
                   help="""Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
FLAGS.add_argument('--imb_fun', type=str, default='mmd_lin',
                   help="""Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
FLAGS.add_argument('--output_csv', type=int, default=0, help="""Whether to save a CSV file with the results_try1""")
FLAGS.add_argument('--output_delay', type=int, default=100, help="""Number of iterations between log/loss outputs. """)
FLAGS.add_argument('--pred_output_delay', type=int, default=-1,
                   help="""Number of iterations between prediction outputs. (-1 gives no intermediate datasets). """)
FLAGS.add_argument('--debug', type=int, default=0, help="""Debug mode. """)
FLAGS.add_argument('--save_rep', type=int, default=0, help="""Save representations after training. """)
FLAGS.add_argument('--val_part', type=float, default=0.3, help="""Validation part. """)
FLAGS.add_argument('--split_output', type=bool, default=1,
                   help="""Whether to split datasets layers= between treated and control. """)
FLAGS.add_argument('--reweight_sample', type=bool, default=1,
                   help="""Whether to reweight sample for prediction loss with average treatment probability. """)
FLAGS.add_argument('--p_pddm', type=float, default=0.0, help="""PDDM unit parameter """)
FLAGS.add_argument('--p_mid_point_mini', type=float, default=0.0,
                   help="""Mid point distance minimization parameter """)
FLAGS.add_argument('--dim_pddm', type=float, default=200.0, help="""Dimension in PDDM fist layer """)
FLAGS.add_argument('--dim_c', type=float, default=200.0, help="""Dimension in PDDM unit for c """)
FLAGS.add_argument('--dim_s', type=float, default=100.0, help="""Dimension in PDDM unit for s """)
FLAGS.add_argument('--propensity_dir', type=str, default='./SITE/propensity_score/ihdp_propensity_model.sav',
                   help="""Dir where the propensity model is saved""")
FLAGS.add_argument('--equal_sample', type=bool, default=0,
                   help="""Whether to fetch equal number of samples with different labels. """)
FLAGS = FLAGS.parse_args()
NUM_ITERATIONS_PER_DECAY = 100


def three_pair_extration(x, t, yf, propensity_dir):
    '''
    :param x: pre-treatment covariates
    :param t: treatment
    :param yf: factual outcome
    :param propensity_dir: the directory that saves propensity model
    :return: the selected three pairs' pre-treatment covariates, index, treatment,
    factual_outcome, and the similarity score ((x_k, x_l), (x_m, x_n), (x_k, x_m), (x_i, x_k), (x_j, x_m))
    '''
    three_pairs, I_three_pairs = find_three_pairs(x, t, propensity_dir)
    t_three_pairs = t[I_three_pairs]
    y_three_pairs = yf[I_three_pairs]
    three_pairs_simi = get_three_pair_simi(three_pairs, propensity_dir)
    return three_pairs, I_three_pairs, t_three_pairs, y_three_pairs, three_pairs_simi


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        print(name, param)
        if not param.requires_grad: continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def train(site_model, optimizer, D, I_valid, loss_calculator, D_test, logfile, i_exp, scheduler):

    """ Train/validation split """
    n = D['x'].shape[0]
    I = range(n)
    I_train = list(set(I) - set(I_valid))
    n_train = len(I_train)

    x_train = D['x'][I_train, :]
    x_val = D['x'][I_valid, :]
    t_train = D['t'][I_train, :]
    t_val = D['t'][I_valid, :]
    yf_train = D['yf'][I_train, :]
    yf_val = D['yf'][I_valid, :]

    if D['HAVE_TRUTH']:
        x_cf = x_train
        t_cf = 1 - t_train
        y_cf = D['ycf'][I_train, :]

    ''' Compute treatment probability'''
    p_treated = np.mean(t_train)

    ''' Set up three pairs for calculating losses'''
    p_t = p_treated
    # print(p_t)
    three_pairs_train, _, _, _, three_pairs_simi_train = three_pair_extration(
        x_train, t_train, yf_train, FLAGS.propensity_dir)

    if FLAGS.val_part > 0:
        three_pairs_valid, _, _, _, three_pairs_simi_valid = three_pair_extration(
            x_val, t_val, yf_val, FLAGS.propensity_dir)

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []
    losses = []

    objnan = False

    reps = []
    reps_test = []
    valid_list = []
    # print(random.sample(range(0, n_train), FLAGS.batch_size))
    for i in range(FLAGS.iterations):
        ''' Fetch sample '''
        t_index = 0

        while t_index < 0.05 or t_index > 0.95:
            I = random.sample(range(0, n_train), FLAGS.batch_size)
            x_batch = x_train[I, :]
            t_batch = t_train[I]
            y_batch = yf_train[I]
            t_index = np.mean(t_batch)

        ''' Do one step of gradient descent '''
        if not objnan:
            site_model.train()
            for param in site_model.parameters():
                param.requires_grad = True

            ''' Extract three-pairs for training'''
            three_pairs_batch, _, _, _, three_pairs_simi = three_pair_extration(
                x_batch, t_batch, y_batch, FLAGS.propensity_dir)
            # print(three_pairs_batch)
            ''' Make a prediction '''
            y_pred_batch, _= site_model(torch.Tensor(x_batch), torch.Tensor(t_batch))

            ''' Calculate losses'''
            # print(three_pairs_batch)
            pddm_loss, mid_loss = site_model.pddm_mid_loss(torch.Tensor(three_pairs_batch),
                                                           torch.Tensor(three_pairs_simi))
            loss = loss_calculator.calc_loss(t_batch, p_t, torch.Tensor(y_batch), y_pred_batch, pddm_loss, mid_loss)
            # print(t_batch)
            # print(loss_calculator.pred_loss)
            ''' Optimize '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            site_model.eval()
            for param in site_model.parameters():
                param.requires_grad = False

        if i % FLAGS.output_delay == 0 or i == FLAGS.iterations - 1:

            ''' Make a prediction '''
            y_pred_f, rep = site_model(torch.Tensor(x_train), torch.Tensor(t_train))
            # print(rep)
            pddm_loss, mid_loss = site_model.pddm_mid_loss(torch.Tensor(three_pairs_train),
                                                           torch.Tensor(three_pairs_simi_train))

            ''' Calculate the loss '''
            obj_loss = loss_calculator.calc_loss(t_train, p_t, torch.Tensor(yf_train), y_pred_f, pddm_loss, mid_loss)

            f_error = loss_calculator.pred_loss
            pddm_loss_batch = pddm_loss
            mid_dist_batch = mid_loss

            cf_error = np.nan
            valid_obj = np.nan
            valid_f_error = np.nan

            if D['HAVE_TRUTH']:
                y_pred_cf, _ = site_model(torch.Tensor(x_cf), torch.Tensor(t_cf))
                pddm_loss, mid_loss = site_model.pddm_mid_loss(torch.Tensor(three_pairs_train),
                                                               torch.Tensor(three_pairs_simi_train))
                loss_calculator.calc_loss(t_cf, np.nan, torch.Tensor(y_cf), y_pred_cf, pddm_loss, mid_loss)
                cf_error = loss_calculator.pred_loss

            if FLAGS.val_part > 0:
                y_pred_valf, _= site_model(torch.Tensor(x_val), torch.Tensor(t_val))
                pddm_loss, mid_loss = site_model.pddm_mid_loss(torch.Tensor(three_pairs_valid),
                                                               torch.Tensor(three_pairs_simi_valid))
                loss_calculator.calc_loss(t_val, p_t, torch.Tensor(yf_val), y_pred_valf, pddm_loss, mid_loss)
                valid_obj, valid_f_error = loss_calculator.tot_loss, loss_calculator.pred_loss

            ''' Print and save the losses '''
            # print(obj_loss)
            # print([obj_loss, f_error, cf_error, valid_f_error, valid_obj])
            try:
                obj_loss_detach = obj_loss.detach()
                f_error_detach = f_error.detach()
                cf_error_detach = cf_error.detach()
                valid_f_error_detach = valid_f_error.detach()
                valid_obj_detach = valid_obj.detach()
                losses.append([obj_loss_detach, f_error_detach, cf_error_detach, valid_f_error_detach, valid_obj_detach])
                loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tPDDM: %.2g,\tmid_p: %.2g,\tVal: %.3f,\tValObj: %.2f' \
                           % (obj_loss_detach, f_error_detach, cf_error_detach, pddm_loss_batch, mid_dist_batch, valid_f_error_detach, valid_obj_detach)
            except AttributeError:
                print("Loss Has NaN")
                losses.append([obj_loss, f_error, cf_error, valid_f_error, valid_obj])
                loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tPDDM: %.2g,\tmid_p: %.2g,\tVal: %.3f,\tValObj: %.2f' \
                           % (obj_loss, f_error, cf_error, pddm_loss_batch, mid_dist_batch, valid_f_error, valid_obj)
            # print losses
            valid_list.append(valid_f_error)
            if FLAGS.loss == 'log':
                y_pred, _= site_model(torch.Tensor(x_batch), torch.Tensor(t_batch))
                y_pred = 1.0 * (y_pred > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred.numpy())))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

            if torch.isnan(obj_loss):
                log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i == FLAGS.iterations - 1:

            y_pred_f, _ = site_model(torch.Tensor(D['x']), torch.Tensor(D['t']))
            # for name, param in site_model.named_parameters():
            #     print(name, param)
            y_pred_cf, _ = site_model(torch.Tensor(D['x']), torch.Tensor(1-D['t']))
            # print(y_pred_f)

            preds_train.append(np.concatenate((y_pred_f.detach(), y_pred_cf.detach()), axis=1))
            # print(preds_train)
            if D_test is not None:

                # print(D_test['x'])
                y_pred_f_test, _ = site_model(torch.Tensor(D_test['x']), torch.Tensor(D_test['t']))
                y_pred_cf_test, _ = site_model(torch.Tensor(D_test['x']), torch.Tensor(1-D_test['t']))
                preds_test.append(np.concatenate((y_pred_f_test.detach(), y_pred_cf_test.detach()), axis=1))

            if FLAGS.save_rep and i_exp == 1:
                # The D['t'] here does not affect the result, since reps_i only depend on D['x']
                _, reps_i = site_model(torch.Tensor(D['x']), torch.Tensor(D['t']))
                reps.append(reps_i)

                if D_test is not None:
                    # The D_test['t'] here does not affect the result
                    _, reps_test_i = site_model(torch.Tensor(D_test['x']), torch.Tensor(D_test['t']))
                    reps_test.append(reps_test_i)

            # A rudimentary way to calculate pehe
            # print(pehe_nn(y_pred_f.detach().numpy(), y_pred_cf.detach().numpy(), D['yf'], D['x'], D['t']))

    # fig, ax = plt.subplots(figsize=(10,5))
    # ax.plot(range(len(valid_list)), valid_list, '-b',label='validation')
    # plt.show()

    return losses, preds_train, preds_test, reps, reps_test


def site(x_train, t_train, yf_train, dim, parameters):
    param_model = parameters['model']
    param_optim = parameters['optim']
    param_loss = parameters['loss']
    dims = [dim, param_model['dim_in'], param_model['dim_out'], int(param_model['dim_pddm']),
            int(param_model['dim_c']), int(param_model['dim_s'])]
    site_model = SiteNet(dims, dropout_in=param_model['dropout_in'], dropout_out=param_model['dropout_out'], FLAGS=param_model)

    # print(site_model.inLayers.layers.parameters())
    # pddm_model = pddmTransformation(dims, FLAGS)

    # Defining optimizer
    if parameters['optimizer'] == 'Adagrad':
        optimizer = torch.optim.Adagrad(site_model.parameters(), lr=param_optim['lrate'], weight_decay=param_loss['p_lambda'])
    elif parameters['optimizer'] == 'GradientDescent':
        optimizer = torch.optim.SGD(site_model.parameters(), lr=param_optim['lrate'], weight_decay=param_loss['p_lambda'])
    elif parameters['optimizer'] == 'Adam':
        # optimizer = torch.optim.AdamW(site_model.parameters(), lr=param_optim['lrate'], weight_decay=param_loss['p_lambda'])
        optimizer = torch.optim.Adam(site_model.parameters(), lr=param_optim['lrate'])
    else:
        optimizer = torch.optim.RMSprop(site_model.parameters(), lr=param_optim['lrate'], weight_decay=param_optim['decay'])

    # Defining scheduler
    scheduler = StepLR(optimizer, step_size=param_optim['iter_per_decay'], gamma=param_optim['lrate_decay'])

    # Defining loss calculater, which we would use to calculate multiple losses
    lossCalculator = lossCalc(param_loss['p_lambda'], param_loss['p_mid_point_mini'], param_loss['p_pddm'], param_loss)
    trained_model = train(site_model, optimizer, scheduler, lossCalculator, x_train, t_train, yf_train, parameters)
    return trained_model


def train(model, optimizer, scheduler, lossCalculator, x_train, t_train, yf_train, parameters):
    objnan = False
    n_train = t_train.shape[0]
    # print(n_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(t_train)

    ''' Set up three pairs for calculating losses'''
    p_t = p_treated

    # T1 = time.time()
    # print(T1)
    three_pairs_train, _, _, _, three_pairs_simi_train = three_pair_extration(
        x_train, t_train, yf_train, parameters["propensity_dir"])
    # T2 = time.time()
    # print(T2-T1)
    # three_pairs_valid, _, _, _, three_pairs_simi_valid = three_pair_extration(
    #     x_val, t_val, yf_val, parameters["propensity_dir"])
    for i in range(parameters['iteration']):
        # print(i)
        ''' Fetch sample '''
        t_index = 0

        while t_index < 0.05 or t_index > 0.95:
            I = random.sample(range(0, n_train), parameters['batch_size'])
            x_batch = x_train[I, :]
            t_batch = t_train[I]
            y_batch = yf_train[I]
            t_index = np.mean(t_batch)

        if not objnan:
            model.train()
            for param in model.parameters():
                param.requires_grad = True

            ''' Extract three-pairs for training'''
            three_pairs_batch, _, _, _, three_pairs_simi = three_pair_extration(
                x_batch, t_batch, y_batch, parameters['propensity_dir'])
            # print(three_pairs_batch)
            ''' Make a prediction '''
            y_pred_batch, _ = model(torch.Tensor(x_batch), torch.Tensor(t_batch))
            ''' Calculate losses'''
            # print(three_pairs_batch)
            pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_batch),
                                                           torch.Tensor(three_pairs_simi))
            loss = lossCalculator.calc_loss(t_batch, p_t, torch.Tensor(y_batch), y_pred_batch, pddm_loss, mid_loss)
            # print(t_batch)
            # print(loss_calculator.pred_loss)
            ''' Optimize '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        if i % parameters['output_delay'] == 0 or i == parameters['iteration'] - 1:

            ''' Make a prediction '''
            y_pred_f, rep = model(torch.Tensor(x_train), torch.Tensor(t_train))
            # print(rep)
            pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_train),
                                                           torch.Tensor(three_pairs_simi_train))

            ''' Calculate the loss '''
            obj_loss = lossCalculator.calc_loss(t_train, p_t, torch.Tensor(yf_train), y_pred_f, pddm_loss, mid_loss)

            f_error = lossCalculator.pred_loss
            pddm_loss_batch = pddm_loss
            mid_dist_batch = mid_loss

            cf_error = np.nan
            valid_obj = np.nan
            valid_f_error = np.nan

            # if D['HAVE_TRUTH']:
            #     y_pred_cf, _ = model(torch.Tensor(x_cf), torch.Tensor(t_cf))
            #     pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_train),
            #                                                    torch.Tensor(three_pairs_simi_train))
            #     lossCalculator.calc_loss(t_cf, np.nan, torch.Tensor(y_cf), y_pred_cf, pddm_loss, mid_loss)
            #     cf_error = lossCalculator.pred_loss
            #
            # if FLAGS.val_part > 0:
            #     y_pred_valf, _= model(torch.Tensor(x_val), torch.Tensor(t_val))
            #     pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_valid),
            #                                                    torch.Tensor(three_pairs_simi_valid))
            #     lossCalculator.calc_loss(t_val, p_t, torch.Tensor(yf_val), y_pred_valf, pddm_loss, mid_loss)
            #     valid_obj, valid_f_error = lossCalculator.tot_loss, lossCalculator.pred_loss

            ''' Print and save the losses '''
            losses = []
            valid_list = []
            try:
                obj_loss_detach = obj_loss.detach()
                f_error_detach = f_error.detach()
                cf_error_detach = cf_error.detach()
                valid_f_error_detach = valid_f_error.detach()
                valid_obj_detach = valid_obj.detach()
                losses.append([obj_loss_detach, f_error_detach, cf_error_detach, valid_f_error_detach, valid_obj_detach])
                loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tPDDM: %.2g,\tmid_p: %.2g,\tVal: %.3f,\tValObj: %.2f' \
                           % (obj_loss_detach, f_error_detach, cf_error_detach, pddm_loss_batch, mid_dist_batch, valid_f_error_detach, valid_obj_detach)
            except AttributeError:
                print("Loss Has NaN")
                losses.append([obj_loss, f_error, cf_error, valid_f_error, valid_obj])
                loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tPDDM: %.2g,\tmid_p: %.2g,\tVal: %.3f,\tValObj: %.2f' \
                           % (obj_loss, f_error, cf_error, pddm_loss_batch, mid_dist_batch, valid_f_error, valid_obj)
            # print losses
            valid_list.append(valid_f_error)
            if FLAGS.loss == 'log':
                y_pred, _ = model(torch.Tensor(x_batch), torch.Tensor(t_batch))
                y_pred = 1.0 * (y_pred > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred.numpy())))
                loss_str += ',\tAcc: %.2f%%' % acc

            print(loss_str)
            # log(logfile, loss_str)
            #
            # if torch.isnan(obj_loss):
            #     log(logfile, 'Objective is NaN. Skipping.')
            #     objnan = True

    return model


def site_predict(model, x, t):
    yf, _ = model(x, t)
    ycf, _ = model(x, 1-t)
    y0 = yf * (1 - t) + ycf * t
    y1 = yf * t + ycf * (1 - t)
    # y_hat = torch.cat((yf, ycf), 1)
    return y0.detach().numpy(), y1.detach().numpy()

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir + 'result'
    npzfile_test = outdir + 'result.test'
    repfile = outdir + 'reps'
    repfile_test = outdir + 'reps.test'
    outform = outdir + 'y_pred'
    outform_test = outdir + 'y_pred.test'
    lossform = outdir + 'loss'
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '':  # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir + 'config.txt', FLAGS)

    log(logfile,
        'Training with hyperparameters: p_pddm=%.2g, r_mid_point_mini=%.2g' % (FLAGS.p_pddm, FLAGS.p_mid_point_mini))

    ''' Load data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile, 'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath, FLAGS)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test, FLAGS)



    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out, int(FLAGS.dim_pddm), int(FLAGS.dim_c), int(FLAGS.dim_s)]

    # wd_dict = {}

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    all_train_rep = []
    all_test_rep = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions > 1:
        if FLAGS.experiments > 1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1, n_experiments + 1):

        site_model = SiteNet(dims, dropout_in=FLAGS.dropout_in, dropout_out=FLAGS.dropout_out, FLAGS=FLAGS)

        # print(site_model.inLayers.layers.parameters())
        # pddm_model = pddmTransformation(dims, FLAGS)

        # Defining optimizer
        if FLAGS.optimizer == 'Adagrad':
            optimizer = torch.optim.Adagrad(site_model.parameters(), lr=FLAGS.lrate, weight_decay=FLAGS.p_lambda)
        elif FLAGS.optimizer == 'GradientDescent':
            optimizer = torch.optim.SGD(site_model.parameters(), lr=FLAGS.lrate, weight_decay=FLAGS.p_lambda)
        elif FLAGS.optimizer == 'Adam':
            optimizer = torch.optim.AdamW(site_model.parameters(), lr=FLAGS.lrate, weight_decay=FLAGS.p_lambda)
            # optimizer = torch.optim.Adam(site_model.parameters(), lr=FLAGS.lrate)
        else:
            optimizer = torch.optim.RMSprop(site_model.parameters(), lr=FLAGS.lrate, weight_decay=FLAGS.decay)

        # Defining scheduler
        scheduler = StepLR(optimizer, step_size=NUM_ITERATIONS_PER_DECAY, gamma=FLAGS.lrate_decay)

        # Defining loss calculater, which we would use to calculate multiple losses
        lossCalculator = lossCalc(FLAGS.p_lambda, FLAGS.p_mid_point_mini, FLAGS.p_pddm, FLAGS)

        if FLAGS.repetitions > 1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load data (if multiple repetitions, reuse first set)'''

        if i_exp == 1 or FLAGS.experiments > 1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x'] = D['x'][:, :, i_exp - 1]
                D_exp['t'] = D['t'][:, i_exp - 1:i_exp]
                D_exp['yf'] = D['yf'][:, i_exp - 1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x'] = D_test['x'][:, :, i_exp - 1]
                    D_exp_test['t'] = D_test['t'][:, i_exp - 1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:, i_exp - 1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:, i_exp - 1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath, FLAGS)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test, FLAGS)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']
        ''' Split into training and validation sets '''
        if FLAGS.equal_sample > 0:
            index_y_c_0 = np.intersect1d(np.where(D_exp['t'] < 1), np.where(D_exp['yf'] < 1))
            index_y_c_1 = np.intersect1d(np.where(D_exp['t'] < 1), np.where(D_exp['yf'] > 0))
            index_y_t_0 = np.intersect1d(np.where(D_exp['t'] > 0), np.where(D_exp['yf'] < 1))
            index_y_t_1 = np.intersect1d(np.where(D_exp['t'] > 0), np.where(D_exp['yf'] > 0))

            I_train_c_0, I_valid_c_0 = validation_split_equal(index_y_c_0, FLAGS.val_part)
            I_train_c_1, I_valid_c_1 = validation_split_equal(index_y_c_1, FLAGS.val_part)
            I_train_t_0, I_valid_t_0 = validation_split_equal(index_y_t_0, FLAGS.val_part)
            I_train_t_1, I_valid_t_1 = validation_split_equal(index_y_t_1, FLAGS.val_part)
            I_valid = index_y_c_0[I_valid_c_0].tolist() + index_y_c_1[I_valid_c_1].tolist() + \
                      index_y_t_0[I_valid_t_0].tolist() + index_y_t_1[I_valid_t_1].tolist()
        else:

            I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        losses, preds_train, preds_test, reps, reps_test = \
            train(site_model, optimizer, D_exp, I_valid, lossCalculator, D_exp_test, logfile, i_exp, scheduler)

        # print(preds_train)
        ''' Collect all reps '''
        # all_losses.append(losses.detach())
        # all_preds_train.append(preds_train.detach())
        # all_preds_test.append(preds_test.detach())
        try:
            all_preds_train.append(preds_train.detach())
            all_preds_test.append(preds_test.detach())
            all_losses.append(losses.detach())
        except AttributeError:
            all_preds_train.append(preds_train)
            all_preds_test.append(preds_test)
            all_losses.append(losses)

        ''' Fix shape for datasets (n_units, dim, n_reps, n_outputs) '''
        # print(all_losses)
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        if has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform, i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test, i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform, i_exp), losses, delimiter=',')

        # TODO: Edit to fit pytorch implementation
        ''' Compute weights if doing variable selection '''
        # if FLAGS.varsel:
        #     if i_exp == 1:
        #         all_weights = sess.run(SITE.weights_in[0])
        #         all_beta = sess.run(SITE.weights_pred)
        #     else:
        #         all_weights = np.dstack((all_weights, sess.run(SITE.weights_in[0])))
        #         all_beta = np.dstack((all_beta, sess.run(SITE.weights_pred)))

        # TODO: out_preds_train shape incorrect
        ''' Save results_try1 and predictions '''
        # print(out_preds_train.shape)
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta,
                     val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        # ''' Save representations '''
        # if FLAGS.save_rep and i_exp == 1:
        #     np.savez(repfile, rep=reps)
        #
        #     if has_test:
        #         np.savez(repfile_test, rep=reps_test)


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir + '/results_' + timestamp + '/'
    os.mkdir(outdir)
    run(outdir)
