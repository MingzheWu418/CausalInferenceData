import numpy as np
import torch
import random
from .cfr_net_pytorch import CFR
from .cfr_loss_pytorch import lossCalc
from torch.optim.lr_scheduler import ExponentialLR


def cfrnet(x_train, t_train, yf_train, dim, parameters):
    dims = [dim, parameters['dim_in'], parameters['dim_out']]
    site_model = CFR(dims, dropout_in=parameters['dropout_in'], dropout_out=parameters['dropout_out'], FLAGS=parameters)

    # print(site_model.inLayers.layers.parameters())
    # pddm_model = pddmTransformation(dims, FLAGS)

    # Defining optimizer
    if parameters['optimizer'] == 'Adagrad':
        optimizer = torch.optim.Adagrad(site_model.parameters(), lr=parameters['lrate'])
    elif parameters['optimizer'] == 'GradientDescent':
        optimizer = torch.optim.SGD(site_model.parameters(), lr=parameters['lrate'])
    elif parameters['optimizer'] == 'Adam':
        # optimizer = torch.optim.AdamW(site_model.parameters(), lr=param_optim['lrate'], weight_decay=param_loss['p_lambda'])
        optimizer = torch.optim.Adam(site_model.parameters(), lr=parameters['lrate'])
    else:
        optimizer = torch.optim.RMSprop(site_model.parameters(), lr=parameters['lrate'])

    # Defining scheduler
    scheduler = ExponentialLR(optimizer, gamma=parameters['lrate_decay'])

    # Defining loss calculater, which we would use to calculate multiple losses

    # dims, r_alpha, r_lambda, FLAGS
    lossCalculator = lossCalc(dims, parameters['p_alpha'], parameters['p_lambda'], parameters)
    trained_model = train(site_model, optimizer, scheduler, lossCalculator, x_train, t_train, yf_train, parameters)
    return trained_model


def train(model, optimizer, scheduler, lossCalculator, x_train, t_train, yf_train, parameters):
    objnan = False
    n_train = t_train.shape[0]
    # print(n_train)

    ''' Compute treatment probability'''
    if parameters['use_p_correction']:
        p_treated = np.mean(t_train)
    else:
        p_treated = 0.5

    ''' Set up three pairs for calculating losses'''
    p_t = p_treated

    # three_pairs_train, _, _, _, three_pairs_simi_train = three_pair_extration(
    #     x_train, t_train, yf_train, parameters["propensity_dir"])

    # three_pairs_valid, _, _, _, three_pairs_simi_valid = three_pair_extration(
    #     x_val, t_val, yf_val, parameters["propensity_dir"])
    for i in range(parameters['iterations']):
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

            ''' Make a prediction '''
            y_pred_batch, h_rep = model(torch.Tensor(x_batch), torch.Tensor(t_batch))
            weights_pred, weights_in, weights_out = model.get_weights()
            ''' Calculate losses'''
            # print(three_pairs_batch)
            # pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_batch),
            #                                           torch.Tensor(three_pairs_simi))
            #
            loss = lossCalculator.calc_loss(t_batch, p_t, torch.Tensor(y_batch), y_pred_batch, h_rep, weights_pred, weights_in, weights_out)
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

        if i % parameters['output_delay'] == 0 or i == parameters['iterations'] - 1:

            ''' Make a prediction '''
            y_pred_batch, h_rep = model(torch.Tensor(x_batch), torch.Tensor(t_batch))
            weights_pred, weights_in, weights_out = model.get_weights()
            ''' Calculate losses'''
            # print(three_pairs_batch)
            # pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_batch),
            #                                           torch.Tensor(three_pairs_simi))
            #
            obj_loss = lossCalculator.calc_loss(t_batch, p_t, torch.Tensor(y_batch), y_pred_batch, h_rep, weights_pred, weights_in, weights_out)
            # print(t_batch)
            f_error = lossCalculator.pred_loss
            imb_err = lossCalculator.imb_loss

            cf_error = np.nan
            valid_obj = np.nan
            valid_f_error = np.nan

            # if D['HAVE_TRUTH']:
            #     y_pred_cf, _ = model(torch.Tensor(x_cf), torch.Tensor(t_cf))
            #     pddm_loss, mid_loss = model.pddm_mid_loss(torch.Tensor(three_pairs_train),
            #                                                    torch.Tensor(three_pairs_simi_train))
            #     lossCalculator.calc_loss(t_cf, np.nan, torch.Tensor(y_cf), y_pred_cf, pddm_loss, mid_loss)
            #     cf_error = lossCalculator.pred_loss
            ''' Print and save the losses '''
            losses = []
            valid_list = []
            try:
                obj_loss_detach = obj_loss.detach()
                f_error_detach = f_error.detach()
                cf_error_detach = cf_error.detach()
                imb_err_detach = imb_err.detach()
                valid_obj_detach = valid_obj.detach()
                losses.append([obj_loss_detach, f_error_detach, cf_error, imb_err_detach])
                loss_str = str(i) + '\tObj: %.4g,\tF: %.4g,\tCf: %.4g,\tImb: %.4g' % (obj_loss_detach, f_error_detach, cf_error, imb_err_detach)
            except AttributeError:
                print("Loss Has NaN")
                losses.append([obj_loss, f_error, cf_error, imb_err])
                loss_str = str(i) + '\tObj: %.4g,\tF: %.4g,\tCf: %.4g,\tImb: %.4g' % (obj_loss, f_error, cf_error, imb_err)
            # print losses
            valid_list.append(valid_f_error)
            if parameters['loss'] == 'log':
                y_pred_batch, h_rep = model(torch.Tensor(x_batch), torch.Tensor(t_batch))
                y_pred_batch = 1.0 * (y_pred_batch > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred_batch.numpy())))
                loss_str += ',\tAcc: %.2f%%' % acc

            print(loss_str)
            # log(logfile, loss_str)
            #
            # if torch.isnan(obj_loss):
            #     log(logfile, 'Objective is NaN. Skipping.')
            #     objnan = True

    return model


def site_predict(model, x, t):
    y_pred, _ = model(x, t)
    return y_pred.detach().numpy()