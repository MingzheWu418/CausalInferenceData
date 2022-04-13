
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from models_pytorch import *
from torch.utils.data import TensorDataset, DataLoader

# import math
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def train_and_predict_dragons(t, y_unscaled, x, y_cf, mu_0, mu_1, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64):
    verbose = 0
    
    # scale data
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)

    # y_mean = np.mean(y_unscaled)
    # y_std = np.std(y_unscaled)
    # y = (y_unscaled - y_mean) / y_std

    train_outputs = []
    test_outputs = []
    if dragon == 'dragonnet':
        print("I am here making dragonnet")
        dragonnet = Gragonnet(x.shape[1])

    if targeted_regularization:
        criterion = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        criterion = knob_loss

    i = 0
    np.random.seed(i)
    
    # split the dataset 
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)

    index_train_val = train_index
    x_train_val = torch.Tensor(x[train_index])
    y_train_val = y[train_index]
    t_train_val = t[train_index]
    y_cf_train_val = y_cf[train_index]
    mu0_train_val = mu_0[train_index]
    mu1_train_val = mu_1[train_index]


    train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]
    x_validation = x[validation_index]
    y_validation = y[validation_index]
    t_validation = t[validation_index]


    yt_train = np.concatenate([y_train, t_train], 1)
    yt_validation = np.concatenate([y_validation, t_validation], 1)


    tensor_x = torch.Tensor(x_train)
    tensor_yt = torch.Tensor(yt_train)
    tensor_x_val = torch.Tensor(x_validation)
    tensor_y_val = torch.Tensor(yt_validation)

    # training dataset
    dataset_train = TensorDataset(tensor_x, tensor_yt)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size)

    # validation dataset
    dataset_validation = TensorDataset(tensor_x_val, tensor_y_val)
    dataloader_validation = DataLoader(dataset_validation, batch_size = batch_size)

    # testing dataset
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)
    tensor_t_test = torch.Tensor(t_test)

    y_cf_test = y_cf[test_index]
    mu0_test = mu_0[test_index]
    mu1_test = mu_1[test_index]


    # Adam optimizer, weight decay corresponds to l2; l2 is only applied to y calculation in tf model
    optimizer = optim.Adam(dragonnet.parameters(), lr=1e-3, weight_decay=0.01)

    # implement reduce lr on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                patience = 5, factor=0.5, cooldown=0, min_lr=0, threshold=1e-8)


    # variables used to detect early stopping
    last_loss_1 = 1000
    trigger_times_1 = 0


    # first training loop
    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        binary_accuracy = 0.0
  
        dragonnet.train()
        # target is a list of y-and-t value pair
        from scipy.stats import logistic
        for i, (inputs, target) in enumerate(dataloader_train):
            optimizer.zero_grad()
            yhat = dragonnet(inputs)
            loss = criterion(target, yhat)
            loss.backward()
            binary_accuracy += treatment_accuracy(target, yhat)
            optimizer.step()
            running_loss += loss.item()

        # print binary accuracy
        binary_accuracy = 100*binary_accuracy/x_train.shape[0]
        print('binary accuracy for epoch',epoch,' is:',binary_accuracy)
        print('running loss for epoch',epoch,' is:',running_loss/(i+1))
    
        #validation for early stopping
        dragonnet.eval()
        loss_total = 0
        validation_patience = 2

        for i, (inputs, target) in enumerate(dataloader_validation):
            optimizer.zero_grad()
            yhat = dragonnet(inputs)
            loss = criterion(target, yhat)
            loss_total += loss.item()

        if loss_total+1 > last_loss_1:
            trigger_times_1 +=1

            if trigger_times_1 >= validation_patience:
                print('early stopping at epoch',epoch)
                break
        else:
            trigger_times_1 = 0

        last_loss_1 = loss_total
        scheduler.step(loss)
        


    # second training loop
    last_loss_2 = 1000
    trigger_times_2 = 0
    epochs = 300

    optimizer_2 = optim.SGD(dragonnet.parameters(), lr=1e-5, momentum = 0.9, weight_decay=0.01, nesterov=True)
    scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='min',patience = 5, factor=0.5, cooldown=0, min_lr=0, threshold=0)

    for epoch in range(epochs):
        #training 
        binary_accuracy = 0.0
        running_loss = 0.0
        dragonnet.train()
     
        for i, (inputs, target) in enumerate(dataloader_train):
            optimizer_2.zero_grad()
            yhat = dragonnet(inputs)
            loss = criterion(target, yhat)
            loss.backward()
            optimizer_2.step()
            binary_accuracy += treatment_accuracy(target, yhat)
            running_loss += loss.item()

        # print binary accuracy
        binary_accuracy = 100*binary_accuracy/x_train.shape[0]
        print('second training loop binary accuracy for epoch',epoch,' is:',binary_accuracy)
        print('second training loop running loss for epoch',epoch,' is:',running_loss/(i+1))
    
        #validation for early stopping
        dragonnet.eval()
        loss_total = 0
        validation_patience = 40

        for i, (inputs, target) in enumerate(dataloader_validation):
            optimizer_2.zero_grad()
            yhat = dragonnet(inputs)
            loss = criterion(target, yhat)
            loss_total += loss.item()

        if loss_total+1 > last_loss_2:
            trigger_times_2 +=1

            if trigger_times_2 >= validation_patience:
                print('second training loop early stopping at epoch',epoch)
                break
        else:
            trigger_times_2 = 0

        last_loss_2 = loss_total
        scheduler_2.step(loss)




    propensity_score = []
    y0_list = []
    y1_list = []
    t_pred = []
    eps = []

    # evaluate training + validation dataset
    for i, inputs in enumerate(x_train_val):
        y_pred = dragonnet(inputs[None,:])
        y_pred = y_pred.detach().numpy()[0]

        propensity_score.append(y_pred[2])
        y0_list.append(y_pred[0])
        y1_list.append(y_pred[1])
        eps.append(y_pred[3])

    ps_treated = np.mean((np.asarray(propensity_score))[np.asarray(t_train_val).squeeze()==1.])
    ps_control = np.mean((np.asarray(propensity_score))[np.asarray(t_train_val).squeeze()==0.])
 
    # store results in the same format as the original model
    test_outputs = [{'q_t0': y0_list, 'q_t1': y1_list, 'g': propensity_score, 't': t_train_val, 'y': y_train_val, 'x': x_train_val, 'index': index_train_val, 'eps': eps}]

    #calculate epsilon ate
    ite_true = np.subtract(mu1_train_val, mu0_train_val)
    ite_pred = np.subtract(y1_list, y0_list)

    assert len(ite_pred) == len(ite_pred)
    epsilon_ate = np.absolute(np.mean(ite_pred - ite_true))


    #calculate pehe
    pehe = np.mean(np.power((ite_pred - ite_true), 2))

    #print out results
    print('here is the evaluation results for training+validation dataset:')
    print('epsilon ate:',epsilon_ate)
    print('pehe:',pehe)
    print('ps for treated and control:',ps_treated,ps_control)
    print('')



    # evaludate testing dataset
    propensity_score = []
    y0_list = []
    y1_list = []
    t_pred = []
    eps = []
    for i, inputs in enumerate(tensor_x_test):
        y_pred = dragonnet(inputs[None,:])
        y_pred = y_pred.detach().numpy()[0]

        # yt_hat_test.append(y_pred)
        propensity_score.append(y_pred[2])
        y0_list.append(y_pred[0])
        y1_list.append(y_pred[1])
        eps.append(y_pred[3])

    ps_treated = np.mean((np.asarray(propensity_score))[np.asarray(tensor_t_test).squeeze()==1.])
    ps_control = np.mean((np.asarray(propensity_score))[np.asarray(tensor_t_test).squeeze()==0.])
 
    # store results in the same format as the original model
    test_outputs = [{'q_t0': y0_list, 'q_t1': y1_list, 'g': propensity_score, 't': t_test, 'y': y_test, 'x': x_test, 'index': test_index, 'eps': eps}]

    #calculate epsilon ate
    ite_true = np.subtract(mu1_test, mu0_test)
    ite_pred = np.subtract(y1_list, y0_list)

    assert len(ite_pred) == len(ite_pred)
    epsilon_ate = np.absolute(np.mean(ite_pred - ite_true))


    #calculate pehe
    pehe = np.mean(np.power((ite_pred - ite_true), 2))

    #print out results
    print('here is the evaluation results for test dataset:')
    print('epsilon ate:',epsilon_ate)
    print('pehe:',pehe)
    print('ps for treated and control:',ps_treated,ps_control)
    print('')

    return test_outputs, train_outputs
       


def run_ihdp(data_base_dir='/Users/claudiashi/data/ihdp_csv', output_dir='~/result/ihdp/',
             knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon=''):
    print("the dragon is {}".format(dragon))

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))

    for idx, simulation_file in enumerate(simulation_files):
        print('currently running file',idx, simulation_file)

        simulation_output_dir = os.path.join(output_dir, str(idx))

        os.makedirs(simulation_output_dir, exist_ok=True)

        x = load_and_format_covariates_ihdp(simulation_file)
        t, y, y_cf, mu_0, mu_1 = load_all_other_crap(simulation_file)
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)

        for is_targeted_regularization in [True, False]:
            print("Is targeted regularization: {}".format(is_targeted_regularization))
            if dragon == 'nednet':
                test_outputs, train_output = train_and_predict_ned(t, y, x,
                                                                   targeted_regularization=is_targeted_regularization,
                                                                   output_dir=simulation_output_dir,
                                                                   knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                                   val_split=0.2, batch_size=64)
            else:

                test_outputs, train_output = train_and_predict_dragons(t, y, x, y_cf, mu_0, mu_1,
                                                                       targeted_regularization=is_targeted_regularization,
                                                                       output_dir=simulation_output_dir,
                                                                       knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                                       val_split=0.2, batch_size=64)

            if is_targeted_regularization:
                train_output_dir = os.path.join(simulation_output_dir, "targeted_regularization")
            else:
                train_output_dir = os.path.join(simulation_output_dir, "baseline")
            os.makedirs(train_output_dir, exist_ok=True)

            # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                    **output)

            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                    **output)


def turn_knob(data_base_dir='/Users/claudiashi/data/test/', knob='dragonnet',
              output_base_dir=''):
    output_dir = os.path.join(output_base_dir, knob)

    if knob == 'dragonnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='dragonnet')

    if knob == 'tarnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='tarnet')

    if knob == 'nednet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='nednet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD")
    parser.add_argument('--knob', type=str, default='early_stopping',
                        help="dragonnet or tarnet or nednet")

    parser.add_argument('--output_base_dir', type=str, help="directory to save the output")

    args = parser.parse_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir)


if __name__ == '__main__':
    main()
