# import ast
import csv

import pandas as pd
import numpy as np
import json

import sklearn
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.special import expit


# Initial approach:
class format_processor:

    def __init__(self):
        """
        Create a data loader
        """

        # self.data typically contains the following information,
        # where t is the observed treatment,
        # yf the factual outcome, ycf the counter factual
        # and x is the other features
        self.data = {"t": [], "yf": [], "ycf": [], "x": []}
        self.dataset = ""
        self.filename = ""

    def load_data(self, dataset):
        """
        Load data of a particular dataset
        :param dataset: the name of the dataset, case insensitive
        :return: the dataset loaded
        """

        name = dataset.upper()
        assert name in ["TWINS", "IHDP", "JOBS"]
        self.dataset = name
        self.filename = "./raw_data/" + name + "/"

        # Have to hard code each dataset, given each dataset has different structure
        if self.dataset == "TWINS":
            self.data = self.__load_TWINS()
        elif self.dataset == "IHDP":
            self.data = self.__load_IHDP()
            # print(self.data)
        elif self.dataset == "JOBS":
            self.data = self.__load_JOBS()
        else:
            pass
        return self.data

    def data_simulate(self, mode):
        pass

    def __load_TWINS(self):
        """
        Private helper to load TWINS.
        Notice controlled group (T == 0) is always saved in the first half of an array,
        and treatment group (T == 1) always saved in the last half of an array.

        :return: the data loaded
        """
        data = {}

        """Load twins data."""

        # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
        ori_data = np.loadtxt("./raw_data/TWINS/Twin_data.csv", delimiter=",", skiprows=1)
        # Define features
        x = ori_data[:, :30]
        no, dim = x.shape

        # Define potential outcomes
        potential_y = ori_data[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        potential_y = np.array(potential_y < 9999, dtype=float)
        print(potential_y)

        ## Assign treatment
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])

        ## Define observable outcomes
        # y = np.zeros([no,1])
        y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
        ycf = np.transpose(1 - t) * potential_y[:, 1] + np.transpose(t) * potential_y[:, 0]
        y = np.reshape(np.transpose(y), [no, ])
        ycf = np.reshape(np.transpose(ycf), [no, ])
        data["t"] = t
        data["yf"] = y
        data["ycf"] = ycf
        data["x"] = x
        # data["y_groundtruth"] = potential_y

        """
        # This part served as keeping track of the type of each variable:
        # whether it is binary, categorical, or numerical.
        
        file = open(self.filename + "covar_type.txt", "r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        data["label_type"] = dictionary
        """
        return data

    def __load_IHDP(self):
        """
        Private helper to load IHDP
        :return: the data loaded
        """

        # Read in raw_data

        data_train = np.load(self.filename + "ihdp_npci_1-100.train.npz")
        data_test = np.load(self.filename + "ihdp_npci_1-100.test.npz")
        data = {'x': np.concatenate((data_train['x'], data_test['x']), axis=0).reshape(-1, data_train['x'].shape[1]),
                't': np.concatenate((data_train['t'], data_test['t']), axis=0).flatten(),
                'yf': np.concatenate((data_train['yf'], data_test['yf']), axis=0).flatten()}
        try:
            data['ycf'] = np.concatenate((data_train['ycf'], data_test['ycf']), axis=0).flatten()
        except:
            data['ycf'] = None
        return data

    def __load_JOBS(self):
        """
        Private helper to load IHDP
        :return: the data loaded
        """

        # Read in raw_data

        data_train = np.load(self.filename + "jobs_DW_bin.train.npz")
        data_test = np.load(self.filename + "jobs_DW_bin.test.npz")
        data = {'x': np.concatenate((data_train['x'], data_test['x']), axis=0).reshape(-1, data_train['x'].shape[1]),
                't': np.concatenate((data_train['t'], data_test['t']), axis=0).flatten(),
                'yf': np.concatenate((data_train['yf'], data_test['yf']), axis=0).flatten()}
        try:
            data['ycf'] = np.concatenate((data_train['ycf'], data_test['ycf']), axis=0).flatten()
        except:
            data['ycf'] = None
        return data


if __name__ == "__main__":
    loader = format_processor()
    # loader.load_data("IHDP")
    # dct = loader.data
    # np.savez("./datasets/ihdp_npci_1-100", **dct)
    loader.load_data("JOBS")
    dct = loader.data
    np.savez("./datasets/jobs_DW_bin", **dct)
    # loader.load_data("TWINS")
    # dct = loader.data
    # np.savez("./datasets/twins", **dct)
    # for key, value in dct.items():
    #     print(key, value.shape)
    # print(dct)

    # reshape TWINS into batches
    # dct_train = {}
    # dct_test = {}
    # idx = np.random.permutation(1140)
    # train_rate = 0.9
    # train_idx = idx[:int(train_rate * 1140)]
    # test_idx = idx[int(train_rate * 1140):]
    # for key, item in loader.data.items():
    #     if key == "x":
    #
    #         dct_train[key] = item.reshape(-1, 30, 10)[train_idx,:,:].reshape(-1,30)
    #
    #         dct_test[key] = item.reshape(-1, 30, 10)[test_idx,:,:].reshape(-1,30)
    #     else:
    #         dct_train[key] = item.reshape(-1, 10)[train_idx,:].reshape(-1,1)
    #         dct_test[key] = item.reshape(-1, 10)[test_idx,:].reshape(-1,1)
    #     # print("-----")
    #     print(key, dct[key].shape)
    #     print(key, dct_train[key].shape)
    #     print(key, dct_test[key].shape)

    # dct_train = {}
    # dct_test = {}
    # no = dct["x"].shape[0]
    # idx = np.random.permutation(no)
    # train_rate = 0.8
    # train_idx = idx[:int(train_rate * no)]
    # test_idx = idx[int(train_rate * no):]
    #
    # # reshape TWINS into batches
    # for key, item in loader.data.items():
    #     if key == "x":
    #         dct_train[key] = item[train_idx,:].reshape(-1, 30, 1)
    #
    #         dct_test[key] = item[test_idx,:].reshape(-1, 30, 1)
    #     else:
    #         dct_train[key] = item[train_idx].reshape(-1, 1)
    #         dct_test[key] = item[test_idx].reshape(-1,1)
    #     # print("-----")
    #     print(key, dct[key].shape)
    #     print(key, dct_train[key].shape)
    #     print(key, dct_test[key].shape)

    # reshape ihdp into batches
    # dct_train = {}
    # dct_test = {}
    # for key, item in loader.data.items():
    #     if key == "x":
    #         dct_train[key] = item.reshape(-1, 25, num_batch)[:,:,:80]
    #         dct_test[key] = item.reshape(-1, 25, num_batch)[:,:,80:]
    #     else:
    #         dct_train[key] = item.reshape(-1, num_batch)[:,:80]
    #         dct_test[key] = item.reshape(-1, num_batch)[:,80:]
    #     # print("-----")
    #     print(key, dct[key].shape)
    #     print(key, dct_train[key].shape)
    #     print(key, dct_test[key].shape)

    # np.savez("./raw_data/TWINS/twins_test", **dct_test)
    # x = np.load("../raw_data/JOBS/jobs_DW_bin.train.npz")
    # x = np.load("../raw_data/TWINS/twins_test_preprocessed.npz")
    # for key in x:
    #     print(key, x[key].shape)
