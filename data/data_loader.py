import random

import numpy as np

from utils.utils_pytorch import comb_potential_outcome


def cross_val(dataset, folds):
    no = dataset["n"]
    assert no % folds == 0
    idx = np.arange(no)
    np.random.shuffle(idx)
    idx = idx.reshape(folds, -1)
    list_ds = []
    for val_index in idx:
        train_index = np.setdiff1d(np.arange(no), val_index)
        # print(train_index.shape)
        # print(val_index.shape)
        train_d = {}
        val_d = {}
        for key, item in dataset.items():
            try:
                # print(key, np.mean(item))
                train_d[key] = dataset[key][train_index]
                val_d[key] = dataset[key][val_index]
            except TypeError:
                # print(key, item)
                if key == 'n':
                    train_d[key] = train_index.shape[0]
                    val_d[key] = val_index.shape[0]
                else:
                    train_d[key] = dataset[key]
                    val_d[key] = dataset[key]
        list_ds.append((train_d, val_d))
    return list_ds


def cross_val_index(dataset, folds):
    no = dataset["n"]
    if no % folds == 0:
        idx = np.arange(no)
    else:
        idx = np.arange(no - no % folds)
    np.random.shuffle(idx)
    idx = idx.reshape(folds, -1)
    return idx


def split(dataset, train_rate):
    train_d = {}
    test_d = {}

    train_rate = train_rate
    no = dataset["n"]
    idx = np.random.permutation(no) #length n
    train_idx = idx[:int(train_rate * no)]
    test_idx = idx[int(train_rate * no):]

    # print("-----")
    for key, item in dataset.items():
        try:
            # print(key, np.mean(item))
            train_d[key] = dataset[key][train_idx]
            test_d[key] = dataset[key][test_idx]
        except TypeError:
            if key == 'n':
                train_d[key] = train_idx.shape[0]
                test_d[key] = test_idx.shape[0]
            # print(key, item)
            train_d[key] = dataset[key]
            test_d[key] = dataset[key]
    # self.data[""]

    # print("-----")
    # for key, item in train_d.items():
    #     try:
    #         print(key, np.mean(item))
    #     except:
    #         print(key)
    # print("-----")
    #
    # for key, item in test_d.items():
    #     try:
    #         print(key, np.mean(item))
    #     except:
    #         print(key)
    # print("-----")

    train_d["n"] = len(train_idx)
    test_d["n"] = len(test_idx)
    return train_d, test_d


def load_batch(dataset, batch_size=1):
    result = {}

    batch_num = dataset['n'] // batch_size
    for key, value in dataset.items():
        if type(value) == int or type(value) == bool:
            result[key] = value
        else:
            try:
                result[key] = dataset[key][:batch_size * batch_num, :] \
                    .reshape(-1, dataset["dim"], batch_size)
            except:
                result[key] = dataset[key][:batch_size * batch_num].reshape(-1, batch_size)
    result['n'] = result['n']//100
    return result


def load_data(filename):
    """ Load data set """
    if filename[-3:] == 'npz':
        data_in = np.load(filename)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            print("Counterfactual not available")
            data['ycf'] = None

        try:
            data['mu0'] = data_in['mu0']
            data['mu1'] = data_in['mu1']
        except:
            print("No MU available")
            mu0s = None
            mu1s = None

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data


if __name__ == "__main__":
    seed = 42
    batch_size = 11
    random.seed(seed)
    np.random.seed(seed)
    fname = "./datasets/twins.npz"
    d = load_data(fname)
    train_dataset, test_dataset = split(d, 0.8)
    train_batch = load_batch(train_dataset, batch_size)
    print(comb_potential_outcome(d['yf'], d['ycf'], d['t']))
