from SITE.simi_ite.propensity import *
import numpy as np
import sys
import torch



''' Original version '''
# def main(dataform, propensity_dir):
#     load_data = np.load(dataform)
#     i_exp = 1
#     treatment = load_data['t'][:,i_exp-1:i_exp]
#     data = {'x': load_data['x'][:,:,i_exp-1], 't':treatment }
#
#     propensity_score, clf = propensity_score_training(data['x'], data['t'], 'Logistic-regression')
#     filename = propensity_dir
#     torch.save(clf, open(filename, 'wb'))
#
#
# if __name__ == "__main__":
#     main(sys.argv[1], sys.argv[2])


def main(dataform, propensity_dir):
    load_data = np.load(dataform)
    # i_exp = 1
    length = len(load_data['t'])
    treatment = load_data['t'][:length//10]
    data = {'x': load_data['x'][:length//10, :], 't': treatment }
    propensity_score, clf = propensity_score_training(data['x'], data['t'], 'SVM')
    filename = propensity_dir
    torch.save(clf, open(filename, 'wb'))


if __name__ == "__main__":
    main("../datasets/acic.npz", "./propensity_score/acic_propensity_model.sav")
    # main(sys.argv[1], sys.argv[2])