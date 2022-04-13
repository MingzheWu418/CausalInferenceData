from weakref import ref
import numpy as np
import pandas as pd
from inverseProbabilityWeighting import InverseProbabilityWeighting
from propensityScoreMatching import PropensityScoreMatching
from doublyRobustEstimation import DoublyRobustEstimation
from tlearner import Tlearner
from xlearner import Xlearner
from slearner import Slearner
from metric import Metric
# from parametric import PropensityScoreMatching as ref_psm
''' Currently cannot perform ref_psm b.c. missing a file causality.util
    Enable this part when this issue is solved'''

if __name__ == "__main__":
    # N: number of instance
    N = 100
    # d: feature dimension
    d = 25
    X = np.random.normal(0, 1, size=(N, d))
    # print(X)
    W_t = np.random.normal(0, 1, size=(d, 1))
    T = np.matmul(X, W_t)
    T = np.divide(1, 1+np.exp(np.multiply(-1, T)))
    assert T.shape == (N, 1)

    T = T >= 0.5
    T = [int(ele) for ele in T]

    W0 = np.random.normal(0, 1, size=(d, 1))
    W1 = np.random.normal(0, 1, size=(d, 1))
    y0 = np.dot(X, W0)
    y1 = np.dot(X, W1)
    assert y0.shape == (N, 1)

    covariate_names = ['covariate'+str(n) for n in range(d)]
    cov = pd.DataFrame(X, columns = covariate_names)

    df = pd.DataFrame(
        {'treatment': T,
        'y0': y0.flatten(),
        'y1': y1.flatten()
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



