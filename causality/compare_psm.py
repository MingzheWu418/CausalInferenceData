
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from propensityScoreMatching import PropensityScoreMatching
import matplotlib.pyplot as plt

from metric import Metric
from parametric import PropensityScoreMatching as ref_psm

# from causalityref.estimation.parametric. import PropensityScoreMatching as ref_psm
# import sys
# # sys.path.append('../')
# sys.path.append('~/Desktop/UVA/4.1/causalityref/causality/estimation')
# import parametric
# from parametric import  PropensityScoreMatching as ref_psm
# sys.path.insert(1,' ~/Desktop/UVA/4.1/causalityref/causality/estimation')
# from parametric import PropensityScoreMatching as ref_psm

if __name__ == "__main__":
    for j in range(2):
        mine = []
        ref = []
        
        for i in range(500):
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
            psm_model = PropensityScoreMatching()
            psm_model.calculateScore(df, covariate_names)
            psm_model.match_control_with_treated()
            psm_model.match_treated_with_control()
            metric = Metric()
            epsilon_ate = metric.get_epsilon_ate(*psm_model.get_ite())
            ate_true = psm_model.get_ate_true()
            mine.append(epsilon_ate)
    

            # compare with other psm model
            matcher = ref_psm()
            df.loc[:, 'y obs'] = np.where(df['treatment'], df['y1'], df['y0'])
            cov = {}
            for c in covariate_names:
                cov[c] = 'c'
            ate_pred = matcher.estimate_ATE(df, 'treatment', 'y obs', cov, n_neighbors=1)
            ref_epsilon_ate = np.absolute(ate_pred - ate_true)
            ref.append(ref_epsilon_ate)

            # print('reference result:', matcher.estimate_ATE(df, 'treatment', 'y obs', cov))
        ans = ttest_ind(mine,ref)
        mine_mean = round(np.mean(mine),4)
        mine_var =  round(np.var(mine),4)
        ref_mean =  round(np.mean(ref),4)
        ref_var =  round(np.var(ref),4)
        # print('min of mine:', min(mine))
        # print('min of ref:',min(ref))
        # fig, ax = plt.subplots()
        # ax.boxplot([mine, ref])
        # plt.savefig('try.png')

        print(ans)
        f = open('psm_comp_correction.txt', 'a')
        # f.write('sample:'+str(500))
        f.write('\nt statistics:'+str(round(ans.statistic,4)))
        f.write('\np value:'+str(round(ans.pvalue,4)))
        f.write('\nmy psm mean:'+str(mine_mean)+', var:'+str(mine_var))
        f.write('\nreference psm mean:'+str(ref_mean)+', var:'+str(ref_var))
        f.write('\n')
        f.close()
