from os import close
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression


class DoublyRobustEstimation():
    def __init__(self):
        """
        covariates: a list of variables names that are covariates
        df: the pandas dataframe of data
        logit_model: the logit model used to calculate propensity score
        treatment: how the treatment group is named in the dataset
        knn: the k-neighbor model used during matching
        control_df: the control group data
        treated_df: the treated group data
        """
        self.covariates = []
        self.df = None
        self.logit_model = None
        self.treatment = None
        self.control_knn = None
        self.treated_knn = None
        self.control_df = None
        self.treated_df = None
        self.y0 = 'y0'
        self.y1 = 'y1'
        self.sk_model = None
        self.ite_pred = None
        self.ite_true = None
        self.control = 0
        self.treated = 1
        

    def calculateScore(self, dataset, covariates, treatment='treatment'):
        """
        Calculate the propensity score for each instance.

        :param dataset: the pandas dataframe of data
        :param covariates: a list of variables names that are covariates
        :param treatment: how the treatment group is named in the dataset
        """
        self.covariates = covariates
        self.df = dataset
        self.treatment = treatment
        assert self.treatment in self.df.columns

        #  # using logit
        # logit = Logit(self.df[self.treatment], self.df[self.covariates])
        # self.logit_model = logit.fit(method = 'bfgs')
        # self.df.loc[:, 'propensity score'] = self.logit_model.predict(self.df[self.covariates])

        # using sk-learn
        self.sk_model = LogisticRegression()
        self.sk_model.fit(self.df[self.covariates], self.df[self.treatment])
        self.df.loc[:, 'propensity score'] = [prob[1] for prob in self.sk_model.predict_proba(self.df[self.covariates])]
        self.df.loc[:, 'potential outcome'] = self.sk_model.predict(self.df[self.covariates])

    def get_ate(self):
        treated_group = self.df[self.df[self.treatment] == self.treated].copy()
        control_group = self.df[self.df[self.treatment] == self.control].copy()
        treated_1 = np.divide(np.subtract(treated_group[self.y1].values, np.multiply(treated_group['potential outcome'], (1-treated_group['propensity score']))), treated_group['propensity score'])
        treated_0 = np.divide(np.subtract(0, np.multiply(control_group['potential outcome'], (0-control_group['propensity score']))), control_group['propensity score'])
        treated = np.append(treated_1, treated_0)

        control_1 = np.divide(np.subtract(0, np.multiply(treated_group['potential outcome'], (1-treated_group['propensity score']))), np.subtract(1, treated_group['propensity score']))
        control_0 = np.divide(np.subtract(control_group[self.y0].values, np.multiply(control_group['potential outcome'], (0-control_group['propensity score']))), np.subtract(1, control_group['propensity score']))
        control = np.append(control_1, control_0)

        ate_pred = np.mean(treated) - np.mean(control)
        ate_true = np.mean(np.subtract(self.df[self.y1], self.df[self.y0]))

        return ate_pred, ate_true

    def get_control_with_treated(self):
        return self.control_df

    def get_treated_with_control(self):
        return self.treated_df

