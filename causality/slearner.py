from os import close
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class Slearner():
    def __init__(self, base_model = 'Linear Regression'):
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
        self.epsilon = 10**-10
        self.base_model = base_model



    def calculateScore(self, dataset, covariates, treatment='treatment', control = 0, treated = 1):
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

        # using sk-learn
        if self.base_model == 'Linear Regression':
            self.sk_model = LinearRegression()

        elif self.base_model == 'svm':
            self.sk_model = SVR()

        self.df.loc[:, 'y obs'] = np.where(self.df[self.treatment], self.df[self.y1], self.df[self.y0])
        self.sk_model.fit(self.df[[*self.covariates, self.treatment]], self.df['y obs'])

        self.df.loc[:, 'constant'] = 0
        self.df.loc[:, 'miu0'] = list(self.sk_model.predict(np.array(self.df[[*self.covariates, 'constant']])))
        self.df.loc[:, 'constant'] = 1
        self.df.loc[:, 'miu1'] = list(self.sk_model.predict(np.array(self.df[[*self.covariates, 'constant']])))
    

    def calculate_ite(self):
        self.ite_pred = np.subtract(self.df['miu1'], self.df['miu0'])
        self.ite_true = np.subtract(self.df[self.y1], self.df[self.y0])

    def get_ite(self):
        self.calculate_ite()
        return self.ite_pred, self.ite_true

    def get_control_with_treated(self):
        return self.control_df

    def get_treated_with_control(self):
        return self.treated_df

