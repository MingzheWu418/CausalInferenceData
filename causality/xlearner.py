from os import close
from numpy.lib.function_base import _calculate_shapes
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class Xlearner():
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
        self.sk_model_control = None
        self.sk_model_treated = None
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

        self.control_df = self.df[self.df[self.treatment] == control].copy()
        self.control_df.reset_index()
        self.treated_df = self.df[self.df[self.treatment] == treated].copy()
        self.treated_df.reset_index()

        # using sk-learn
        if self.base_model == 'Linear Regression':
            self.sk_model_control = LinearRegression()
            self.sk_model_treated = LinearRegression()

        elif self.base_model == 'svm':
            self.sk_model_control = SVR()
            self.sk_model_treated = SVR()

        self.sk_model_control.fit(self.control_df[self.covariates], self.control_df[self.y0])
        self.sk_model_treated.fit(self.treated_df[self.covariates], self.treated_df[self.y1])
       
        self.df.loc[:, 'miu0'] = list(self.sk_model_control.predict(self.df[self.covariates]))
        self.df.loc[:, 'miu1'] = list(self.sk_model_treated.predict(self.df[self.covariates]))

    def calculate_imputed_treatment_effect(self, tao_model = 'Linear Regression', control = 0, treated = 1):
        D1 = np.subtract(self.treated_df[self.y1], self.df[self.df[self.treatment] == treated]['miu0'])
        D0 = np.subtract(self.df[self.df[self.treatment] == control]['miu1'], self.control_df[self.y0])
        
        if tao_model == 'Linear Regression':
            self.tao_model_control = LinearRegression()
            self.tao_model_treated = LinearRegression()
        elif tao_model == 'svm':
            self.tao_model_control = SVR()
            self.tao_model_treated = SVR()

        self.tao_model_control.fit(self.control_df[self.covariates], D0)
        self.tao_model_treated.fit(self.treated_df[self.covariates], D1)  
        self.df.loc[:, 'tao control'] = list(self.tao_model_control.predict(self.df[self.covariates]))
        self.df.loc[:, 'tao treated'] = list(self.tao_model_treated.predict(self.df[self.covariates]))


    def calculate_ite(self, tao_model = 'Linear Regression', gx = 'Logistic Regression'):
        self.calculate_imputed_treatment_effect(tao_model = tao_model)

        if gx == 'Logistic Regression':
            self.gx = LogisticRegression()
            self.gx.fit(self.df[self.covariates], self.df[self.treatment])
            self.df.loc[:, 'propensity score'] = [prob[1] for prob in self.gx.predict_proba(self.df[self.covariates])]

        self.ite_pred = np.add(np.multiply(self.df['tao control'], self.df['propensity score']), np.multiply(self.df['tao treated'], np.subtract(1, np.array(self.df['propensity score']))))
        self.ite_true = np.subtract(self.df[self.y1], self.df[self.y0])

    def get_ite(self):
        self.calculate_ite()
        return self.ite_pred, self.ite_true

    def get_control_with_treated(self):
        return self.control_df

    def get_treated_with_control(self):
        return self.treated_df

