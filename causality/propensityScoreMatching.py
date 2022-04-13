from os import close
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression



class PropensityScoreMatching():
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
        self.y0 = None
        self.y1 = None
        self.sk_model = None
        self.ite_pred = None
        self.ite_true = None
        self.epsilon = 10**-10



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


    def match_control_with_treated(self, control = 0, treated = 1, n_neighbors = 1, y0='y0', y1='y1'):
        """
        match each instance from the control group with a certain number of instances from the treated group

        :param control: the treatment value that signifies the control group
        :param treated: the treatment value that signifies the treated group
        :variable: the variable of interests
        :n_neighbors: the number of treated group instances that will be mapped to each control group instance
        """
        if self.control_df is None:
            self.control_df = self.df[self.df[self.treatment] == control].copy()
            self.control_df.reset_index()
        if self.treated_df is None:
            self.treated_df = self.df[self.df[self.treatment] == treated].copy()
            self.treated_df.reset_index()

        self.control_knn = NearestNeighbors(metric = 'euclidean', n_neighbors=n_neighbors)
        self.control_knn.fit(self.treated_df[['propensity score']].values)
        diff = []
        arr = []

        self.y0 = y0
        self.y1 = y1
        
        for __, row in self.control_df.iterrows():
            dist = self.control_knn.kneighbors([[row['propensity score']]])[0].flatten().flatten()[0]
            closest_value_low = row['propensity score'] - dist - self.epsilon
            closest_value_high = row['propensity score'] + dist + self.epsilon
            matches = self.treated_df[(self.treated_df['propensity score'] >= closest_value_low) & (self.treated_df['propensity score'] <= closest_value_high)].sample(n_neighbors)
            match_index = matches.index.array
            value_of_interest = matches[y1].values
            # if more than one neighbors are required, calculate the mean of all the neighbors
            value_of_interest = np.mean(value_of_interest)
            diff.append(value_of_interest - row[y0])
            arr.append(match_index)
        self.control_df['treated-control'] = diff
        self.control_df['match index'] = arr


    def match_treated_with_control(self, control = 0, treated = 1, n_neighbors = 1, y0='y0', y1='y1'):
        """
        match each instance from the treated group with a certain number of instances from the control group

        :param control: the treatment value that signifies the control group
        :param treated: the treatment value that signifies the treated group
        :variable: the variable of interests
        :n_neighbors: the number of control group instances that will be mapped to each treated group instance
        """
        if self.control_df is None:
            self.control_df = self.df[self.df[self.treatment] == control].copy()
            self.control_df.reset_index()
        if self.treated_df is None:
            self.treated_df = self.df[self.df[self.treatment] == treated].copy()
            self.treated_df.reset_index()

        self.treated_knn = NearestNeighbors(metric = 'euclidean', n_neighbors=n_neighbors)
        self.treated_knn.fit(self.control_df[['propensity score']].values)
        diff = []
        arr = []

        self.y0 = y0
        self.y1 = y1

        for __, row in self.treated_df.iterrows():
            dist = max(self.treated_knn.kneighbors([[row['propensity score']]])[0].flatten())
            closest_value_low = row['propensity score'] - dist - self.epsilon
            closest_value_high = row['propensity score'] + dist + self.epsilon
            matches = self.control_df[(self.control_df['propensity score'] >= closest_value_low) & (self.control_df['propensity score'] <= closest_value_high)].sample(n_neighbors)
            match_index = matches.index.array
            value_of_interest = matches[y0].values
            # if more than one neighbors are required, calculate the mean of all the neighbors
            value_of_interest = np.mean(value_of_interest)
            diff.append(row[y1] - value_of_interest)
            arr.append(match_index)
        self.treated_df['treated-control'] = diff
        self.treated_df['match index'] = arr

    def calculate_ite(self):
        self.ite_pred = np.append(self.control_df['treated-control'].values,self.treated_df['treated-control'].values)
        self.ite_true = np.append(np.subtract(self.control_df[self.y1], self.control_df[self.y0]), np.subtract(self.treated_df[self.y1], self.treated_df[self.y0]))

    def get_ite(self):
        self.calculate_ite()
        return self.ite_pred, self.ite_true

    def get_control_with_treated(self):
        return self.control_df

    def get_treated_with_control(self):
        return self.treated_df

    # added for psm result comparison; should be deleted later
    def get_ate_true(self):
        return np.mean(self.ite_true)