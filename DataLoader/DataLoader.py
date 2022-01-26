# import ast

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Initial approach:
class DataHandler:

    def __init__(self):
        """
        Create a data loader
        """

        # self.data typically contains the following information,
        # where T is the observed treatment,
        # Y the factual outcome
        # Y0 and Y1 the potential outcome with T==0 or 1,
        # X being the other variables, and
        # raw indicating if each of the previous parts is raw or simulated
        self.data = {"t": [], "y": [], "y0": [], "y1": [], "x": [], "dim": [], "n" : []}
        self.dataset = ""
        self.filename = ""

    def load_data(self, dataset):
        """
        Load data of a particular dataset
        :param dataset: the name of the dataset, case insensitive
        :return: the dataset loaded
        """

        name = dataset.upper()
        assert name in ["TWINS", "IHDP"]
        self.dataset = name
        self.filename = "../datasets/" + name + "/"

        # Have to hard code each dataset, given each dataset has different structure
        if self.dataset == "TWINS":
            self.data = self.__load_TWINS()
        elif self.dataset == "IHDP":
            self.data = self.__load_IHDP()
            # print(self.data)
        else:
            pass
        return self.data

    def data_processing(self, Y_transformer, X_transformer):
        """
        Takes in two initialized transformers, and return the transformed dataset.
        :param Y_transformer: The transformer of Y, most likely provided by scikit learn.
        :param X_transformer: The transformer of X, most likely provided by scikit learn.
        :return: the preprocessed dataset
        """

        # I am assuming we only need to transform Y and X here.
        Y_transformer.fit(self.data["y"])
        self.data["y"] = Y_transformer.transform(self.data["y"])

        Y_transformer.fit(self.data["y0"])
        self.data["y0"] = Y_transformer.transform(self.data["y0"])

        Y_transformer.fit(self.data["y1"])
        self.data["y1"] = Y_transformer.transform(self.data["y1"])

        X_transformer.fit(self.data["x"])
        self.data["x"] = X_transformer.transform(self.data["x"])
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

        # Read in treatment
        t = pd.read_csv(self.filename + "twin_pairs_T_3years_samesex.csv")
        # Store T as 2Nx1 np array, and T0, T1.
        # Here T is saved as binary values,
        # while T0 and T1 contains the weights of infants
        data["t"] = np.append(np.zeros(len(t["dbirwt_0"])), np.ones(len(t["dbirwt_1"]))).reshape(-1,1)
        data["t0"] = t["dbirwt_0"].to_numpy().reshape(-1,1)
        data["t1"] = t["dbirwt_1"].to_numpy().reshape(-1,1)

        # Read in outcome
        y = pd.read_csv(self.filename + "twin_pairs_Y_3years_samesex.csv")
        # Store Y as 2Nx1 np array, and Y0, Y1
        data["y"] = np.append(y["mort_0"].to_numpy(), y["mort_1"].to_numpy()).reshape(-1,1)
        data["y0"] = y["mort_0"].to_numpy().reshape(-1,1)
        data["y1"] = y["mort_1"].to_numpy().reshape(-1,1)

        # Read in other factors
        x = pd.read_csv(self.filename + "twin_pairs_X_3years_samesex.csv")
        x = x.iloc[:, 2:]

        # 2D array, first axis is variables, second axis is instances
        # i.e. data["X"][0] returns the value of the first X variable for all patients
        x0 = x.loc[:, ~x.columns.isin(["infant_id_1", "bord_1"])]
        x1 = x.loc[:, ~x.columns.isin(["infant_id_0", "bord_0"])]
        x0 = x0.rename(columns={"infant_id_0": "infant_id", "bord_0": "bord"})
        x1 = x1.rename(columns={"infant_id_1": "infant_id", "bord_1": "bord"})
        x_concat = pd.concat([x0, x1], ignore_index=True)
        data["x"] = x_concat.to_numpy()

        # We also keep the label for each row
        data["x_label"] = x_concat.columns.to_numpy()

        """
        # This part served as keeping track of the type of each variable:
        # whether it is binary, categorical, or numerical.
        
        file = open(self.filename + "covar_type.txt", "r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        data["label_type"] = dictionary
        """

        # Have to manually input this part based on prior knowledge
        # data["raw"] = {"T": 1, "Y0": 1, "Y1": 1, "Y": 1, "X": 1}
        data["HAVE_TRUTH"] = True
        return data

    def __load_IHDP(self):
        """
        Private helper to load IHDP
        :return: the data loaded
        """
        # Initialization
        data = {}

        # Read in datasets
        dataset = pd.read_csv(self.filename + "csv/ihdp_npci_1.csv", header=None)
        for index in range(2, 11):
            dataset = pd.concat([dataset, pd.read_csv(self.filename + "csv/ihdp_npci_" + str(index) + ".csv", header=None)])
            dataset = dataset.reset_index().drop(columns=["index"])

        # Assign column names
        col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]
        for i in range(1, 26):
            col.append("x" + str(i))
        dataset.columns = col

        # Directly store T, Y, and X from the dataset
        # Also store them as Nx1 numpy array
        data["t"] = dataset["treatment"].to_numpy().reshape(-1,1)
        data["y"] = dataset["y_factual"].to_numpy().reshape(-1,1)

        # If T == 0, then y_factual is y0 and y_cfactual is y1.
        # vice versa
        y0 = []
        y1 = []
        for i in range(len(dataset["treatment"])):
            if dataset["treatment"][i] == 0:
                y0.append([dataset["y_factual"][i]])
                y1.append([dataset["y_cfactual"][i]])
            else:
                y0.append([dataset["y_cfactual"][i]])
                y1.append([dataset["y_factual"][i]])
        data["y0"] = np.asarray(y0)
        data["y1"] = np.asarray(y1)

        # numpy array, each row is one instance, each column is one variable
        # i.e. data["X"][0] returns the all variables for the 0th patient
        x = dataset.iloc[:, 5:]
        data["x"] = x.to_numpy()

        # Record the label
        data["x_label"] = np.array(col[5:])

        # Have to manually input this part based on prior knowledge
        # data["raw"] = {"T": 1, "Y0": 0, "Y1": 0, "Y": 1, "X": 1}
        data["HAVE_TRUTH"] = True
        return data


if __name__ == "__main__":
    loader = DataHandler()
    loader.load_data("TWINS")
    pipeline = Pipeline([('imputer', SimpleImputer()), ('scaler', Normalizer())])
    print(loader.data_processing(Y_transformer=StandardScaler(), X_transformer=pipeline))

    # Verify if the data is correctly loaded
    for key, item in loader.data.items():
        print(key, item.shape)

    # save the data

