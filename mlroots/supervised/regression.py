# This file implements various classes to handle parametric and
# nonparamtric regression models
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


import numpy as np

from mlroots.utils.utils import *


class LinMod(object):
    """ Linear model base class

    Defines base linear model based on OLS regression
    """


    def __init__(self, response, design_matrix):
        """ Linear model constructor

        Takes a response array and an n x p design matrix where
        n is the sample size and p is the number of features or
        variables

        Adds a vector of ones to the design matrix
        """
        verify_data_type(design_matrix)
        verify_data_type(response, design_matrix)

        try: 
            self.n, self.features = design_matrix.shape
        except ValueError:
            self.n, self.features = (len(design_matrix),1)

        try:
            self.design_matrix = np.hstack([
                np.array([[1]] * self.n),
                design_matrix
            ])
        except ValueError:
            self.design_matrix = np.hstack([
                np.array([[1]] * self.n),
                np.transpose(np.array([design_matrix]))
            ])
        
        self.response = response
        self.coeff = None


    @classmethod
    def from_df(cls, response, df):
        """ Class method to construct design matrix and response
        array from pandas df for convenience
        """
        try:
            df.shape
            np_design = df.as_matrix()

        except ValueError:
            np_design = np.asarray(df)

        np_response = np.asarray(response)

        return cls(np_response, np_design)


    def fit(self):
        """ Calculates the k coefficients that minimize the sum
        of the squares of differences between the observed and
        predicted responses

        It can be shown that the vector of parameters that minimize
        this sum of squares is given by:

        b = (X'X)^-1 X'Y

        Where X is the design matrix, X' is its transpose, and
        Y is the vector of observed responses

        More details: https://goo.gl/LA1Mt7
        """
        xt = np.transpose(self.design_matrix)
        xtx = xt.dot(self.design_matrix)
        xtx_inv = np.linalg.inv(xtx)
        xty = xt.dot(self.response)

        self.coeff = xtx_inv.dot(xty)

        return self.coeff