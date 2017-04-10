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
from scipy.special import stdtr

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

        design_matrix = convert_to_array(design_matrix)
        response = convert_to_array(response)

        try: 
            self.n, self.k = design_matrix.shape

        except ValueError:
            self.n, self.k = (len(design_matrix),1)

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
        
        self.orig_design_matrix = design_matrix
        self.response = response


        # Default model result values
        self.coeff = None
        self.predicted = None
        self.residuals = None
        self.ss_res = None
        self.ss_total = None
        self.ss_reg = None
        self.r_sq = None
        self.cov_matrix = None
        self.coeff_se = None
        self.coeff_t = None
        self.coeff_p = None


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
        yty = self.response.dot(self.response)
        sq_y_n = (np.sum(self.response)**2) / self.n

        # Calculated coefficients, predicted values, and residuals
        self.coeff = xtx_inv.dot(xty)
        self.predicted = self.design_matrix.dot(self.coeff)
        self.residuals = self.response - self.predicted

        # Calculate sum of squares and R squared
        self.ss_res = self.residuals.dot(self.residuals)
        self.ss_total = yty - sq_y_n
        self.ss_reg = self.ss_total - self.ss_res
        self.r_sq = self.ss_reg / self.ss_total

        # Calculate covariance matrix and standard error of coefficients
        self.cov_matrix = (self.ss_res / (self.n - self.k - 1))*xtx_inv
        self.coeff_se = np.sqrt(np.diagonal(self.cov_matrix))

        # Calculate p-values based on t test on coefficient estimates
        self.coeff_t = self.coeff/self.coeff_se
        self.coeff_p = (
            1 - stdtr(self.n - self.k - 1, np.absolute(self.coeff_t)))*2

        return self.coeff


    def predict(self, test_matrix):
        """ Given a matrix of test data, returns a vector of predicted y
        values calculated with coefficients based on training data
        """
        test_matrix = convert_to_array(test_matrix)
        verify_data_type(test_matrix, self.coeff)

        try:
            test_matrix = np.hstack([
                np.array([[1]] * self.n),
                test_matrix
            ])

        except ValueError:
            test_matrix = np.hstack([
                np.array([[1]] * self.n),
                np.transpose(np.array([test_matrix]))
            ])

        return test_matrix.dot(self.coeff)


    def summary(self):
        """ Print summary of regression output to console """
        col_headers = np.array([
            "Coeff", "Estimate", "Std. Error", "t-stat", "p-value"
        ])
        coeff_header = np.array([np.array([i]) for i in range(self.k + 1)])
        coeff_output = np.array([
            self.coeff, self.coeff_se, self.coeff_t, self.coeff_p
        ])
        print_matrix = np.hstack([coeff_header, np.transpose(coeff_output)])

        print("\nCoefficients:\n\n{:^15} {:^15} {:^15} {:^15} {:^15}".format(*col_headers))

        for row in print_matrix:
            formatted_row = "{:^15.0f} {:^15.2f} {:^15.2f} {:^15.2f} {:^15.2f}".format(*row)
            print(formatted_row)