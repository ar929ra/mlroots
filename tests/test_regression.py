# This file contains unit tests of the regression implementatipon in
# mlroots/supervised/regression.py
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


import unittest
import numpy as np
import pandas as pd

from mlroots.supervised import regression


class FitTest(unittest.TestCase):
    """ Test cases to test the result of fitting univariate and 
    multivariate regressions on dummy data

    Verifies that the coefficients, MSE, R-squared, F score,
    confidence intervals, and p-value estimates are correct
    """


    def setUp(self):
        """ Instantiates regression objects and gets data to test """
        self.uni_x = np.array([1,2,3,4,5,6,7,8,9,10])
        self.multi_x = np.array([[1,2,3,4,5,6,7,8,9,10],
                                 [20,17,18,2,3,4,1,19,17,8]])
        self.multi_x_df = pd.DataFrame(np.transpose(self.multi_x))
        self.y = np.array([-3.2,-36.3,100.4,55.9,
                            73.4,69.9,90.1,-85.5,
                           -39.1, 126.1])

        # Linear model with one feature
        self.uni_coeff = np.array([24.713,1.901])
        self.multi_coeff = np.array([110.397,-1.885,-5.950])

        self.uni_model = regression.LinMod(self.y, self.uni_x)
        self.uni_fit = self.uni_model.fit()

        # Linear model with more than 1 feature
        self.multi_model = regression.LinMod(
            self.y, 
            np.transpose(self.multi_x)
        )
        self.multi_fit = self.multi_model.fit()

        # Linear model from data frame predictor data
        self.multi_model_df = regression.LinMod.from_df(
            self.y,
            self.multi_x_df
        )
        self.multi_df_fit = self.multi_model_df.fit()


    def test_uni_coeff(self):
        """ Tests univariate coefficients """
        for idx, coeff in enumerate(self.uni_fit):
            self.assertAlmostEqual(coeff, self.uni_coeff[idx], places = 2,
            msg = "Univariate regression coefficients incorrectly calculated"
        )


    def test_multi_coeff(self):
        """ Tests multivariate coefficients """
        for idx, coeff in enumerate(self.multi_fit):
            self.assertAlmostEqual(coeff, self.multi_coeff[idx], places = 2,
            msg = "Multivariate regression coefficients incorrectly calculated"
        )
            

    def test_multi_df_coeff(self):
        """ Tests multivariate coefficients from df formatted model """
        for idx, coeff in enumerate(self.multi_df_fit):
            self.assertAlmostEqual(coeff, self.multi_coeff[idx], places = 2,
            msg = "From df LinMod class method yielding wrong coefficients"
        )

    
    def test_rsq(self):
        self.assertAlmostEqual(self.multi_model.r_sq, 0.4252, places = 2,
            msg = "Multiple R squared calculated incorrectly"
        )


if __name__ == "__main__":
    unittest.main()
