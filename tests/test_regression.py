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

from mlroots.supervised import regression


class FitTest(unittest.TestCase):
    """ Test cases to test the result of fitting univariate
    and multivariate regressions on dummy data

    Verifies that the coefficients, MSE, R-squared, F score,
    confidence intervals, and p-value estimates are correct
    """


    def setUp(self):
        """ Instantiates regression objects and gets data
        to test
        """
        self.uni_x = np.array([1,2,3,4,5,6,7,8,9,10])
        self.y = np.array([-3.2,-36.3,100.4,55.9,
                            73.4,69.9,90.1,-85.5,
                           -39.1, 126.1])
        self.uni_a = 24.713
        self.uni_b = 1.901

        self.uni_model = regression.LinMod(self.y,self.uni_x)
        self.uni_fit = self.uni_model.fit()


    def test_uni_intercept(self):
        """ Tests univariate intercept """
        a = self.uni_fit[0]

        self.assertAlmostEqual(a, self.uni_a, places = 2, msg = 
            "Univariate regression intercept not correct"
        )


    def test_uni_slope(self):
        """ Tests univariate slope """
        b = self.uni_fit[1]

        self.assertAlmostEqual(b, self.uni_b, places = 2, msg =
            "Univariate regression slope not correct"
        )


if __name__ == "__main__":
    unittest.main()