# This file implements utility functions used across all the other
# modules
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)

import numpy as np
import string

from mlroots.errors.errors import *


# Data integrity and cleaning

def verify_data_type(data, compare_length = None):
    data_type = type(data).__module__

    if not isinstance(data, list) and data_type != np.__name__:
        raise DataInputError(
	        "Data must be passed as lists or np arrays"
        )

    if compare_length and len(data) != len(compare_length):
        raise LengthMismatchError(
            "Predictor data must be the same length as response data"
        )


def clean_text(text):
    l_text_no_space = text.replace("\n","").lstrip().rstrip().lower()
    return "".join(l for l in l_text_no_space if l not in string.punctuation)


# Probability density functions

def norm_pdf(x, mu, sig):
    var = float(sig)**2
    den = (2*np.pi*var)**.5
    num = np.exp(-(float(x) - float(mu))**2 / (2*var))

    return num/den