# This file implements utility functions used across all the other
# modules
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


import string


def clean_text(text):
    l_text_no_space = text.replace("\n","").lstrip().rstrip().lower()
    return "".join(l for l in l_text_no_space if l not in string.punctuation)