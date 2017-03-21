# This file implements the base abstract class for supervised classification
# problems
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)

import numpy as np
from abc import ABCMeta, abstractmethod

from mlroots.utils.utils import *


class Classifier(object):
    """ Abstract base class for supervised classification

    Defines common interface for supervised classification child classes
    as well as abstract methods that must be implemented by these children

    Accepts a list of classes or discrete outcomes and key value pairs
    for training data where the values must be lists
    """
    __metaclass__ = ABCMeta


    def __init__(self, train_classes, **kwargs):
        verify_data_type(train_classes)

        self.classes = train_classes
        self.unique_classes = set()
        self.num_of_classes = len(self.classes)
        self.class_map = {}
        self.class_probabilities = {}

        self.last_prediction = np.empty(len(train_classes), dtype = "object")
        self.accuracy = 0

        self.data = {}

        for name, value in kwargs.items():
            verify_data_type(value, train_classes)

            value = convert_to_array(value)

            self.data[name] = value


    def _create_class_map(self):
        """ Private method to initialize an empty bag of words container for
        each class in the training outcomes as well as prior probabilities of
        each class in the series of training data
        """
        self.unique_classes = set(self.classes)
        self.class_map = dict([[c,{}] for c in self.unique_classes])

        self.class_probabilities = {c:(self.classes.count(c) * 1.0) / \
        self.num_of_classes for c in self.unique_classes}


    def _create_data_matrix(self, data_dict):
        """ Private method to create an m x n matrix from n data vectors

        Used to simplify classifier related vector calculations across
        all the features at once and to handle data in matrix form
        """
        matrix_array = np.array([v for k,v in data_dict.items()])

        return np.transpose(matrix_array)


    @abstractmethod
    def predict(self, **kwargs): 
        """ Public method returning an array of predicted outcomes from given data

        Must be implemented by children 
        """
        pass


    @abstractmethod
    def get_accuracy(self, test_classes, **kwargs): 
        """ Public method returning an accuracy score of predicted outcomes

        Must be implemented by children 
        """
        pass