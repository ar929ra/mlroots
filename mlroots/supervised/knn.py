# This file implements various kNN algorithms to classify based on 
# both text and quantitative data
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


import numpy as np

from .classifier import Classifier
from mlroots.errors.errors import *
from mlroots.utils.utils import *


class Knn(Classifier):
    """ K-Nearest Neighbor classifier

    Accepts a list or np array of discrete outcomes and m quantitative
    features/dimensions/columns also as either lists or np arrays and
    builds a classifier based on these data

    Explanation of algorithm: https://goo.gl/9nBRCY
    """


    def __init__(self, classes, **kwargs):
    	""" Knn constructor

    	The classes parameter refers to the training class responses.

    	k must be included as a key-value pair in kwargs in order
    	to define a neighborhood of related points for each test
    	observation.

    	The training data should be passed in as key-value pairs where
    	each key corresponds to a feature/dimension/column in the
    	data. An easy way to go from a numpy matrix to a dictionary
    	that can be passed as kwargs is:

    	my_data = np.array([[1,2,3], [4,5,6], [7,8,9]])
    	kwargs = dict(enumerate(my_data))

    	To keep named features/dimensions/columns:

    	my_data = np.array([[1,2,3], [4,5,6], [7,8,9]])
    	columns = ["col1", "col2", "col3"] #np arrays are unhashable
    	kwargs = dict(zip(columns, my_data))
    	"""
    	super(Knn, self).__init__(classes, **kwargs)

    	self.classes = convert_to_array(classes)


    def predict(self, **kwargs):
    	if "k" not in kwargs:
    		raise DataInputError("A k value needs to be specified")

    	elif len(kwargs) != len(self.data) + 1:
    		raise LengthMismatchError(
    			"Num of test variables must equal num of training variables"
    		)

    	else:
    		k = kwargs["k"]
    		del kwargs["k"]

    		test_matrix = self._create_data_matrix(kwargs)
    		train_matrix = self._create_data_matrix(self.data)
    		train_rows, _ = train_matrix.shape
    		test_rows, _ = test_matrix.shape

    		self.last_prediction = np.empty(test_rows, dtype = "object")

    		# Iterate through each test observation
    		for obsv in range(test_rows):
    			similarities = np.empty(train_rows, dtype = np.float_)

    			# Get similarity scores to all training observations
    			for idx in range(train_rows):
    				similarities[idx] = euclidean_dist(
    					test_matrix[obsv], train_matrix[idx]
    				)

    			# Get neighbors of test observation
    			try:
    				smallest_k = np.argpartition(similarities, k)[:k]

    			except ValueError:
    				tmp_k = similarities.size - 1
    				smallest_k = np.argpartition(similarities, tmp_k)[:k]

    			neighbors = self.classes[smallest_k]

    			# Classify as most likely neighbor
    			neighbor_cls, counts = np.unique(neighbors, return_counts = True)

    			try:
    				matched_class = neighbor_cls[np.argpartition(counts, 1)[-1:]][0]

    			except ValueError:
    				matched_class = neighbor_cls[0]

    			self.last_prediction[obsv] = matched_class

    		return self.last_prediction

    def get_accuracy(self, test_classes, **kwargs): pass