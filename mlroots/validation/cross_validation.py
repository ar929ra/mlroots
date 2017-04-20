# This file implements various cross validation methods for model
# validation and selection
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


import numpy as np

from mlroots.supervised.classifier import Classifier
from mlroots.utils.utils import *


class KFold(object):

	@convert_arr
	def __init__(self, rslt, data, models, folds = 10):
		self.rslt = rslt
		self.folds = folds
		self.models = models

		try:
			if k in data:
				self.k = data["k"]
			self.data = np.transpose(
				np.asarray([itm for ind,itm in data if itm != 'k'])
			)

		except TypeError:
			self.k = None
			self.data = data

		try: 
            self.n, self.col = self.data.shape

        except ValueError:
            self.n, self.col = (len(self.data),1)

        self.size = np.round(self.n * .25)
		self.result = np.arange(len(models), dtype = np.float32)


	def validate(self, folds = None, size = None):
		if not folds:
			folds = self.folds

		for idx,model in enumerate(models):

			for fold in folds:
				test_idx = np.random.choice(self.n, self.size, replace = False)
				train_idx = np.setdiff1d(np.arange(self.n),train_idx)


class LOOV(object):

	@convert_arr
	def __init__(self, rslt, data, models):
		self.rslt = rslt
		self.data = data
		self.models = models