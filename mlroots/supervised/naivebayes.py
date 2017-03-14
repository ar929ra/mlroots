# This file implements various Naive Bayes algorithms to classify based on 
# both text and quantitative data
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


import math

from .classifier import Classifier
from mlroots.errors.classifier_errors import *
from mlroots.utils.utils import *


class NbText(Classifier):


    def __init__(self, classes, **kwargs):
        if "documents" in kwargs.keys():
            super(NbText, self).__init__(classes, **kwargs)
            self._create_class_map()

            self.word_counts = dict([[c,0] for c in self.unique_classes])
            self.vocabulary = 0

            self._create_bag_of_words()

        else:
            raise DocumentNotFoundError("Document training data not found")


    def _create_bag_of_words(self):
        text = self.data["documents"]
        word_instances = []

        for idx,c in enumerate(self.classes):
            words = str(text[idx]).split(" ")

            for word in words:
	            word_instances.append(word)
	            self.word_counts[c] += 1

	            if self.class_map[c].get(word):
		            self.class_map[c][word] += 1
	            else:
		            self.class_map[c][word] = 1

        self.vocabulary = len(set(word_instances))


    def predict(self, **kwargs):
        self.last_prediction[:] = []

        if "test_documents" in kwargs.keys():

            for document in kwargs["test_documents"]:
                max_prob = 0
                tmp_class = ""
                clean_doc = clean_text(document)

                for idx,text_class in enumerate(self.class_map):
                    prob = math.log(self.class_probabilities[text_class])

                    for word in clean_doc.split(" "):
                        num = self.class_map[text_class].get(word,0) + 1
                        den = self.word_counts[text_class] + self.vocabulary

                        prob += math.log((num * 1.0)/den)

                        if idx == 0 or prob > max_prob:
                            max_prob = prob
                            tmp_class = text_class

                self.last_prediction.append(tmp_class)

            return self.last_prediction

        else:
            raise DocumentNotFoundError("Document test data not found")


    def get_accuracy(self, test_classes, **kwargs): pass


class NbGaussian(Classifier): pass