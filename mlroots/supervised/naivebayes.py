# This file implements various Naive Bayes algorithms to classify based on 
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


class NbText(Classifier):
    """ Naive Bayes text classifier

    Accepts a list or np array of discrete outcomes and an input of text
    documents and builds a classifier based on these data.

    Explanation of algorithm: https://goo.gl/H4ZgXw.
    """


    def __init__(self, classes, **kwargs):
        """ NbText constructor

        The classes parameter refers to the training class responses.

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
        if "documents" in kwargs:
            super(NbText, self).__init__(classes, **kwargs)
            self._create_class_map()

            self.word_counts = dict([[c,0] for c in self.unique_classes])
            self.vocabulary = 0

            self._create_bag_of_words()

        else:
            raise DocumentNotFoundError("Document training data not found")


    def _create_bag_of_words(self):
        """ Private method that counts the number of words in the training
        documents by class and the number of words in total.

        Used to calculate the probability P(w|c) for each w word given c
        class.
        """
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

        # Vocabulary = all unique words in the training data
        self.vocabulary = len(set(word_instances))


    def predict(self, **kwargs):
        """ Returns an np array of predicted classes given an input of
        documents.

        The class any one document is assigned to is given by:

            C_nb = argmax P(c_j) prod P(x_i|c_j)

        We can correctly classify test documents by iterating through 
        each word in the document and calculating P(x|c_j) for each
        j class. The product of all conditional probabilities of each
        word given each class with the prior probabilities of each
        class denoted as P(c_j) is compared between all classes. The
        class with the max probability given a document is assigned
        to that document.
        """
        if "test_documents" in kwargs:
            data = kwargs["test_documents"]
            verify_data_type(data)

            self.last_prediction = np.empty(
                len(data), dtype = "object"
            )

            for doc_idx,document in enumerate(data):
                max_prob = 0
                tmp_class = None
                clean_doc = clean_text(document)

                for class_idx,text_class in enumerate(self.class_map):
                    prob = np.log(self.class_probabilities[text_class])

                    for word in clean_doc.split(" "):
                        num = self.class_map[text_class].get(word,0) + 1
                        den = self.word_counts[text_class] + self.vocabulary
                        
                        # Sum of logs avoids floating-point underflow 
                        prob += np.log((num * 1.0)/den)

                    if class_idx == 0 or prob > max_prob:
                        max_prob = prob
                        tmp_class = text_class

                self.last_prediction[doc_idx] = tmp_class

            return self.last_prediction

        else:
            raise DocumentNotFoundError("Document test data not found")


    def get_accuracy(self, test_classes, **kwargs):
        """ Given a list or array of classes and an optional input
        of documents, this function returns the percent of correctly
        classified documents.

        If no test documents are given, the function compares the given
        test classes to the last predicted classes when either this
        function or predict were called.
        """
        verify_data_type(test_classes)
        data_type = type(test_classes).__module__

        if data_type != np.__name__:
            test_classes = np.asarray(test_classes)

        if "test_documents" in kwargs:
            data = kwargs["test_documents"]
            verify_data_type(data, test_classes)

            self.predict(test_documents = data)

        else:
            verify_data_type(test_classes, self.last_prediction)

        correct_predictions = test_classes == self.last_prediction

        return np.sum(correct_predictions)/len(test_classes)


class NbGaussian(Classifier):


    def __init__(self, classes, **kwargs):
        super(NbGaussian, self).__init__(classes, **kwargs)
        self._create_class_map()

        self.parameters = dict([[c,{}] for c in self.unique_classes])

        self._get_class_parameters()


    def _get_class_parameters(self):
        tmp_classes = np.asarray(self.classes);

        for c in self.unique_classes:
            indexes = np.where(tmp_classes == c)[0]

            for key in self.data:
                values_at_ind = data_arr[indexes]

                mean = np.mean(values_at_ind)
                sdev = np.std(values_at_ind)

                self.parameters[c][key] = (mean,sdev)


    def predict(self, **kwargs):
        if len(kwargs) == len(self.data):
            data_len = len(kwargs[list(kwargs.keys())[0]])
            self.last_prediction = np.empty(data_len, dtype = "object")

            for test_item in np.arange(data_len):
                max_prob = 0
                tmp_class = None

                for class_idx,res_class in enumerate(self.class_map):
                    prob = np.log(self.class_probabilities[res_class])

                    for var,obsv in kwargs.items():
                        mean, sdev = self.parameters[res_class][var]

                        prob += norm_pdf(obsv[test_item], mean, sdev)

                    if class_idx == 0 or prob > max_prob:
                        max_prob = prob
                        tmp_class = res_class

                self.last_prediction[test_item] = tmp_class

            return self.last_prediction

        else:
            raise LengthMismatchError(
                "Num of test variables must equal num of training variables"
            )

    def get_accuracy(self, test_classes, **kwargs): pass