# This file implements various custom errors
#
# Part of mlroots, a project to implement well-known machine learning 
# algorithms from scratch
#
# Originally authored by Adams Rosales <adamisto92@gmail.com>
#
# Licensed under MIT License (see LICENSE.txt for details)


class DocumentNotFoundError(Exception):
    pass

class DataInputError(Exception):
    pass

class LengthMismatchError(Exception):
    pass

class ModelSpecificationError(Exception):
    pass
