import unittest
import string
import numpy as np

from mlroots.utils.utils import *
from mlroots.errors.errors import *


class ArrayConversionTest(unittest.TestCase):


    def array_check(self, data, msg):
        rslt = convert_to_array(data)
        data_type = type(rslt).__module__

        self.assertEqual(data_type, np.__name__, msg = msg)


    def same_check(self, data, msg):
        rslt = convert_to_array(data)
        data_type = type(rslt)

        self.assertIsInstance(data, data_type, msg = msg)


    def test_array_input(self):
        arr = np.array([1,2,3,4,5])

        self.array_check(
            arr, 
            "convert_to_array fails for array input"
        )


    def test_list_input(self):
        lst = [[1],[2],3,4,5]

        self.array_check(
            lst,
            "convert_to_array fails for lst input"
        )


    def test_list_conversion(self):
        lst = [[1],[2],3,4,5]
        arr = np.array([[1],[2],3,4,5])

        for idx,val in enumerate(lst):
            self.assertEqual(val, arr[idx],
                msg = "convert_to_array returns wrong structure"
            )


    def test_str_input(self):
        text = "convert me please"

        self.same_check(
            text,
            "convert_to_array fails for str input"
        )


    def test_num_input(self):
        num = 1.2

        self.same_check(
            num,
            "convert_to_array fails for numerical input"
        )


class TypeVerificationTest(unittest.TestCase):


    def test_type_error(self):
        self.assertRaises(
            DataInputError,
            verify_data_type,
            "wrong input"
        )


    def test_dimension_error(self):
        self.assertRaises(
            LengthMismatchError,
            verify_data_type,
            [1,2,3,4],
            [4,5,6]
        )


class TextCleanTest(unittest.TestCase):
    """ Text cases to test text cleaning utility function"""


    def test_string_clean(self):
        """ Tests whether all unwanted characters are removed by
        clean_text function
        """
        orig_str = string.punctuation + "\nhello"
        self.assertEqual(clean_text(orig_str), "hello", msg = 
            "clean_text does not remove unwanted characters"
        )


if __name__ == "__main__":
    unittest.main()