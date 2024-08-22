### **How to write unit tests**
* As explained at the [contribution guide](contribute.md) it's a good practice to write test cases for any additions you make. For our test cases we use the unittest python3 library. The following is an example of a unit test with unittest library:

    ```python

    import unittest

    class CheckThisAndThat(unittest.TestCase):

        def testing_this(self):
            self.data1 = ...
            self.data2 = ...
            # run your functions...
            # result1 = my_func(self.data1, ...)
            # result2 = my_func(self.data2, ...)
            self.assertTrue(result1 == CorrectResult1)
            self.assertTrue(result2 == CorrectResult2)

        def testing_that(self):
            self.param1 = ...
            self.param2 = ...
            # run something
            # model = my_model(self.param1, self.param2, ...)
            # result1 = my_model.do_sth()
            self.assertTrue(result1 == CorrectResult1)
    ```

    The following will run 2 test cases and 3 assertions. We explain how to run test cases at the [contribution guide](contribute.md).
