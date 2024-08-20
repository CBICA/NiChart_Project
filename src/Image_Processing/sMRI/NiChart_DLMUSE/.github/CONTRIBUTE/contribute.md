### **How to contribute?**
* Open a pull request exaplaining your addition/fix. You can also see the [Bug report](../ISSUE_TEMPLATE/bug_report.md)

* Implementation is not everything. You have to add test cases for every function that you add at the [tests](../../tests/) folder.

* Last but not least, some documentation will be good for the users to learn how to use your code. You can add your documentation at the [docs](../../docs/) folder.

### **Good practices**
* We have automated test cases that are located at the [tests](../../tests/) folder. The workflow will automatically run the test cases on your PR, but, before creating a PR make sure your test cases run correctly locally using this command ```cd tests/unit && python -m unittest discover -s . -p "*.py"```

* Make sure that you add any new libraries that you may use at [requirements](../../requirements.txt) as well as updating the [setup.py](../../setup.py) file(if needed).

* Try to add docstrings on every new function that you implement as it will automatically be generated to documentation.Comments should look like this
    - For functions:
    ```python
        def your_function(param1: type, param2: type, ...) -> return_type:
            """
                Describe briefly what your function does

                :param param1: Describe what param1 is used for
                :type param1: type of param1
                :param param2: Describe what param2 is used for
                :type param2: type of param2
                etc.
                :return: Explain what your function returns
                :rtype: return type of the function

            """
    ```
    - For classes:
    ```python
        class your_class:
            """
                Describe briefly what your class does

                    :param param1: Explain what param1 is used for
                    :type param1: type of param1
                    :param param2: Explain what param2 is used for
                    :type param2: type of param2
                    etc.

            """
        def __init__(self, param1: type, param2: type, ...):
            ...

        def func1(self, param1: type, param2: type, ...) -> return_type: 
            ...
    ```

* As seen above, it is a good practice to always set parameter types in functions as well as return types. 


