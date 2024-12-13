############
Contributing
############

This document explains how to prepare your environment to contribute to our package.
Note that we always update our main branch if we want to update the stable release on PyPI, thus, creating
your own branch is recommended so that you can open a pull request.

*****************
Third party tools
*****************

We use pre-commit as our main tool for code formatting and checking. You always have to make sure that pre-commit pass in order to open a pull request.
You can perform this test just following these steps: ::

    # If you don't have pre-commit installed
    $ pip install pre-commit && pre-commit install

    # Then run pre-commit
    pre-commit run --color=always --all-files

We already have all the necessairy hooks at our `configuration <../.pre-commit-config.yaml>`_.

For our unit testing, we use pytest. Again, in order to contribute you need to make sure that all of the unit tests pass.
You can perform this test following these steps: ::

    # If you don't have pytest installed
    $ pip install pytest pytest-cov

    # Run pytest
    cd tests/ && pytest

.. note::

    We highly suggest you to install the dev requirements before contributing: ::

       $ pip install -r requirements.txt

************
Coding style
************

We follow a specific coding style that, luckily, can be easily checked with pre-commit. First and foremost, you should always add documentation to any new features you add.
We follow ``sphinx``'s docstring style so that the parser can detect the code comments and display them beautifully at our read-the-docs. Let's assume that you add a new function:

.. code-block:: python

    def foo(param1: int, param2: str, param3: list) -> int:
        """
        Brief explanation of what foo does.

        :param param1: Description of param1.
        :type param1: int
        :param param2: Description of param2.
        :type param2: str
        :param param3: Description of param3.
        :type param3: list
        :return: Description of the return value.
        :rtype: int
        """

``pre-commit`` will also check if you include libraries that are not needed, any extra spaces you might have, logic errors and much more.

Also, as you see above, you should always add types and return types to every function and class. We only write Python 3 code. If you don't follow all of the above, ``pre-commit`` won't pass, and a maintainer won't review your pull request.

**************
Good practices
**************

    - At your pull request, explain your additions, why are they useful or what they fix. There are currently no templates to follow for this, but you can just write a sufficient explanation so that a maintainer can understand your additions/fixes.

    - Write a nice commit message.

    - Make sure to follow the issue template if you want to open an issue.

    - Make sure to always add any third-party libraries that you add at the requirements so that the actions can install them.
