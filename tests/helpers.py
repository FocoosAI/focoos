from contextlib import contextmanager
from typing import Type

ExceptionType = Type[BaseException]


@contextmanager
def not_raises(expected_exception: ExceptionType):
    """
    A context manager that ensures the code inside the 'with' block does not raise
    the specified expected_exception. If it does, an AssertionError is raised.

    Args:
        expected_exception (ExceptionType): The exception type that should not be raised.

    Raises:
        AssertionError: If the specified exception is raised or any unexpected exception occurs.
    """
    try:
        yield
    except expected_exception as err:
        raise AssertionError(
            f"Expected no exception of type {repr(expected_exception)} to be raised, but got: {err}"
        )
    except Exception as err:
        raise AssertionError(
            f"An unexpected exception of type {type(err).__name__} was raised: {err}"
        )
