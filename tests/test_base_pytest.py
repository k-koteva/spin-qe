"""
This module contains tests for the Temperature class in the spin_qe.base module.
"""
import numpy as np
from spin_qe.base import Temperature

def test_validate_array():
    """
    Test the validate_array method of the Temperature class.
    """
    input_values = [1, 100, 300, -1, 400]
    expected_output = np.array([1, 100, 300])
    np.testing.assert_array_equal(Temperature.validate_array(input_values), expected_output)

def test_validate_single():
    """
    Test the validate_single method of the Temperature class.
    """
    assert Temperature.validate_single(100) == 100
    assert Temperature.validate_single(0) is None
    assert Temperature.validate_single(301) is None

def test_create_temperature():
    """
    Test the create_temperature method of the Temperature class.
    """
    temp_instance = Temperature.create_temperature(100)
    assert temp_instance is not None
    assert temp_instance.value == 100

    temp_instance = Temperature.create_temperature(0)
    assert temp_instance is None

    temp_instance = Temperature.create_temperature(301)
    assert temp_instance is None


# export JUPYTER_PLATFORM_DIRS=1    -> run this to get rid of warning in the terminal