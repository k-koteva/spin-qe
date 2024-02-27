"""
This module contains tests for the Temperature class in the spin_qe.base module.
"""


import pytest

from spin_qe.validation.temperature import Temp


def test_make_single_value():
    temp_instance = Temp.make(K=100)
    assert temp_instance.K == 100
    assert temp_instance.mK == 100000


def test_make_invalid_values():
    with pytest.raises(ValueError):
        Temp.make()
    with pytest.raises(ValueError):
        Temp.make(K=100, mK=100000)


def test_val_single_float():
    assert Temp.val(K=100) == 100
    assert Temp.val(mK=100000, convert_to='K') == 100


def test_val_list_float():
    assert Temp.val(K=[100, 200]) == [100, 200]
    assert Temp.val(mK=[100000, 200000], convert_to='K') == [100, 200]


def test_val_invalid_input_type():
    with pytest.raises(ValueError):
        Temp.val(K="string") # type: ignore


def test_temp_init_no_parameters():
    with pytest.raises(ValueError) as e:
        Temp()
    assert str(e.value) == "Exactly one of K or mK must be provided."

def test_temp_init_multiple_parameters():
    with pytest.raises(ValueError) as e:
        Temp(K=100, mK=100000)
    assert str(e.value) == "Exactly one of K or mK must be provided."


def test_val_invalid_convert_to():
    with pytest.raises(ValueError):
        Temp.val(K=100, convert_to='invalid')


def test_val_none_input():
    assert Temp.val(K=None) is None
    assert Temp.val(mK=None) is None


def test_val_edge_cases():
    with pytest.raises(ValueError):
        Temp.val(K=0)
    with pytest.raises(ValueError):
        Temp.val(K=301)
    with pytest.raises(ValueError):
        Temp.val(mK=0)
    with pytest.raises(ValueError):
        Temp.val(mK=300001)

def test_temp_make_with_mK():
    temp_instance = Temp.make(mK=273000)
    assert temp_instance.mK == 273000
    assert temp_instance.K == 273  # Verifying the conversion
