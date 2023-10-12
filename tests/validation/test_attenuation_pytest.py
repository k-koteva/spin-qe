"""
This module contains tests for the Atten class in the spin_qe.validation.attenuation module.
It tests the initialization of the Atten class with different parameters and the validate_and_convert method.
"""

import pytest

from spin_qe.validation.attenuation import Atten


def test_make_single_value():
    atten_instance = Atten.make(dB=10)
    assert atten_instance.dB == 10


def test_make_raises_exception():
    with pytest.raises(ValueError):
        Atten.make(dB=10, perc=20)


def test_val_single_float():
    dB_val = Atten.val(dB=10)
    assert dB_val == 10


def test_val_single_float():
    dB_val = Atten.val(frac=1, convert_to='dB')
    assert dB_val == 0


def test_val_convert_to():
    perc_val = Atten.val(perc=10, convert_to='dB')
    assert perc_val == 10


def test_val_list():
    dB_list = Atten.val(dB=[10, 20])
    assert dB_list == [10, 20]


def test_val_convert_list():
    perc_list = Atten.val(perc=[10, 50], convert_to='dB')
    assert perc_list == pytest.approx([10, 3.01029], abs=1e-5)


def test_val_none_input():
    none_val = Atten.val()
    assert none_val is None


def test_val_invalid_input():
    with pytest.raises(ValueError):
        Atten.val(dB="invalid")
