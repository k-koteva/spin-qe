"""
This module contains tests for the Atten class in the spin_qe.validation.attenuation module.
It tests the initialization of the Atten class with different parameters and the validate_and_convert method.
"""

import math

import pytest

from spin_qe.validation.attenuation import Atten


def test_make_single_value():
    atten_instance = Atten.make(dB=10)
    assert atten_instance.dB == 10


def test_make_raises_exception():
    with pytest.raises(ValueError):
        Atten.make(dB=10, perc=20)


def test_val_single_float1():
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
    with pytest.raises(ValueError) as exc_info:
        Atten.val()
    assert str(exc_info.value) == "At least one of dB, perc, or frac must be provided."

def test_atten_initialization_no_parameters():
    with pytest.raises(ValueError) as excinfo:
        Atten()
    assert "Exactly one of dB, perc, or frac must be provided." in str(excinfo.value)

def test_val_invalid_input():
    with pytest.raises(ValueError):
        Atten.val(dB="invalid") # noqa

def test_atten_initialization_multiple_parameters():
    with pytest.raises(ValueError) as excinfo:
        Atten(dB=10, perc=50)
    assert "Exactly one of dB, perc, or frac must be provided." in str(excinfo.value)

    # You can also add additional assertions for other combinations of parameters if desired
    with pytest.raises(ValueError) as excinfo:
        Atten(dB=10, frac=0.1)
    assert "Exactly one of dB, perc, or frac must be provided." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Atten(perc=50, frac=0.1)
    assert "Exactly one of dB, perc, or frac must be provided." in str(excinfo.value)

def test_atten_make_with_perc():
    atten_instance = Atten.make(perc=50)
    assert atten_instance.perc == 50
    # You can also add assertions to verify the correct calculation of `dB` and `frac` based on `perc`
    expected_db = -10 * math.log10(atten_instance.frac)
    assert atten_instance.dB == pytest.approx(expected_db, rel=1e-3)
    assert atten_instance.frac == 0.5

def test_atten_make_with_frac():
    atten_instance = Atten.make(frac=0.1)
    assert atten_instance.frac == 0.1
    # Similarly, verify the correct calculation of `dB` and `perc` based on `frac`
    expected_db = -10 * math.log10(atten_instance.frac)
    assert atten_instance.dB == pytest.approx(expected_db, rel=1e-3)
    assert atten_instance.perc == 10

def test_atten_val_invalid_convert_to():
    with pytest.raises(ValueError) as excinfo:
        Atten.val(dB=10, convert_to='invalid_parameter')
    assert "Invalid convert_to, must be one of 'dB', 'perc', 'frac'" in str(excinfo.value)



