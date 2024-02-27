import pytest

from spin_qe.validation.frequency import \
    Frequency  # Adjust this import according to your module structure


def test_make_single_value():
    frequency_instance = Frequency.make(Hz=1500)
    assert frequency_instance.Hz == 1500
    assert frequency_instance.kHz == 1.5
    assert frequency_instance.MHz == 0.0015
    assert frequency_instance.GHz == 0.0000015

def test_make_invalid_values():
    with pytest.raises(ValueError):
        Frequency.make()
    with pytest.raises(ValueError):
        Frequency.make(Hz=1500, kHz=1.5)

def test_val_single_float():
    assert Frequency.val(Hz=1500) == 1500
    assert Frequency.val(kHz=1.5, convert_to='Hz') == 1500
    assert Frequency.val(MHz=0.0015, convert_to='Hz') == 1500
    assert Frequency.val(GHz=0.0000015, convert_to='Hz') == 1500

def test_val_list_float():
    assert Frequency.val(Hz=[1500, 2000]) == [1500, 2000]
    assert Frequency.val(kHz=[1.5, 2], convert_to='Hz') == [1500, 2000]
    assert Frequency.val(MHz=[0.0015, 0.002], convert_to='Hz') == [1500, 2000]
    assert Frequency.val(GHz=[0.0000015, 0.000002], convert_to='Hz') == [1500, 2000]

def test_val_invalid_input_type():
    with pytest.raises(ValueError):
        Frequency.val(Hz="string")

def test_val_invalid_convert_to():
    with pytest.raises(ValueError):
        Frequency.val(Hz=1500, convert_to='invalid')

def test_val_none_input():
    assert Frequency.val(Hz=None) is None
    assert Frequency.val(kHz=None) is None
    assert Frequency.val(MHz=None) is None
    assert Frequency.val(GHz=None) is None

def test_val_edge_cases():
    with pytest.raises(ValueError):
        Frequency.val(Hz=-1)
    with pytest.raises(ValueError):
        Frequency.val(kHz=-1)
    with pytest.raises(ValueError):
        Frequency.val(MHz=-1)
    with pytest.raises(ValueError):
        Frequency.val(GHz=-1)

def test_val_nested_list():
    nested_list = [[1500, 2000], [2500, 3000]]
    converted = Frequency.val(Hz=nested_list)
    assert converted == [1500, 2000, 2500, 3000]

    nested_list = [[1.5, 2], [2.5, 3]]
    converted = Frequency.val(kHz=nested_list, convert_to='Hz')
    assert converted == [1500, 2000, 2500, 3000]

    nested_list = [[0.0015, 0.002], [0.0025, 0.003]]
    converted = Frequency.val(MHz=nested_list, convert_to='Hz')
    assert converted == [1500, 2000, 2500, 3000]

    nested_list = [[0.0000015, 0.000002], [0.0000025, 0.000003]]
    converted = Frequency.val(GHz=nested_list, convert_to='Hz')
    assert converted == [1500, 2000, 2500, 3000]
