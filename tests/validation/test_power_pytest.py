import pytest

from spin_qe.validation.power import Power


def test_make_single_value():
    power_instance = Power.make(mW=5000)
    assert power_instance.mW == 5000
    assert power_instance.W == 5
    assert power_instance.kW == 0.005


def test_make_invalid_values():
    with pytest.raises(ValueError):
        Power.make()
    with pytest.raises(ValueError):
        Power.make(mW=5000, W=5)


def test_val_single_float():
    assert Power.val(mW=5000) == 5000
    assert Power.val(W=5, convert_to='mW') == 5000
    assert Power.val(kW=0.005, convert_to='mW') == 5000


def test_val_list_float():
    assert Power.val(mW=[5000, 6000]) == [5000, 6000]
    assert Power.val(W=[5, 6], convert_to='mW') == [5000, 6000]
    assert Power.val(kW=[0.005, 0.006], convert_to='mW') == [5000, 6000]


def test_val_invalid_input_type():
    with pytest.raises(ValueError):
        Power.val(mW="string") # type: ignore


def test_val_invalid_convert_to():
    with pytest.raises(ValueError):
        Power.val(mW=5000, convert_to='invalid')


def test_val_none_input():
    assert Power.val(mW=None) is None
    assert Power.val(W=None) is None
    assert Power.val(kW=None) is None


def test_val_edge_cases():
    with pytest.raises(ValueError):
        Power.val(mW=-1)
    with pytest.raises(ValueError):
        Power.val(W=-1)
    with pytest.raises(ValueError):
        Power.val(kW=-1)


def test_val_nested_list():
    nested_list = [[5000, 6000], [7000, 8000]]
    converted = Power.val(mW=nested_list) # type: ignore
    assert converted == [5000, 6000, 7000, 8000]

    nested_list = [[5, 6], [7, 8]]
    converted = Power.val(W=nested_list, convert_to='mW') # type: ignore
    assert converted == [5000, 6000, 7000, 8000]

    nested_list = [[0.005, 0.006], [0.007, 0.008]]
    converted = Power.val(kW=nested_list, convert_to='mW') # type: ignore
    assert converted == [5000, 6000, 7000, 8000]

def test_power_init_no_parameters():
    with pytest.raises(ValueError) as e:
        Power()
    assert str(e.value) == "Exactly one of mW, W, or kW must be provided."

def test_power_init_multiple_parameters():
    with pytest.raises(ValueError) as e:
        Power(mW=1000, W=1)
    assert str(e.value) == "Exactly one of mW, W, or kW must be provided."

def test_power_make_with_mW():
    power_instance = Power.make(mW=1000)
    assert power_instance.mW == 1000

def test_power_make_with_kW():
    power_instance = Power.make(kW=1)
    assert power_instance.kW == 1
    # Verify the correct conversion logic, if applicable.


def test_power_make_with_W():
    power_instance = Power.make(W=1000)
    assert power_instance.W == 1000
