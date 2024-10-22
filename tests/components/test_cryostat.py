#Test cryogenics

import pytest
from pandas import DataFrame

from spin_qe.components.cryostat import (Cryo, calculate_specific_power,
                                         overlay_heat_evacuated_plots,
                                         plot_total_power_vs_Si_abs_and_Tq)
from spin_qe.validation.attenuation import Atten
from spin_qe.validation.power import Power
from spin_qe.validation.temperature import Temp
from pandas import DataFrame
from pandas.testing import assert_frame_equal


@pytest.fixture
def cryo_instance():
    temps = [4, 0.8, 0.1]
    attens = [4, 7, 0.1]
    Tq = 0.2
    Si_abs = 0.0
    per_cable_atten = 6
    return Cryo(
        Tq=Tq,
        temps=temps,
        attens=attens,
        Si_abs=Si_abs,
        per_cable_atten=per_cable_atten,
        efficiency='Small System'
    )

def test_set_stages(cryo_instance):
    # Run the set_stages function
    result_df = cryo_instance.stages

    # Define what the expected DataFrame should look like
    expected_df = DataFrame({
        'temps': [300, 4, 0.8, 0.2],
        'attens': [6, 10, 13, 6]
    })

    # Assert that the result matches the expected DataFrame
    assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-5, rtol=1e-5)

    # Optionally, print the DataFrame for visual verification during tests
    print(result_df)
# def test_temps_and_attens_validation(cryo_instance):
#     assert len(cryo_instance.temps) == len(cryo_instance.attens)
#     assert all(isinstance(temp, float) for temp in cryo_instance.temps)
#     assert all(isinstance(atten, float) for atten in cryo_instance.attens)

# def test_heat_evacuated_at_stage(cryo_instance):
#     power = 1.0
#     temp = 0.8
#     heat_evacuated = cryo_instance.heat_evacuated_at_stage(temp, power)
#     assert isinstance(heat_evacuated, float)
#     assert heat_evacuated > 0

# def test_total_heat_evacuated(cryo_instance):
#     power = 1.0
#     total_heat = cryo_instance.total_heat_evacuated(power)
#     assert isinstance(total_heat, float)
#     assert total_heat > 0

# def test_calculate_input_power(cryo_instance):
#     power_at_Tq = 0.5e-3
#     input_power = cryo_instance.calculate_input_power(power_at_Tq)
#     assert isinstance(input_power, float)
#     assert input_power > power_at_Tq

# def test_error_on_invalid_temps_length():
#     Tq = 0.04
#     Si_abs = 0.5
#     cables_atten = 30
#     with pytest.raises(ValueError):
#         Cryo(
#             Tq=Tq,
#             temps=[4, 0.8],  # Incorrect length
#             attens=[10, 3, 0.1],
#             Si_abs=Si_abs,
#             cables_atten=cables_atten
#         )

# def test_plot_generation(cryo_instance):
#     power = 1.0
#     try:
#         cryo_instance.plot_heat_evacuated_vs_temperature(power)
#     except Exception as e:
#         pytest.fail(f"plot_heat_evacuated_vs_temperature raised an exception {e}")

# if __name__ == '__main__':
#     pytest.main()


# def test_validate_Tq():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     assert cryo.Tq == 0.02

# def test_validate_temps():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     assert cryo.temps == [Temp.val(K=4)]

# def test_validate_attens():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     assert cryo.attens == [Atten.val(dB=3.01)]

# def test_set_stages():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     assert isinstance(cryo.stages, DataFrame)
#     assert 'temps' in cryo.stages.columns
#     assert 'attens' in cryo.stages.columns

# def test_eff_carnot():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01], efficiency='Carnot')
#     efficiency = cryo.eff(stage_T=4)
#     assert efficiency > 0

# def test_eff_small_system():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01], efficiency='Small System')
#     efficiency = cryo.eff(stage_T=4)
#     assert efficiency > 0


# def test_total_heat_evacuated():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     total_heat = cryo.total_heat_evacuated(power=1.0)
#     assert total_heat > 0

# def test_calculate_input_power():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     input_power = cryo.calculate_input_power(power_at_Tq=0.001)
#     assert input_power > 0

# def test_power_to_evacuate_heat_at_stage():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     power = cryo.power_to_evacuate_heat_at_stage(temp=4, power=1.0)
#     assert power > 0

# def test_total_power():
#     cryo = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     total_power = cryo.total_power(power=1.0)
#     assert total_power > 0

# def test_calculate_specific_power():
#     power = calculate_specific_power(temp=4)
#     assert power > 0

# def test_overlay_heat_evacuated_plots():
#     cryo1 = Cryo(Tq=0.02, temps=[4], attens=[3.01])
#     cryo2 = Cryo(Tq=0.04, temps=[4], attens=[3.01])
#     overlay_heat_evacuated_plots([cryo1, cryo2], power=1.0)

# def test_plot_total_power_vs_Si_abs_and_Tq():
#     Tq_values = [0.02, 0.04]
#     temps = [4]
#     attens = [3.01]
#     Si_abs_range = [0.1, 0.2, 0.3]
#     plot_total_power_vs_Si_abs_and_Tq(Tq_values, temps, attens, Si_abs_range)
