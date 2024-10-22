import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from pandas import DataFrame
from pydantic import BaseModel, Field, validator

from spin_qe.validation.attenuation import Atten
from spin_qe.validation.power import Power
from spin_qe.validation.temperature import Temp


class Cryo(BaseModel):
    Tq: float
    temps: List[float]
    attens: List[float]
    Si_abs: Optional[float] = Field(0, ge=0, le=1)
    stages: DataFrame = DataFrame()
    # efficiency: str = 'Small System'
    efficiency: str = 'Carnot'
    per_cable_atten: Optional[float] = Field(6, ge=0)
    amplifiers: Optional[bool] = False

    @validator("Tq", pre=True, always=True)
    def validate_Tq(cls, value):  # pylint: disable=no-self-argument
        return Temp.val(K=value)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

    @validator("temps", pre=True, always=True)
    def validate_temps(cls, value):
        return [Temp.val(K=temp) for temp in value]

    @validator("attens", pre=True, always=True)
    def validate_attens(cls, value):
        return [Atten.val(dB=atten) for atten in value]

    @validator("stages", pre=True, always=True)
    def set_stages(cls, value, values):
        """
        Configures cryogenic system stages by adjusting attenuations based on cable losses and filtering
        out stages below the operational temperature threshold (Tq). Adds a standard 300K stage if not present
        and applies silicon absorption adjustments if specified. Outputs a DataFrame of adjusted stages 
        sorted in descending temperature order.

        Returns:
        - DataFrame: Stages with temperatures and corresponding attenuations.

        Raises:
        - ValueError: If temps and attens lengths mismatch.
        - TypeError: If Si_abs is not a float or if its dB conversion fails.

        Warning: 
        - per_cable_atten is set to 6dB by default.

        """
        temps = values.get('temps')
        attens = values.get('attens')
        Tq = values.get('Tq')
        Si_abs = values.get('Si_abs')
        per_cable_atten = values.get('per_cable_atten', 6)
        logger.warning(f"per_cable_atten: {per_cable_atten}")

        if len(temps) != len(attens):
            raise ValueError(
                "Length of temperatures and attenuations must match.")

        # Add 0 attenuation for 300 degrees stage if not included
        if 300 not in temps:
            temps.append(300)
            attens.append(0)

        # Add cable attenuation at each stage
        attens = [atten + per_cable_atten for atten in attens]
        zipped_dict = {temp: atten for temp, atten in zip(temps, attens)}

        # Remove entries where temperature is lower or equal to Tq
        zipped_dict = {k: v for k, v in zipped_dict.items() if k > Tq}
        if isinstance(Si_abs, float):
            si_abs_value = Atten.val(frac=(1 - Si_abs), convert_to='dB')
            if isinstance(si_abs_value, (float, int)):  # Ensure it's a numeric type
                si_abs = round(float(si_abs_value), 2)
            else:
                raise TypeError("Expected a numeric type from Atten.val")
        else:
            raise TypeError("Si_abs must be a float")
        # si_abs = round(float(Atten.val(frac=(1 - Si_abs), convert_to='dB')), 2)

        # Add Tq: Si_abs as a key-value pair
        zipped_dict[Tq] = si_abs + per_cable_atten
        # logger.critical(f"Chip attenuation: {si_abs}")

        # Sort keys and create the desired list
        sorted_keys = sorted(zipped_dict.keys(), reverse=True)
        final_list = sorted_keys #+ sorted_keys[::-1][1:] (more detailed cryo model)

        # Create values list based on the final_list
        values_list = [zipped_dict[temp] for temp in final_list]
        # Create DataFrame
        df = DataFrame({
            'temps': final_list,
            'attens': values_list
        })
        logger.info(f"Stages: {df}")

        return df
    


    def eff(self, stage_T: float):
        """
        While the function is called eff (short from efficiency), 
        this is in fact the specific power of the cyostat 
        or 1/COP (coefficient of performance).
        This will come 
        """
        if self.efficiency == 'Carnot':
            logger.warning(f"Efficiency: 'Carnot'")
            specific_power = (300 - stage_T) / stage_T
        elif self.efficiency == 'Small System':
            logger.warning(f"Efficiency: 'Small System'")
            specific_power = calculate_specific_power(stage_T)
        else:        
            raise ValueError("Unknown efficiency")
        
        return specific_power

    def heat_evacuated_at_stage(self, temp: float, power: float) -> float:
        if temp == 300:
            # Special case when temperature is 300
            all_attenuations = self.stages['attens'].tolist()
            summed_attenuation = sum(all_attenuations)
            fraction = Atten.val(dB=summed_attenuation, convert_to='frac')
        else:
            indices_for_temp = self.stages.index[self.stages['temps'] == temp].tolist(
            )
            if not indices_for_temp:
                raise ValueError(f"No stages found for temperature {temp}.")

            sums = []
            for index in indices_for_temp:

                # Calculate sum including the current index and convert to fraction
                summed_with_current = self.stages.loc[:index, 'attens'].sum()
                fraction_with_current = Atten.val(
                    dB=summed_with_current, convert_to='frac')
                assert isinstance(fraction_with_current, float)

                # Calculate sum excluding the current index and convert to fraction
                summed_excluding_current = self.stages.loc[:index -
                                                           1, 'attens'].sum()
                fraction_excluding_current = Atten.val(
                    dB=summed_excluding_current, convert_to='frac')
                assert isinstance(fraction_excluding_current, float)

                # Calculate the difference between the two fractions (with positions exchanged)
                diff = fraction_excluding_current - fraction_with_current

                sums.append(diff)

            fraction = np.sum(sums)
        assert isinstance(fraction, float)
        answer = power * fraction

        return answer

    def total_heat_evacuated(self, power: float) -> float:
        total_heat = 0
        for temp in self.stages['temps'].unique():
            new_heat = self.heat_evacuated_at_stage(temp=temp, power=power)
            total_heat += new_heat
        # logger.info(f"Total heat should be equal to power: {total_heat}")
        return total_heat

    def calculate_input_power(self, power_at_Tq: float) -> float:
        """
        Calculate the input power required to deliver a specific power at Tq,
        considering the attenuations from the top stage to just before Tq.
        """
        # Find the index of the stage where Tq is located
        tq_index = self.stages.index[self.stages['temps'] == self.Tq].tolist()
        if not tq_index:
            raise ValueError(f"No stage found for Tq temperature {self.Tq}.")

        # Sum attenuations from the top stage to just before the Tq
        # Exclude the attenuation at the Tq stage itself
        summed_attenuation = self.stages.loc[:tq_index[0] - 1, 'attens'].sum()
        attenuation_fraction = Atten.val(
            dB=summed_attenuation, convert_to='frac')
        assert isinstance(attenuation_fraction,
                          float), "attenuation_fraction must be a float"
        # Calculate the input power
        input_power = power_at_Tq / attenuation_fraction

        # logger.info(f"Calculated input power: {input_power}")
        return input_power

    def power_to_evacuate_heat_at_stage(self, temp: float, power: float) -> float:

        efficiency = self.eff(temp)
        heat_at_stage = self.heat_evacuated_at_stage(temp, power)
        return efficiency * heat_at_stage

    def total_power(self, power: float) -> float:
        total_power = 0
        for temp in self.stages['temps'].unique():
            total_power += self.power_to_evacuate_heat_at_stage(temp, power)
        return total_power

    def plot_heat_evacuated_vs_temperature(self, power: float):
        unique_temps = self.stages['temps'].unique()
        heat_evacuated = []

        for temp in unique_temps:
            heat_stage = self.heat_evacuated_at_stage(temp, power)
            heat_evacuated.append(heat_stage)

        plt.bar(unique_temps, heat_evacuated, align='center', alpha=0.7)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Heat Evacuated')
        plt.title('Heat Evacuated vs Temperature')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    class Config:
        arbitrary_types_allowed = True


def overlay_heat_evacuated_plots(cryo_instances, power):
    plt.figure(figsize=(10, 6))

    for cryo_instance in cryo_instances:
        unique_temps = cryo_instance.stages['temps'].unique()
        heat_evacuated = []

        for temp in unique_temps:
            heat_stage = cryo_instance.heat_evacuated_at_stage(temp, power)
            heat_evacuated.append(heat_stage)

        plt.bar(unique_temps, heat_evacuated, align='center',
                alpha=0.7, label=f'Tq={cryo_instance.Tq}')

    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Evacuated')
    plt.title('Heat Evacuated vs Temperature for Different Cryo Instances')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def calculate_specific_power(temp: Union[int, float]) -> float:
    # k = 236236.0  # coefficient from the logarithmic fit
    k = 3.2*1e5 
    efficiency = k * (1-temp/300) * (temp ** (-2))
    return efficiency

# def calculate_specific_power(temp: Union[int, float]) -> float:
#     k = 2.86e5  # coefficient from the logarithmic fit
#     return k * temp ** -2.09

def plot_total_power_vs_Si_abs_and_Tq(
    Tq_values: List[float],
    temps: List[float],
    attens: List[float],
    Si_abs_range: List[float],
    input_power: float = 1.0,
    save_dir: Optional[Union[Path, str]] = None
) -> None:
    plt.figure(figsize=(10, 6))

    for Tq in Tq_values:
        total_powers = []

        for Si_abs in Si_abs_range:
            # Create a Cryo instance with Si_abs=0.0 for reference power calculation
            reference_cryo = Cryo(
                Tq=Tq,
                temps=temps,
                attens=attens,
                Si_abs=0.0,
                per_cable_atten = 6
            )

            # Calculate reference power with Si_abs=0.0
            reference_power = reference_cryo.total_power(power=input_power)

            # Create a Cryo instance with the provided Si_abs for the main calculation
            main_cryo = Cryo(
                Tq=Tq,
                temps=temps,
                attens=attens,
                Si_abs=Si_abs,
                per_cable_atten = 6
            )

            # Calculate power with the provided Si_abs
            calculated_power = main_cryo.total_power(power=input_power)

            # Use the calculated power only if it's higher than the minimum power
            if calculated_power >= reference_power:
                total_powers.append(calculated_power)
                # logger.info('Calculated power used')
            else:
                total_powers.append(reference_power)
                # logger.info('Minimum power used')

        W_list = Power.val(mW=total_powers,  # pylint: disable=invalid-name
                           convert_to='W')
        plt.plot(Si_abs_range, W_list, label=f'Tq={np.round(Tq,3)}K')

    pwd = os.getcwd()

    # Define the 'results' folder path
    results_dir = os.path.join(pwd, 'results')

    plt.xlabel('Si abs')
    plt.ylabel('Total Power (W)')
    plt.title(
        f'Total Power vs Si abs for Different Tq Values and {int(input_power)}mW input power')
    plt.grid(True)
    plt.ylim(0, None)
    plt.legend()

    if save_dir is not None:
        # Specify the file path to save the PDF file
        file_name = f'Total_Power_vs_Si_abs_and{int(input_power)}mW input power saturation.pdf'
        file_path = Path(results_dir) / file_name

        # Save the figure as a PDF image
        plt.savefig(file_path, format='pdf')

    # Optionally, you can check if the PDF image was saved successfully
        if file_path.exists():
            print(f"PDF image saved to {file_path}")
        else:
            print(f"Failed to save PDF image to {file_path}")
    else:
        plt.show()

    plt.close()


def main():
    # Tq = 0.04
    # temps = [4]
    # attens = [2]
    Si_abs_range = np.linspace(0.01, 0.99, 100)

    # plot_total_power_vs_Si_abs(Tq, temps, attens, Si_abs_range)

    Tq_values = [0.006, 0.02, 0.04, 0.1, 1]
    Tq = 0.02
    temps = [4]
    attens = [3.01]
    # temps = [4, 0.8, 0.1]
    # attens = [10, 3, 0.0001]
    # Si_abs_range = np.linspace(0.0001, 0.01, 100)

    plot_total_power_vs_Si_abs_and_Tq(Tq_values, temps, attens, Si_abs_range)
    # plot_heat_per_stage_vs_Si_abs(Tq, temps, attens, Si_abs_range)

    # function to test the input power calculation
    power_at_Tq = 0.5e-3 # in Watts (or any consistent unit)
    per_cable_atten = 6
    # Create an instance of Cryo
    # cryo_instance = Cryo(Tq=Tq, temps=temps, attens=attens, Si_abs=0, per_cable_atten=per_cable_atten)
    cryo_instance = Cryo(Tq=0.02, temps=[4], attens=[3.01], Si_abs=0, per_cable_atten=30)

    print(f"Cables attenuation: {cryo_instance.per_cable_atten}dB")

    # Calculate the input power required
    input_power = cryo_instance.calculate_input_power(power_at_Tq)

    print(
        f"Input power required to deliver {power_at_Tq}W at Tq ({Tq}K): {input_power}W")
    
    cryo_instance = Cryo(Tq=0.02, temps=[4], attens=[3.01], Si_abs=0.5, per_cable_atten=30)

    print(f"Cables attenuation: {cryo_instance.per_cable_atten}dB")

    # Calculate the input power required
    input_power = cryo_instance.calculate_input_power(power_at_Tq)

    print(
        f"Input power required to deliver {power_at_Tq}W at Tq ({Tq}K): {input_power}W")

    # Example usage:
    # cryo_instance1 = Cryo(
    #     Tq=0.06,
    #     temps=[4],
    #     attens=[3.01],
    #     Si_abs=0.0
    # )

    # cryo_instance2 = Cryo(
    #     Tq=0.06,
    #     temps=[4],
    #     attens=[3.01],
    #     Si_abs=0.5
    # )

    # power = 1  # Adjust the power as needed

    # overlay_heat_evacuated_plots([cryo_instance1, cryo_instance2], power)
    # cryo_instance = Cryo(
    #     Tq=0.06,
    #     temps=[4],
    #     attens=[3.01],
    #     Si_abs=0.5
    # )

    # power = 1  # Adjust the power as needed

    # cryo_instance.plot_heat_evacuated_vs_temperature(power)

    # Example usage:
#     Tq = 0.04  # Specify Tq here
#    # Tq = 0.02
#     temps = [4]
#     attens = [3.01]
#     power = 5  # Adjust the power as needed

#     plot_heat_evacuated_vs_temperature(Tq, temps, attens, power)

    # try:
    #     cryo_instance = Cryo(
    #         Tq=100,
    #         temps=[100, 200, 150, 300],
    #         attens=[10, 20, 15, 0],
    #         Si_abs=0.5
    #     )

    #     logger.info(f"Tq: {cryo_instance.Tq}")
    #     logger.info(f"Temps: {cryo_instance.temps}")
    #     logger.info(f"Attens: {cryo_instance.attens}")
    #     logger.info(f"Si_abs: {cryo_instance.Si_abs}")
    #     logger.info(f"Stages: {cryo_instance.stages}")

    #     heat_at_stage = cryo_instance.heat_evacuated_at_stage(
    #         temp=200, power=5)
    #     logger.info(f"Heat evacuated at stage with temp 200: {heat_at_stage}")

    #     total_heat = cryo_instance.total_heat_evacuated(power=1)
    #     logger.info(f"Total heat evacuated: {total_heat}")

    #     another_cryo_instance = Cryo(
    #         Tq=5,
    #         temps=[9],
    #         attens=[3.01],
    #         Si_abs=0.00
    #     )
    #     another_cryo_instance.total_heat_evacuated(power=1)
    #     logger.info(
    #         f"Total heat evacuated should equal the: {another_cryo_instance.total_heat_evacuated(power=1)}")

    #     and_another_cryo_instance = Cryo(
    #         Tq=5,
    #         temps=[9, 20, 30],
    #         attens=[15, 20, 30],
    #         Si_abs=0.5
    #     )
    #     and_another_cryo_instance.total_heat_evacuated(power=1)
    #     logger.info(
    #         f"Total heat evacuated should equal the: {and_another_cryo_instance.total_heat_evacuated(power=1)}")

    # except Exception as e:
    #     logger.error(f"An error occurred: {e}")

    # try:
    #     cryo_instance = Cryo(
    #         Tq=0.06,
    #         temps=[4, 10],
    #         attens=[3, 3],
    #         Si_abs=0.5
    #     )

    #     logger.info(f"Tq: {cryo_instance.Tq}")
    #     logger.info(f"Temps: {cryo_instance.temps}")
    #     logger.info(f"Attens: {cryo_instance.attens}")
    #     logger.info(f"Si_abs: {cryo_instance.Si_abs}")
    #     logger.info(f"Stages: {cryo_instance.stages}")

    #     heat_at_stage = cryo_instance.heat_evacuated_at_stage(
    #         temp=4, power=5)
    #     logger.info(f"Heat evacuated at stage with temp 200: {heat_at_stage}")

    #     total_heat = cryo_instance.total_heat_evacuated(power=1)
    #     logger.info(f"Total heat evacuated: {total_heat}")

    #     # Test power_to_evacuate_heat_at_stage and total_power
    #     temp_to_test = 4
    #     power_to_test = 1
    #     power_at_stage = cryo_instance.power_to_evacuate_heat_at_stage(
    #         temp=temp_to_test, power=power_to_test)
    #     logger.info(
    #         f"Power required at stage with temp {temp_to_test}: {power_at_stage}")

    #     total_power = cryo_instance.total_power(power=power_to_test)
    #     logger.info(f"Total power required: {total_power}")

    # except Exception as e:
    #     logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

# file to contain the cryostat class
# it has:
#  Attributes:
#    - dictionary of temperatures as keys and attenuations as values
#      - for driving cables
#      - for measurement cables (not so urgent)
#    - reflectivity of the chip
#    - temperature of the qubit
#    - number of cables of different types: likely a dictionary
#
#  Methods:
#    - validation of all physical values (import temperature, attenuation)
#    - comparison of qubit temperature with dictionary, any temperature bellow the qubit temperature should be removed
#    - zipping of the temperatures as seen by the photons and phonons
#    - the functions below should be able to do this with either the driving signal, the thermal phonons or both depending on the decision of the user, how this is handles or determined should be thought out
# quite likely: do them both if both are provided, only one if one is provided, have an optional parameter if you want only one of them when both are provided, set a default logger to print in which regime it is and make sure the print statements correspond correctly through test functions. ( all of this should be completed after the rest of the points are completed for the driving cables)
#    - calculating the heat lost at each stage
#    - calculating the total heat that is lost
#    - calculating the power required to evacuate the heat at each stage
#    - calculating the total power required to evacuate the heat
#    - tests: a function that checks that the total evacuated heat is always the same regardless of distribution of the attenuators.
#
# Very important: be able to change the value of reflectivity and make sure everything else is updated accrodingly
# add test functions to this effect (in all files for that matter)


# Example usage:
