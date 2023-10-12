from typing import List, Optional

import numpy as np
from loguru import logger
from pandas import DataFrame
from pydantic import BaseModel, Field, validator

from spin_qe.validation.attenuation import Atten
from spin_qe.validation.temperature import Temp


class Cryo(BaseModel):
    Tq: float
    temps: List[float]
    attens: List[float]
    Si_abs: Optional[float] = Field(0, ge=0, le=1)
    stages: DataFrame = DataFrame()

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
        temps = values.get('temps')
        attens = values.get('attens')
        Tq = values.get('Tq')
        Si_abs = values.get('Si_abs')

        if len(temps) != len(attens):
            raise ValueError(
                "Length of temperatures and attenuations must match.")

        zipped_dict = {temp: atten for temp, atten in zip(temps, attens)}

        # Remove entries where temperature is lower or equal to Tq
        zipped_dict = {k: v for k, v in zipped_dict.items() if k > Tq}

        si_abs = round(Atten.val(frac=(1 - Si_abs), convert_to='dB'), 2)

        # Add Tq: Si_abs as a key-value pair
        zipped_dict[Tq] = si_abs

        # Sort keys and create the desired list
        sorted_keys = sorted(zipped_dict.keys(), reverse=True)
        final_list = sorted_keys + sorted_keys[::-1][1:]

        # Create values list based on the final_list
        values_list = [zipped_dict[temp] for temp in final_list]

        # Create DataFrame
        df = DataFrame({
            'temps': final_list,
            'attens': values_list
        })

        return df

    def heat_evacuated_at_stage(self, temp: float, power: float) -> float:
        indices_for_temp = self.stages.index[self.stages['temps'] == temp].tolist(
        )

        if not indices_for_temp:
            raise ValueError(f"No stages found for temperature {temp}.")

        sums = []
        for index in indices_for_temp:
            summed = self.stages.loc[:index, 'attens'].sum()
            sums.append(summed)

        fractions = [Atten.val(dB=atten, convert_to='frac') for atten in sums]
        answer = power * np.sum(fractions)

        return answer

    def total_heat_evacuated(self, power: float) -> float:
        total_heat = 0
        for temp in self.stages['temps'].unique():
            new_heat = self.heat_evacuated_at_stage(temp=temp, power=power)
            logger.info(f"New heat: {new_heat}")
            total_heat += new_heat
        return total_heat

    class Config:
        arbitrary_types_allowed = True


def main():
    try:
        cryo_instance = Cryo(
            Tq=100,
            temps=[100, 200, 150, 300],
            attens=[10, 20, 15, 30],
            Si_abs=0.5
        )

        logger.info(f"Tq: {cryo_instance.Tq}")
        logger.info(f"Temps: {cryo_instance.temps}")
        logger.info(f"Attens: {cryo_instance.attens}")
        logger.info(f"Si_abs: {cryo_instance.Si_abs}")
        logger.info(f"Stages: {cryo_instance.stages}")

        heat_at_stage = cryo_instance.heat_evacuated_at_stage(
            temp=200, power=5)
        logger.info(f"Heat evacuated at stage with temp 200: {heat_at_stage}")

        total_heat = cryo_instance.total_heat_evacuated(power=1)
        logger.info(f"Total heat evacuated: {total_heat}")

        another_cryo_instance = Cryo(
            Tq=5,
            temps=[9, 300],
            attens=[3.01, 0],
            Si_abs=0
        )
        another_cryo_instance.total_heat_evacuated(power=1)
        logger.info(
            f"Total heat evacuated small: {another_cryo_instance.total_heat_evacuated(power=1)}")
        si_abs = round(Atten.val(frac=(1 - 0.1), convert_to='dB'), 2)
        logger.info(si_abs)

    except Exception as e:
        logger.error(f"An error occurred: {e}")


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
