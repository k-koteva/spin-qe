"""
This module provides a class for attenuation calculations and conversions.
It includes a class `Atten` and a main function for testing purposes.
"""
import math
from typing import List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class Atten(BaseModel):
    dB: Optional[float] = Field(None, ge=0)
    perc: Optional[float] = Field(None, ge=0, le=100)
    frac: Optional[float] = Field(None, ge=0, le=1)

    def __init__(self, **data):
        input_count = sum(1 for key in ['dB', 'perc', 'frac'] if key in data)
        if input_count != 1:
            raise ValueError(
                "Exactly one of dB, perc, or frac must be provided.")
        super().__init__(**data)
        self._update_values()

    def _update_values(self):
        if self.dB is not None:
            self.frac = 10 ** (-self.dB / 10)
            self.perc = self.frac * 100
        elif self.perc is not None:
            self.frac = self.perc / 100
            self.dB = -10 * \
                math.log10(self.frac)  # pylint: disable=invalid-name
        elif self.frac is not None:
            self.perc = self.frac * 100
            self.dB = -10 * math.log10(self.frac)

    @classmethod
    def make(
            cls,
            dB: Optional[float] = None,  # pylint: disable=invalid-name
            perc: Optional[float] = None,
            frac: Optional[float] = None) -> 'Atten':
        params = {}
        input_count = sum(1 for value in [dB, perc, frac] if value is not None)
        if input_count != 1:
            raise ValueError(
                "Exactly one of dB, perc, or frac must be provided.")

        if dB is not None:
            params['dB'] = dB
        if perc is not None:
            params['perc'] = perc
        if frac is not None:
            params['frac'] = frac
        return cls(**params)

    @classmethod
    def val(
            cls,
            dB: Optional[Union[None, float, List[float]]] = None,
            perc: Optional[Union[None, float, List[float]]] = None,
            frac: Optional[Union[None, float, List[float]]] = None,
            convert_to: Optional[str] = None) -> Union[float, List[float]]:
        if dB is None and perc is None and frac is None:
            raise ValueError(
                "At least one of dB, perc, or frac must be provided.")
        input_type = 'dB' if dB is not None else (
            'perc' if perc is not None else 'frac')
        if convert_to and convert_to not in ['dB', 'perc', 'frac']:
            raise ValueError(
                "Invalid convert_to, must be one of 'dB', 'perc', 'frac'")
        if convert_to is None:
            convert_to = input_type
        value = dB if dB is not None else (perc if perc is not None else frac)

        if isinstance(value, (float, int)):
            instance_c = cls(**{input_type: float(value)})
            return getattr(instance_c, convert_to)

        elif isinstance(value, list):
            validated_values = [
                cls.val(
                    dB=v if input_type == 'dB' else None,
                    perc=v if input_type == 'perc' else None,
                    frac=v if input_type == 'frac' else None,
                    convert_to=convert_to
                )
                for v in value
            ]
            return [
                val
                for sublist in validated_values
                for val in (sublist if isinstance(sublist, list) else [sublist])
                if val is not None
            ]
        else:
            raise ValueError("Invalid input type.")


def main():
    # Using make factory method
    atten_instance = Atten.make(dB=10)
    logger.info(
        f"Created Atten instance with dB: {atten_instance.dB}, perc: {atten_instance.perc}, frac: {atten_instance.frac}")

    # Using val method for single float
    dB_val = Atten.val(dB=10)
    logger.info(f"Single float in dB: {dB_val}")

    perc_val = Atten.val(perc=10, convert_to='dB')
    logger.info(f"Single float in perc converted to dB: {perc_val}")

    frac_val = Atten.val(frac=0.1, convert_to='dB')
    logger.info(f"Single float in frac converted to dB: {frac_val}")

    # Using val method for list of floats
    dB_list = Atten.val(dB=[10, 20])
    logger.info(f"List of floats in dB: {dB_list}")

    perc_list = Atten.val(perc=[10, 20], convert_to='dB')
    logger.info(f"List of floats in perc converted to dB: {perc_list}")

    frac_list = Atten.val(frac=[0.1, 0.2], convert_to='dB')
    logger.info(f"List of floats in frac converted to dB: {frac_list}")


if __name__ == "__main__":
    main()
