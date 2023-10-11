"""
This module defines the Attenuation class for validating and converting attenuation values.
It also provides a factory function for creating instances of the Attenuation class.
"""

from typing import Optional, Union, List
import math
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from loguru import logger


class Attenuation(BaseModel):
    """
    Attenuation class for validating attenuation values.
    """
    dB: Optional[float] = Field(None, ge=0)
    perc: Optional[float] = Field(None, ge=0, le=100)
    frac: Optional[float] = Field(None, ge=0, le=1)

    def __post_init_post_parse__(self):
        attrs = [self.dB, self.perc, self.frac]
        if sum(attr is not None for attr in attrs) != 1:
            raise ValueError("Exactly one of 'dB', 'perc', or 'frac' must be provided.")
        
        if self.dB is not None:
            self.frac = 10 ** (-self.dB / 10)
            self.perc = self.frac * 100
        elif self.perc is not None:
            self.frac = self.perc / 100
            self.dB = -10 * math.log10(self.frac) #noqa
        elif self.frac is not None:
            self.perc = self.frac * 100
            self.dB = -10 * math.log10(self.frac)
    
    @classmethod
    def create_attenuation(cls, dB: Optional[float] = None, perc: Optional[float] = None, frac: Optional[float] = None) -> 'Attenuation':
        """
        Factory function to create an Attenuation instance.
        """
        try:
            return cls(dB=dB, perc=perc, frac=frac)
        except (ValidationError, ValueError) as exc:
            logger.error(f"Validation error: {exc}")
            return None
        
    @classmethod
    def validate_and_convert(cls, value: float, value_type: str, convert_to: Optional[str] = None) -> Union[float, None]:
        """
        Validate and optionally convert a value.
        """
        try:
            if value_type == 'dB':
                att_instance = cls(dB=value)
            elif value_type == 'perc':
                att_instance = cls(perc=value)
            elif value_type == 'frac':
                att_instance = cls(frac=value)
            else:
                raise ValueError("Invalid value_type. Choose from 'dB', 'perc', or 'frac'.")

            if convert_to:
                return getattr(att_instance, convert_to)
            else:
                return getattr(att_instance, value_type)
        except (ValidationError, ValueError) as exc:
            logger.error(f"Validation error: {exc}")
            return None
        
    @classmethod
    def validate_and_convert_array(cls, values: Union[List[float], np.ndarray], value_type: str, convert_to: Optional[str] = None) -> Union[List[float], np.ndarray, None]:
        """
        Validate and optionally convert an array of values.
        """
        try:
            output = []
            for value in values:
                validated_value = cls.validate_and_convert(value, value_type, convert_to)
                if validated_value is not None:
                    output.append(validated_value)

            if isinstance(values, np.ndarray):
                return np.array(output)
            return output
        except (ValidationError, ValueError) as exc:
            logger.error(f"Validation error: {exc}")
            return None
        


def main():
    """
    Main function to demonstrate the usage of the Attenuation class.
    """
    att_instance = Attenuation.create_attenuation(dB=3)
    if att_instance:
        print(f"Attenuation instance created with dB: {att_instance.dB}, perc: {att_instance.perc}, frac: {att_instance.frac}")

    att_instance = Attenuation.create_attenuation(perc=50)
    if att_instance:
        print(f"Attenuation instance created with dB: {att_instance.dB}, perc: {att_instance.perc}, frac: {att_instance.frac}")

    att_instance = Attenuation.create_attenuation(frac=0.5)
    if att_instance:
        print(f"Attenuation instance created with dB: {att_instance.dB}, perc: {att_instance.perc}, frac: {att_instance.frac}")

    att_instance = Attenuation.create_attenuation()
    if att_instance:
        print("Attenuation instance created")
    else:
        print("Failed to create Attenuation instance")

    print(Attenuation.validate_and_convert(3, 'dB', 'frac'))  # Should print the fraction corresponding to dB=3
    print(Attenuation.validate_and_convert(50, 'perc'))  # Should print 50 as it's already in percentage
    print(Attenuation.validate_and_convert(0.5, 'frac', 'dB'))  # Should print the dB value corresponding to fraction=0.5
    print(Attenuation.validate_and_convert_array([3, 4, 5], 'dB', 'frac'))  # Should print the fractions corresponding to dB values
    print(Attenuation.validate_and_convert_array(np.array([50, 60]), 'perc'))  # Should print a NumPy array with [50, 60]


if __name__ == "__main__":
    main()
