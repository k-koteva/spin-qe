from typing import List, Union, Optional
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from loguru import logger

class Temperature(BaseModel):
    """
    Temperature class for validating temperature values.
    """
    value: float = Field(..., gt=0, le=300)

    @classmethod
    def validate_array(cls, temp_values: List[float]) -> np.ndarray:
        """
        Validates an array of temperature values and returns a NumPy array of valid temperatures.
        """
        validated_temps = []
        for temp in temp_values:
            try:
                validated_temp = cls(value=temp)
                validated_temps.append(validated_temp.value)
            except ValidationError as exc:
                logger.error(f"Validation error for temperature {temp}: {exc}")
        return np.array(validated_temps)

    @classmethod
    def validate_single(cls, value: float) -> Optional[float]:
        """
        Validates a single temperature value and returns it if valid, otherwise returns None.
        """
        try:
            validated_temp = cls(value=value)
            return validated_temp.value
        except ValidationError as exc:
            logger.error(f"Validation error: {exc}")
            return None

    @classmethod
    def create_temperature(cls, value: float) -> Union['Temperature', None]:
        """
        Factory function to create a Temperature instance.
        """
        try:
            return cls(value=value)
        except ValidationError as exc:
            logger.error(f"Validation error: {exc}")
            return None

if __name__ == "__main__":
    # Sample data
    temperature_values = [1, 100, 300, -1, 400]

    # Validate and create NumPy array
    my_validated_temps = Temperature.validate_array(temperature_values)
    print("NumPy array from array validation:")
    print(my_validated_temps)

    # Validate a single value
    single_temp = Temperature.validate_single(100)
    if single_temp is not None:
        print(f"Validated single temperature value: {single_temp}")

    # Use factory function to create a Temperature instance
    temp_instance = Temperature.create_temperature(100)
    if temp_instance:
        print(f"Temperature instance created with value: {temp_instance.value}")
