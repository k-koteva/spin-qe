from typing import List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class Temp(BaseModel):
    K: Optional[float] = Field(None, gt=0, le=300)
    mK: Optional[float] = Field(None, gt=0, le=300000)

    def __init__(self, **data):
        input_count = sum(1 for key in ['K', 'mK'] if key in data)
        if input_count != 1:
            raise ValueError("Exactly one of K or mK must be provided.")
        super().__init__(**data)
        self._update_values()

    def _update_values(self):
        if self.K is not None:
            self.mK = self.K * 1000  # pylint: disable=invalid-name
        elif self.mK is not None:
            self.K = self.mK / 1000  # pylint: disable=invalid-name

    @classmethod
    def make(
            cls,
            K: Optional[float] = None,
            mK: Optional[float] = None
    ) -> 'Temp':
        params = {}
        input_count = sum(1 for value in [K, mK] if value is not None)
        if input_count != 1:
            raise ValueError("Exactly one of K or mK must be provided.")

        if K is not None:
            params['K'] = K
        if mK is not None:
            params['mK'] = mK

        return cls(**params)

    @classmethod
    def val(
        cls,
        K: Optional[Union[None, float, List[float]]  # pylint: disable=invalid-name
                    ] = None,
        mK: Optional[Union[None, float, List[float]]  # pylint: disable=invalid-name
                     ] = None,
        convert_to: Optional[str] = None
    ) -> Union[None, float, List[float]]:

        if K is None and mK is None:
            return None
        input_type = 'K' if K is not None else 'mK'
        if convert_to and convert_to not in ['K', 'mK']:
            raise ValueError("Invalid convert_to, must be one of 'K', 'mK'")

        if convert_to is None:
            convert_to = input_type

        value = K if K is not None else mK

        if isinstance(value, (float, int)):
            instance_c = cls(**{input_type: float(value)})
            return getattr(instance_c, convert_to)

        if isinstance(value, list):
            validated_values = [
                cls.val(
                    K=v if input_type == 'K' else None,
                    mK=v if input_type == 'mK' else None,
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

        raise ValueError("Invalid input type.")

# Example usage


def main():
    # Using make factory method
    temp_instance = Temp.make(K=100)
    logger.info(
        f"Created Temp instance with K: {temp_instance.K}, mK: {temp_instance.mK}")

    # Using val method for single float
    k_val = Temp.val(K=100)
    logger.info(f"Single float in K: {k_val}")

    mK_val = Temp.val(  # pylint: disable=invalid-name
        mK=100000, convert_to='K')
    logger.info(f"Single float in mK converted to K: {mK_val}")

    # Using val method for list of floats
    k_list = Temp.val(K=[100, 200])
    logger.info(f"List of floats in K: {k_list}")

    mK_list = Temp.val(mK=[100000, 200000],  # pylint: disable=invalid-name
                       convert_to='K')
    logger.info(f"List of floats in mK converted to K: {mK_list}")


if __name__ == "__main__":
    main()
