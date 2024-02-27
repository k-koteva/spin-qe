from typing import List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class Power(BaseModel):
    mW: Optional[float] = Field(None, ge=0)
    W: Optional[float] = Field(None, ge=0)
    kW: Optional[float] = Field(None, ge=0)

    def __init__(self, **data):
        input_count = sum(1 for key in ['mW', 'W', 'kW'] if key in data)
        if input_count != 1:
            raise ValueError("Exactly one of mW, W, or kW must be provided.")
        super().__init__(**data)
        self._update_values()

    def _update_values(self):
        if self.mW is not None:
            self.W = self.mW / 1000  # pylint: disable=invalid-name
            self.kW = self.W / 1000  # pylint: disable=invalid-name
        elif self.W is not None:
            self.mW = self.W * 1000  # pylint: disable=invalid-name
            self.kW = self.W / 1000
        elif self.kW is not None:
            self.W = self.kW * 1000
            self.mW = self.W * 1000

    @classmethod
    def make(
            cls,
            mW: Optional[float] = None,
            W: Optional[float] = None,
            kW: Optional[float] = None
    ) -> 'Power':
        params = {}
        input_count = sum(1 for value in [mW, W, kW] if value is not None)
        if input_count != 1:
            raise ValueError("Exactly one of mW, W, or kW must be provided.")

        if mW is not None:
            params['mW'] = mW
        if W is not None:
            params['W'] = W
        if kW is not None:
            params['kW'] = kW

        return cls(**params)

    @classmethod
    def val(
        cls,
        mW: Optional[Union[None, float, List[float]]  # pylint: disable=invalid-name
                     ] = None,
        W: Optional[Union[None, float, List[float]]  # pylint: disable=invalid-name
                    ] = None,
        kW: Optional[Union[None, float, List[float]]  # pylint: disable=invalid-name
                     ] = None,
        convert_to: Optional[str] = None
    ) -> Union[None, float, List[float]]:

        if mW is None and W is None and kW is None:
            return None

        input_type = 'mW' if mW is not None else (
            'W' if W is not None else 'kW')

        if convert_to and convert_to not in ['mW', 'W', 'kW']:
            raise ValueError(
                "Invalid convert_to, must be one of 'mW', 'W', 'kW'")

        if convert_to is None:
            convert_to = input_type

        value = mW if mW is not None else (W if W is not None else kW)

        if isinstance(value, (float, int)):
            instance_c = cls(**{input_type: float(value)})
            return getattr(instance_c, convert_to)

        if isinstance(value, list):
            validated_values = [
                cls.val(
                    mW=v if input_type == 'mW' else None,
                    W=v if input_type == 'W' else None,
                    kW=v if input_type == 'kW' else None,
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


def main():  # pragma: no cover
    # Using make factory method
    power_instance = Power.make(mW=5000)
    logger.info(
        f"Created Power instance with mW: {power_instance.mW}, "
        f"W: {power_instance.W}, kW: {power_instance.kW}"
    )
    # Using val method for single float
    mW_val = Power.val(mW=5000)  # pylint: disable=invalid-name
    logger.info(f"Single float in mW: {mW_val}")

    W_val = Power.val(W=5, convert_to='mW')  # pylint: disable=invalid-name
    logger.info(f"Single float in W converted to mW: {W_val}")

    kW_val = Power.val(  # pylint: disable=invalid-name
        kW=0.005, convert_to='mW')  # pylint: disable=invalid-name
    logger.info(f"Single float in kW converted to mW: {kW_val}")

    # Using val method for list of floats
    mW_list = Power.val(mW=[5000, 6000])  # pylint: disable=invalid-name
    logger.info(f"List of floats in mW: {mW_list}")

    W_list = Power.val(  # pylint: disable=invalid-name
        W=[5, 6], convert_to='mW')
    logger.info(f"List of floats in W converted to mW: {W_list}")

    kW_list = Power.val(kW=[0.005, 0.006],  # pylint: disable=invalid-name
                        convert_to='mW')
    logger.info(f"List of floats in kW converted to mW: {kW_list}")


if __name__ == "__main__":
    main()
