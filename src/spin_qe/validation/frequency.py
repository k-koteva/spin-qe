from typing import List, Optional, Union

from pydantic import BaseModel, Field


class Frequency(BaseModel):
    Hz: Optional[float] = Field(None, ge=0)
    kHz: Optional[float] = Field(None, ge=0)
    MHz: Optional[float] = Field(None, ge=0)
    GHz: Optional[float] = Field(None, ge=0)

    def __init__(self, **data):
        input_count = sum(1 for key in ['Hz', 'kHz', 'MHz', 'GHz'] if key in data)
        if input_count != 1:
            raise ValueError("Exactly one of Hz, kHz, MHz, or GHz must be provided.")
        super().__init__(**data)
        self._update_values()

    def _update_values(self):
        if self.Hz is not None:
            self.kHz = self.Hz / 1_000
            self.MHz = self.kHz / 1_000
            self.GHz = self.MHz / 1_000
        elif self.kHz is not None:
            self.Hz = self.kHz * 1_000
            self.MHz = self.kHz / 1_000
            self.GHz = self.MHz / 1_000
        elif self.MHz is not None:
            self.kHz = self.MHz * 1_000
            self.Hz = self.kHz * 1_000
            self.GHz = self.MHz / 1_000
        elif self.GHz is not None:
            self.MHz = self.GHz * 1_000
            self.kHz = self.MHz * 1_000
            self.Hz = self.kHz * 1_000

    @classmethod
    def make(cls, Hz: Optional[float] = None, kHz: Optional[float] = None,
             MHz: Optional[float] = None, GHz: Optional[float] = None) -> 'Frequency':
        params = {}
        input_count = sum(1 for value in [Hz, kHz, MHz, GHz] if value is not None)
        if input_count != 1:
            raise ValueError("Exactly one of Hz, kHz, MHz, or GHz must be provided.")

        if Hz is not None:
            params['Hz'] = Hz
        if kHz is not None:
            params['kHz'] = kHz
        if MHz is not None:
            params['MHz'] = MHz
        if GHz is not None:
            params['GHz'] = GHz

        return cls(**params)

    @classmethod
    def val(cls, Hz: Optional[Union[None, float, List[float]]] = None,
            kHz: Optional[Union[None, float, List[float]]] = None,
            MHz: Optional[Union[None, float, List[float]]] = None,
            GHz: Optional[Union[None, float, List[float]]] = None,
            convert_to: Optional[str] = None) -> Union[None, float, List[float]]:

        if Hz is None and kHz is None and MHz is None and GHz is None:
            return None

        input_type = 'Hz' if Hz is not None else ('kHz' if kHz is not None else ('MHz' if MHz is not None else 'GHz'))

        if convert_to and convert_to not in ['Hz', 'kHz', 'MHz', 'GHz']:
            raise ValueError("Invalid convert_to, must be one of 'Hz', 'kHz', 'MHz', 'GHz'")

        if convert_to is None:
            convert_to = input_type

        value = Hz if Hz is not None else (kHz if kHz is not None else (MHz if MHz is not None else GHz))

        if isinstance(value, (float, int)):
            instance_c = cls(**{input_type: float(value)})
            return getattr(instance_c, convert_to)

        if isinstance(value, list):
            validated_values = [
                cls.val(
                    Hz=v if input_type == 'Hz' else None,
                    kHz=v if input_type == 'kHz' else None,
                    MHz=v if input_type == 'MHz' else None,
                    GHz=v if input_type == 'GHz' else None,
                    convert_to=convert_to
                ) for v in value
            ]
            return [val for sublist in validated_values for val in (sublist if isinstance(sublist, list) else [sublist]) if val is not None]

        raise ValueError("Invalid input type.")
    
def main():
    # Using make factory method
    frequency_instance = Frequency.make(Hz=1500)
    print(
        f"Created Frequency instance with Hz: {frequency_instance.Hz}, "
        f"kHz: {frequency_instance.kHz}, MHz: {frequency_instance.MHz}, GHz: {frequency_instance.GHz}"
    )

    # Using val method for single float
    Hz_val = Frequency.val(Hz=1500)
    print(f"Single float in Hz: {Hz_val}")

    kHz_val = Frequency.val(kHz=1.5, convert_to='Hz')
    print(f"Single float in kHz converted to Hz: {kHz_val}")

    MHz_val = Frequency.val(MHz=0.0015, convert_to='Hz')
    print(f"Single float in MHz converted to Hz: {MHz_val}")

    GHz_val = Frequency.val(GHz=0.0000015, convert_to='Hz')
    print(f"Single float in GHz converted to Hz: {GHz_val}")

    # Using val method for list of floats
    Hz_list = Frequency.val(Hz=[1500, 2000])
    print(f"List of floats in Hz: {Hz_list}")

    kHz_list = Frequency.val(kHz=[1.5, 2], convert_to='Hz')
    print(f"List of floats in kHz converted to Hz: {kHz_list}")

    MHz_list = Frequency.val(MHz=[0.0015, 0.002], convert_to='Hz')
    print(f"List of floats in MHz converted to Hz: {MHz_list}")

    GHz_list = Frequency.val(GHz=[0.0000015, 0.000002], convert_to='Hz')
    print(f"List of floats in GHz converted to Hz: {GHz_list}")

    MHz_val_real = Frequency.val(kHz=500, convert_to='MHz')
    print(f"Single float in kHz converted to MHz: {MHz_val_real}")

if __name__ == "__main__":
    main()

