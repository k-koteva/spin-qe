from typing import Union


def calculate_specific_power(temp: Union[int, float]) -> float:
    k = 236236.0  # coefficient from the logarithmic fit
    return k * temp ** -2.1

def carnot_efficiency(temp: Union[int, float]) -> float:
    return (300 - temp) / temp
#     return 1 - (77 / temp)

print('Small system specific power at 1K: ', calculate_specific_power(1))
print('Carnot efficiency at 1K: ', carnot_efficiency(1))


