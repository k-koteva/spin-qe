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

