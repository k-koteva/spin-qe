# Marco Fellous-Asiani - 09-01-2023
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from spin_qe.components.cryostat import Cryo

# ------------- PARAMETERS ------------------------
# /!\ Below, I forced the parameters to be the same as in the arxiv paper.
# I suggest you to take the same. Details behind the model are in the supp
# material.
    
    
# Length of the cable over which the heat flow is computed.
# In practice, it corresponds to the length between two consecutive stages
# interstages_length=1 
    
# All the parameters below are related to choosing a specific cable model
# For wire_type=MIXT, we use a different cable for T>Tc_super (conventional) 
# and for T<Tc_super (superconductor)
Tc_super=4 # critical temperature of the SC wire
wire_type='MIXT' # 304SS_NISQ (NISQ coaxial) or 304SS_UT (ultra tiny NISQ) or KAPTON (Kapton around NbTi wire) or MIXT, if MIXT see below
    
# wire_lowT is the cable model used for T<Tc_super. In practice it can be
# superconducting (low heat flow). In such a case, only the insulator around 
# the cable is modelled.
wire_lowT='KAPTON' # if MIXT, this wire will be used for T>10K: 304SS_NISQ or 304SS_UT or KAPTON. If not: variable IGNORED
    
# wire_highT is the cable model used for T>Tc_super. In practice it is a conventional coaxial cable.
wire_highT='304SS_UT' # if MIXT, this wire will be used for T<10K: 304SS_NISQ or 304SS_UT or KAPTON. If not: variable IGNORED
    
# ------------------- FUNCTIONS -----------------------
def integratedlambda_insulator(T1,T2):
    # We neglect the heat flow of insulator vs the heat flow of metal.
    # To change if we want to be super accurate.
    return 0
    
def integratedlambda_304SS(T1,T2): # coaxial cable
    # If T1<T2: positive value
    if(T1<T2):
        sign=1
    else:
        T=T2
        T2=T1
        T1=T
        sign=-1
    if T1 <= 4 and T2 <= 4:
        res=-0.0340495*T1**2 + 0.0340495*T2**2 #Stainless steel
        return res*sign
    if T1 <= 4 and T2 > 4:
        res=-10.741705101665252 - 0.034049499999999996*T1**2 + 178885.33391867118*T2**1.25 - 488022.76610419207*T2**1.2857142857142858 + 481020.28641944804*T2**1.3333333333333333 - 206250.58329655445*T2**1.4 + 36354.211081651556*T2**1.5 - 1990.9339948878915*T2**1.6666666666666667 + 14.80937244446968*T2**2 - 0.0004006687602664789*T2**3
        return res*sign
    if T1 > 4 and T2 > 4:
        res=-178885.33391867118*T1**1.25 + 488022.766104192*T1**1.2857142857142858 - 481020.2864194481*T1**1.3333333333333333 + 206250.58329655448*T1**1.4 - 36354.21108165155*T1**1.5 + 1990.9339948878915*T1**1.6666666666666667 - 14.809372444469679*T1**2 + 0.0004006687602664789*T1**3 + 178885.33391867118*T2**1.25 - 488022.766104192*T2**1.2857142857142858 + 481020.2864194481*T2**1.3333333333333333 - 206250.58329655448*T2**1.4 + 36354.21108165155*T2**1.5 - 1990.9339948878915*T2**1.6666666666666667 + 14.809372444469679*T2**2 - 0.0004006687602664789*T2**3
        return res*sign
    

def integratedlambda_304SS_super(T1,T2): # coaxial cable
    # If T1<T2: positive value
    if(T1<T2):
        sign=1
    else:
        T=T2
        T2=T1
        T1=T
        sign=-1
    if T1 <= 4 and T2 <= 4:
        res=-0.002948717948717949*T1**1.56 + 0.002948717948717949*T2**1.56 # Superconducting cable
        return res*sign
    if T1 <= 4 and T2 > 4:
        res=-10.741705101665252 - 0.034049499999999996*T1**2 + 178885.33391867118*T2**1.25 - 488022.76610419207*T2**1.2857142857142858 + 481020.28641944804*T2**1.3333333333333333 - 206250.58329655445*T2**1.4 + 36354.211081651556*T2**1.5 - 1990.9339948878915*T2**1.6666666666666667 + 14.80937244446968*T2**2 - 0.0004006687602664789*T2**3
        return res*sign
    if T1 > 4 and T2 > 4:
        res=-178885.33391867118*T1**1.25 + 488022.766104192*T1**1.2857142857142858 - 481020.2864194481*T1**1.3333333333333333 + 206250.58329655448*T1**1.4 - 36354.21108165155*T1**1.5 + 1990.9339948878915*T1**1.6666666666666667 - 14.809372444469679*T1**2 + 0.0004006687602664789*T1**3 + 178885.33391867118*T2**1.25 - 488022.766104192*T2**1.2857142857142858 + 481020.2864194481*T2**1.3333333333333333 - 206250.58329655448*T2**1.4 + 36354.21108165155*T2**1.5 - 1990.9339948878915*T2**1.6666666666666667 + 14.809372444469679*T2**2 - 0.0004006687602664789*T2**3
        return res*sign
    
# integral of thermal conductivity for Kapton between T1 and T2
def integratedlambda_Kapton(T1,T2): # Microstrip line
    # If T1<T2: positive value
    if(T1<T2):
        sign=1
    else:
        T=T2
        T2=T1
        T1=T
        sign=-1
    if T1==T2:
        return 0
    if T1 <= 4 and T2 <= 4:
        res = (-0.0681*T1**2 + 0.0681*T2**2)/2 # steel
        return res*sign
    if T1 <= 4 and T2 > 4:
        res=0.002100168817852844 - 0.002948717948717949*T1**1.56 + 0.0015135899767606345*T2**1.9794
        return res*sign
    if T1 > 4 and T2 > 4:
        res=-0.0015135899767606345*T1**1.9794 + 0.0015135899767606345*T2**1.9794
        return res*sign
    
def integratedlambda_Kapton_super(T1,T2):
    # If T1<T2: positive value
    if(T1<T2):
        sign=1
    else:
        T=T2
        T2=T1
        T1=T
        sign=-1
    if T1==T2:
        return 0
    if T1 <= 4 and T2 <= 4:
        res=-0.002948717948717949*T1**1.56 + 0.002948717948717949*T2**1.56 # Superconducting cable 
        return res*sign
    if T1 <= 4 and T2 > 4:
        res=0.002100168817852844 - 0.002948717948717949*T1**1.56 + 0.0015135899767606345*T2**1.9794
        return res*sign
    if T1 > 4 and T2 > 4:
        res=-0.0015135899767606345*T1**1.9794 + 0.0015135899767606345*T2**1.9794
        return res*sign
    
# def integratedlambda_Kapton(T1, T2):
#     if T1 < T2:
#         sign = 1
#     else:
#         T1, T2 = T2, T1
#         sign = -1
#     if T1 == T2:
#         return 0

    # # Define Z[T] for the exact formula
    # def Z(T):
    #     logT = np.log10(T)
    #     return (
    #         5.73101
    #         - 39.5199 * logT
    #         + 79.9313 * logT**2
    #         - 83.8572 * logT**3
    #         + 50.9157 * logT**4
    #         - 17.9835 * logT**5
    #         + 3.42413 * logT**6
    #         - 0.27133 * logT**7
    #     )
    
    # # Calculate lambda for T1 and T2 using the exact formula
    # lambda_T1 = 10 ** Z(T1)
    # lambda_T2 = 10 ** Z(T2)

    # # Approximate the integral as (lambda_T1 + lambda_T2) / 2 * (T2 - T1)
    # res = (lambda_T1 + lambda_T2) / 2 * (T2 - T1)
    # return res * sign
# Heat flow per wires:   
def heatFlow(T1,T2,r1,integrated_lambda1,r2,integrated_lambda2,r3,integrated_lambda3, interstages_length):
    # This function integrate Fourier law between on [T1,T2], T1 being the
    # lower bound of the integral. Thus T1<T2 yields a positive result which
    # corresponds to the heat received by the end of the cable @T1.
    
    # The wire is considered as coaxial having the radius and thermal 
    # conductivity of the different parameters given as input. 
    # As well as the cable length constants.wire_length
    Q=(1/interstages_length)*(integrated_lambda1(T1,T2)*np.pi*r1**2+integrated_lambda2(T1,T2)*np.pi*(r2**2-r1**2)+integrated_lambda3(T1,T2)*np.pi*(r3**2-r2**2))
    return Q
    
def conduction_T1_To_T2_fixedwire(T1,T2,wire):
    interstages_length = 1
    if(wire=="304SS_NISQ"):
        # We do as if we had a cylinder of 1mm of radius for the metal
        # (good order of magnitude of typical equivalent radius for UT-085-SS-SS)
        r1=0.511*10**(-3)/2 # Inner conductor radius
        integrated_lambda1=integratedlambda_304SS #Inner conductor thermal conductivity
        r2=1.676*10**(-3)/2 # Insulator radius
        integrated_lambda2=integratedlambda_insulator # Insulator thermal conductivity
        r3=2.197*10**(-3)/2 #Outter conductor radius
        integrated_lambda3=integratedlambda_304SS #Outter conductor thermal conductivity
    elif(wire=="304SS_UT"):
        # Used for coaxial cable in Marco's paper
        # We do as if we had a cylinder of 1mm of radius for the metal
        # (good order of magnitude of typical equivalent radius for UT-085-SS-SS)
        r1=0.2*10**(-3)/2 # Inner conductor radius
        integrated_lambda1=integratedlambda_304SS #Inner conductor thermal conductivity
        r2=0.66*10**(-3)/2 # Insulator radius
        integrated_lambda2=integratedlambda_insulator # Insulator thermal conductivity
        r3=0.86*10**(-3)/2 #Outter conductor radius
        integrated_lambda3=integratedlambda_304SS #Outter conductor thermal conductivity
    elif(wire=="304SS_UT_S"):
        # Used for coaxial cable in Marco's paper
        # We do as if we had a cylinder of 1mm of radius for the metal
        # (good order of magnitude of typical equivalent radius for UT-085-SS-SS)
        r1=0.2*10**(-3)/2 # Inner conductor radius
        integrated_lambda1=integratedlambda_304SS_super #Inner conductor thermal conductivity
        r2=0.66*10**(-3)/2 # Insulator radius
        integrated_lambda2=integratedlambda_insulator # Insulator thermal conductivity
        r3=0.86*10**(-3)/2 #Outter conductor radius
        integrated_lambda3=integratedlambda_304SS_super #Outter conductor thermal conductivity
    elif(wire=="304SS_UUT"):
        # We do as if we had a cylinder of 1mm of radius for the metal
        # (good order of magnitude of typical equivalent radius for UT-085-SS-SS)
        r1=0.08*10**(-3)/2 # Inner conductor radius
        integrated_lambda1=integratedlambda_304SS #Inner conductor thermal conductivity
        r2=0.26*10**(-3)/2 # Insulator radius
        integrated_lambda2=integratedlambda_insulator # Insulator thermal conductivity
        r3=0.33*10**(-3)/2 #Outter conductor radius
        integrated_lambda3=integratedlambda_304SS #Outter conductor thermal conductivity
    elif(wire=="KAPTON"):
        r1=np.sqrt((1.3*10**(-9)/np.pi)) # code used for microstrip lines, the radius is calculated to give the same cross section
        integrated_lambda1=integratedlambda_Kapton #Inner conductor thermal conductivity
        r2=r1 # Insulator radius
        integrated_lambda2=integratedlambda_Kapton # Insulator thermal conductivity
        r3=r1 #Outter conductor radius
        integrated_lambda3=integratedlambda_Kapton#Outter conductor thermal conductivity
    elif(wire=="KAPTON_S"):
        r1=np.sqrt((1.3*10**(-9)/np.pi)) # code used for microstrip lines, the radius is calculated to give the same cross section
        integrated_lambda1=integratedlambda_Kapton_super #Inner conductor thermal conductivity
        r2=r1 # Insulator radius
        integrated_lambda2=integratedlambda_Kapton_super # Insulator thermal conductivity
        r3=r1 #Outter conductor radius
        integrated_lambda3=integratedlambda_Kapton_super#Outter conductor thermal conductivity
    elif(wire=="KAPTON_stages"):
        interstages_length = 0.2
                # r1=np.sqrt((1.3*10**(-9)/np.pi))
        r1=np.sqrt((10**(-4)/np.pi))
        integrated_lambda1=integratedlambda_Kapton #Inner conductor thermal conductivity
        r2=r1 # Insulator radius
        integrated_lambda2=integratedlambda_Kapton # Insulator thermal conductivity
        r3=r1 #Outter conductor radius
        integrated_lambda3=integratedlambda_Kapton#Outter conductor thermal conductivity

    else:
        raise NameError('Wrong wire type specified ?')
        
    return heatFlow(T1,T2,r1,integrated_lambda1,r2,integrated_lambda2,r3,integrated_lambda3, interstages_length)
    
def conduction_T1_To_T2(T1,T2):
    # This function returns the heat flow between T1 and T2.
    # It gives a positive value if T1<T2
    # Thus it corresponds to the heat received by the part at T1 (counted as
    # positive if T1 really receives it)
    
    # Wire characteristics
    # In this example we consider inner conductor = outter conductor.
    # We also neglect the dissipation coming from the dielectric in the coax
    if(T1>T2):
        raise NameError('T1>T2 in wire conductivity: error, T1=' + str(T1) + 'T2=' + str(T2)) 
    if(wire_type=="MIXT"):
        if(T2<Tc_super):
            heat=conduction_T1_To_T2_fixedwire(T1,T2,wire_lowT)
        elif(T1>Tc_super):
            heat=conduction_T1_To_T2_fixedwire(T1,T2,wire_highT)
        else:
            heat=conduction_T1_To_T2_fixedwire(T1,Tc_super,wire_lowT)+conduction_T1_To_T2_fixedwire(Tc_super,T2,wire_highT)
    else:
        heat=conduction_T1_To_T2_fixedwire(T1,T2,wire_type)
    
    return heat
    

# Some examples.
# The only function to be used in practice is conduction_T1_To_T2(T1,T2)
# T1 must be lower than T2 and it returns a positive value.
    
#Heat flow between 200K and 300K for the cable model specified above
# print(conduction_T1_To_T2(200,300))
    
#Heat flow between 0K and 300K for the cable model specified above.
# It is good to keep in mind the order of magnitude (~1mW)
# print(conduction_T1_To_T2(0,300))


# Assuming all previous code is available and imported

def sum_conduction_heat(cryo: Cryo) -> float:
    total_heat = 0.0
    
    # Ensure the temperatures are unique and sorted in ascending order
    unique_sorted_temps = sorted(cryo.stages['temps'].unique())
    
    # Iterate through the sorted temperatures and calculate conduction heat
    for i in range(len(unique_sorted_temps) - 1):
        T1 = unique_sorted_temps[i]
        T2 = unique_sorted_temps[i + 1]
        heat = conduction_T1_To_T2(T1, T2)
        print(f"Heat flow between {T1}K and {T2}K: {heat}")
        total_heat += heat
    print(f"unique_sorted_temps: {unique_sorted_temps}")
    print(f"total_heat: {total_heat}")
    return total_heat

def sum_conduction_power(cryo: Cryo, wire_type) -> float:
    total_power = 0.0
    
    unique_sorted_temps = sorted(cryo.stages['temps'].unique())
    
    for i in range(len(unique_sorted_temps)):
        if i == len(unique_sorted_temps) - 1:
            break
        T1 = unique_sorted_temps[i]
        T2 = unique_sorted_temps[i + 1]
        # conduction_heat = conduction_T1_To_T2(T1, T2)
        conduction_heat = conduction_T1_To_T2_fixedwire(T1, T2, wire=wire_type)
        
        if i == 0:
            total_power += conduction_heat * cryo.eff(T1)
        else:
            T0 = unique_sorted_temps[i - 1]
            # previous_conduction_heat = conduction_T1_To_T2(T0, T1)
            previous_conduction_heat = conduction_T1_To_T2_fixedwire(T0, T1, wire=wire_type)
            total_power += (conduction_heat - previous_conduction_heat) * cryo.eff(T1)

    return total_power


def sum_emptycryo_power(cryo: Cryo) -> float:
    return sum_conduction_power(cryo=cryo, wire_type="KAPTON_stages")

    # conduction_T1_To_T2_fixedwire(T1, 300, wire="KAPTON")

def main():
    print(conduction_T1_To_T2(200,300))
    print(conduction_T1_To_T2(0,300))

    logger.info(f'Example cryo is is a test log message: {conduction_T1_To_T2(300,300)}')
    # Example cryo is is a test log message", conduction_T1_To_T2(300,300))instance
    cryo_instance = Cryo(Tq=0.04, temps=[0.1, 4, 300], attens=[2, 10, 30], Si_abs=0.0, per_cable_atten=5)
    total_heat = sum_conduction_heat(cryo_instance)
    cryo_instance.plot_heat_evacuated_vs_temperature(1.0)
    print(f"Total conduction heat: {total_heat}")

def plot_conduction_kapton_to_300K(temps, T2=300):
    conduction_values = []
    
    for T1 in temps:
        conduction = conduction_T1_To_T2_fixedwire(T1, T2, wire="KAPTON")
        conduction_values.append(conduction)

    # Plotting the conduction values
    plt.plot(temps, conduction_values, marker="o", label="Kapton conduction (1cm diameter to 300K)")
    plt.xlabel("Starting Temperature (K)")
    plt.ylabel("Conduction Heat Flow to 300K")
    plt.legend()
    plt.title("Conduction Heat Flow for Kapton Cable with 1cm Diameter to 300K")
    plt.show()

def main2():
    # List of specific temperatures for calculation
    temps = [0.007, 0.1, 0.8, 4, 50]
    plot_conduction_kapton_to_300K(temps)


if __name__ == "__main__":
    main2()