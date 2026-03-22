import numpy as np

#Material constants for S355

SIGMA_U = 510  #Ultimte tensile strength in MPa
B = 8          #Fatigue Exponent
C = 2.56e24    #Basquin coefficient
GAMMA = 0.5    #Walker exponent
ALPHA = 0.0003 #Temperature coefficient
BETA = 1.2     #Temperature exponent
T0 = 20        #Reference temperature in °C
SIGMA_N = 0.1  #Noise standard deviation


def fatigue_life(sigma_a, sigma_m, T, R, k_s, noisy=True):
    
    #Goodman correction
    sigma_eff = sigma_a / (1 - sigma_m / SIGMA_U)
    #Temperature correction
    sigma_eff *= (1 + ALPHA * (T - T0)**BETA)
    #Walker correction
    sigma_eff *= (1 - R)**GAMMA
    #Surface roughness correction
    sigma_eff *= (1 / k_s)
    
    #Basquin
    N = C * sigma_eff**(-B)
    
    #Noise
    if noisy:
        N_Log = np.log10(N) + np.random.normal(0, SIGMA_N)
    else:
        N_Log = np.log10(N)
    
    return N_Log
    



