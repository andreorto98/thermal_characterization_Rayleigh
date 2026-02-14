import numpy as np



def get_T(V, Vdd, alpha, B):
    T_0kelvin = 273.15
    return 1./( 1/(25+T_0kelvin) - 1/B * np.log( Vdd/V-1 ) - alpha/B ) - T_0kelvin




def get_dT(V, Vdd, alpha, B,
              V_std, Vdd_std, alpha_std, B_std,
              cov_alpha_B):

    # CHECK but seems ok

    Tref = 25 + 273.15
    T0 = 273.15

    f = 1/Tref - (1/B)*np.log(Vdd/V - 1) - alpha/B
    T = 1/f - T0

    # derivatives of f
    df_dV = (1/B) * Vdd / (V * (Vdd - V))
    df_dVdd = -(1/B) * 1/(Vdd - V)
    df_dalpha = -1/B
    df_dB = (np.log(Vdd/V - 1) + alpha) / B**2

    # uncorrelated part
    var_T = (1/f**4) * (
        (df_dV * V_std)**2 +
        (df_dVdd * Vdd_std)**2 +
        (df_dalpha * alpha_std)**2 +
        (df_dB * B_std)**2
    )

    # covariance term
    dT_dalpha = 1/(f**2 * B)
    dT_dB = -(np.log(Vdd/V - 1) + alpha)/(f**2 * B**2)

    var_T += 2 * dT_dalpha * dT_dB * cov_alpha_B

    return np.sqrt(var_T)
