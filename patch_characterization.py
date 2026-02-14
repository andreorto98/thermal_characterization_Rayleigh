import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats
import json
from glob import glob

from utils import get_T, get_dT


"""
Thermal Sensitivity Analysis of Rayleigh Channels
=================================================

This script estimates the thermal sensitivity of distributed Rayleigh
backscattering channels by fitting the linear dependence between
Rayleigh time shift (µs) and temperature (°C).

For each channel, it computes:
    - Thermal sensitivity (slope m in µs/°C)
    - Uncertainty on the slope (dm)
    - Pearson correlation coefficient

Temperature Reconstruction
--------------------------
Thermistor voltages are converted to temperature using calibrated
parameters (alpha, B).

Binning and Fitting
-------------------
Temperature data are either:
    - binned with fixed resolution (continuous ramp), or
    - grouped by predefined plateaus (step protocol).

Within each bin, the mean Rayleigh signal and its standard deviation
are computed. A weighted linear regression:

    signal = m T + q

is performed using an effective uncertainty that accounts for both
signal noise and propagated temperature uncertainty.

Output
------
The script generates per-channel sensitivity, statistical indicators,
and optional diagnostic plots. Results can be saved for plots and further analyses.
"""



save = False 
des = ''            

show_channels = [10, 50, 60, 70, 81]  

resolution_time = 1.0        # seconds
filename = 'Replica1_Ramp2'

# set filename to one of
#   - Replica1_Ramp2
#   - Replica1_Ramp3
#   - Replica2_Ramp2
#   - Replica2_Ramp3

CUT_AT = None 


if filename in [ 'Replica1_Ramp1', 'Replica1_Ramp2', 'Replica1_Ramp3',
                 'Replica2_Ramp1', 'Replica2_Ramp2', 'Replica2_Ramp3',
                 'Replica3_Ramp1', 'Replica3_Ramp2', 'Replica3_Ramp3',
                 
                 'SerieVAR1_Ramp1', 'SerieVAR1_Ramp2', 'SerieVAR1_Ramp3', 
                 'SerieVAR2_Ramp1', 'SerieVAR2_Ramp2', 'SerieVAR2_Ramp3',
                 'SerieVAR3_Ramp1', 'SerieVAR3_Ramp2', 'SerieVAR3_Ramp3',
                 'SerieVAR4_Ramp1', 'SerieVAR4_Ramp2', 'SerieVAR4_Ramp3',


                 'Serie2_Replica1F200_Ramp1', 'Serie2_Replica1F200_Ramp2', 'Serie2_Replica1F200_Ramp3',
                 'Serie2_Replica2F200_Ramp1', 'Serie2_Replica2F200_Ramp2', 'Serie2_Replica2F200_Ramp3',
                 'Serie2_Replica3F200_Ramp1', 'Serie2_Replica3F200_Ramp2', 'Serie2_Replica3F200_Ramp3',
                 'Serie2_F175_Ramp1', 'Serie2_F175_Ramp2', 'Serie2_F175_Ramp3',
                 
                 ]:
    
    paths = glob(f'data/{filename}_*_{resolution_time*100:.0f}_5.npz')
    print(f'data/{filename}_*_{resolution_time*100:.0f}_5.npz')
    if len(paths) == 1:
        path = paths[0]
    else:
        print("Multiple or no files found for the given filename and resolution:", paths)
        exit()

    if filename in [ 'Replica1_Ramp1', 'Replica1_Ramp2', 'Replica1_Ramp3', 'Replica1_Ramp4Post', 'Replica1_Ramp5Post', 'Replica1_Ramp6', 'Replica1_Ramp7', 'Replica1_Ramp8' ]:
        termistorA = 9
        termistorB = 10
    elif filename in [ 'Replica2_Ramp1', 'Replica2_Ramp2', 'Replica2_Ramp3' ]:
        termistorA = 11
        termistorB = 12
    elif filename in [ 'Replica3_Ramp1', 'Replica3_Ramp2', 'Replica3_Ramp3' ]:
        termistorA = 13
        termistorB = 14

    elif filename in ['SerieVAR1_Ramp1', 'SerieVAR1_Ramp2', 'SerieVAR1_Ramp3']:
        termistorA = 1
        termistorB = 2
    elif filename in ['SerieVAR2_Ramp1', 'SerieVAR2_Ramp2', 'SerieVAR2_Ramp3']:
        termistorA = 3
        termistorB = 4
    elif filename in ['SerieVAR3_Ramp1', 'SerieVAR3_Ramp2', 'SerieVAR3_Ramp3']:
        termistorA = 5
        termistorB = 6
    elif filename in ['SerieVAR4_Ramp1', 'SerieVAR4_Ramp2', 'SerieVAR4_Ramp3']:
        termistorA = 7
        termistorB = 8
    
    if filename in ['Serie2_Replica1F200_Ramp1', 'Serie2_Replica1F200_Ramp2', 'Serie2_Replica1F200_Ramp3' ]:
        termistorA = '1_2'
        termistorB = '2_2'
    if filename in ['Serie2_Replica2F200_Ramp1', 'Serie2_Replica2F200_Ramp2', 'Serie2_Replica2F200_Ramp3' ]:
        termistorA = '3_2'
        termistorB = '4_2'
    if filename in ['Serie2_Replica3F200_Ramp1', 'Serie2_Replica3F200_Ramp2', 'Serie2_Replica3F200_Ramp3' ]:
        termistorA = '5_2'
        termistorB = '6_2'
    if filename in ['Serie2_F175_Ramp1', 'Serie2_F175_Ramp2', 'Serie2_F175_Ramp3' ]:
        termistorA = '7_2'
        termistorB = '8_2'

    T_min_fit = 10
    T_max_fit = 45
    resolution_T = 1.0           # Celsius

    steps = None


if filename in [ 'NoTermReplica1_Gradini1', 'NoTermReplica2_Gradini1', 'NoTermReplica3_Gradini1',
                 'NoTermReplica1_Gradini2', 'NoTermReplica2_Gradini2', 'NoTermReplica3_Gradini2'  
               ]:

    path = f'data/{filename}_*_{resolution_time*100:.0f}_5.npz'
    paths = glob(path)
    if len(paths) == 1:
        path = paths[0]
    else:
        print("Multiple or no files found for the given filename and resolution:", paths)
        exit()

     # these termistors are not calibrated, so we will use them only to estimate the temperature variability, not the absolute temperature
    termistorA = '9_2'
    termistorB = '9_2'

    resolution_T = 1.0           # Celsius
    steps = [ 6, 15, 25, 35, 45, 55 ]

    T_min_fit = min(steps) 
    T_max_fit = max(steps)


def linear_func(x, m, q):
    return m*x + q


''' LOAD and FILTER DATA '''

data = np.load(path)


time = data['time']
V_temp = data['V_temp']                         # has shape n_termistors, n_timesteps   
Vdd_temp = data['Vdd_temp']
V_temp_std = data['V_temp_std']
Vdd_temp_std = data['Vdd_temp_std']

signal_rayl = data['signal_rayl']              # has shape n_channels, n_timesteps
signal_rayl_std = data['signal_rayl_std']

if CUT_AT is not None:
    V_temp = V_temp[:, time <= CUT_AT ]
    V_temp_std = V_temp_std[:, time <= CUT_AT ]
    Vdd_temp = Vdd_temp[ time <= CUT_AT ]
    Vdd_temp_std = Vdd_temp_std[ time <= CUT_AT ]
    signal_rayl = signal_rayl[:, time <= CUT_AT ]
    signal_rayl_std = signal_rayl_std[:, time <= CUT_AT ]
    time = time[ time <= CUT_AT ]

print("time:", time.shape[0], f'timesteps [resolution = {resolution_time} s]')
print("temperature:", V_temp.shape[0], f' termistors')
print("Rayleigh signal:", signal_rayl.shape[0], f' channels')

# filter nans
mask_temp = ~np.isnan(Vdd_temp)
mask_rail = ~np.isnan(signal_rayl[0, :])     
mask = mask_temp * mask_rail

time = time[mask]
V_temp = V_temp[:, mask]
Vdd_temp = Vdd_temp[mask]
V_temp_std = V_temp_std[:, mask]
Vdd_temp_std = Vdd_temp_std[mask]
signal_rayl = signal_rayl[:, mask]
signal_rayl_std = signal_rayl_std[:, mask]

n_timesetps = time.shape[0]
n_channels = signal_rayl.shape[0]
n_termistors = V_temp.shape[0]


assert np.isnan(Vdd_temp).sum() == 0
assert np.isnan(signal_rayl).sum() == 0
assert np.isnan(V_temp).sum() == 0
assert np.isnan(V_temp_std).sum() == 0
assert np.isnan(Vdd_temp_std).sum() == 0
assert np.isnan(signal_rayl_std).sum() == 0
assert np.isnan(time).sum() == 0
assert n_termistors >= 2

if n_termistors > 2:
    print('Warning, extra termistors found: ', n_termistors-2)
    with open('termistors_calibration_params.json', 'r') as f:
        calib_params = json.load(f)
    mean_alpha = np.mean( [ calib_params[f'{i+1}']['alpha'] for i in range(14) ] )
    mean_B = np.mean( [ calib_params[f'{i+1}']['B'] for i in range(14) ] )


''' CONVERT V_temp TO TEMPERATURES '''

# load calibration parameters
with open('termistors_calibration_params.json', 'r') as f:
    calib_params = json.load(f)
with open('termistors_calibration_params_2.json', 'r') as f:
    calib_params_2 = json.load(f)
    for termistor in calib_params_2:
        calib_params[f'{termistor}_2'] = calib_params_2[termistor]


calibration_data_A = calib_params[f'{termistorA}']
calibration_data_B = calib_params[f'{termistorB}']

T_A = get_T(V_temp[0], Vdd_temp, calibration_data_A['alpha'], calibration_data_A['B'])
T_B = get_T(V_temp[1], Vdd_temp, calibration_data_B['alpha'], calibration_data_B['B'])


ms = []
dms = []
p_vals = []
pearsons = []

for i_channel in range(n_channels):

    if True:
        dT = np.abs(T_A - T_B)/2
        T  = (T_A + T_B)/2
    else:
        dT = np.full_like(T_A, 0.1)
        T = get_T(V_temp[2], Vdd_temp, mean_alpha, mean_B)

    if i_channel in show_channels:
        fig, axs = plt.subplots(1, 3, figsize=(12,3), constrained_layout=True)

        axs[0].plot(time, T_A, label=f'Termistor {termistorA}')
        axs[0].plot(time, T_B, label=f'Termistor {termistorB}')

        if n_termistors>2:
            for i in range(2, n_termistors):
                T_i = get_T(V_temp[i], Vdd_temp, mean_alpha, mean_B)
                axs[0].plot(time, T_i, label=f'extra Termistor {i-2} (not cal)')
            axs[0].plot(time, T, label='Mean Temperature', linestyle='--', color='black')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Temperature [C]')
        axs[0].legend()

        axs[1].errorbar(time, signal_rayl[i_channel], yerr=signal_rayl_std[i_channel], label=f'Channel {i_channel}', alpha=0.5)
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel(f'Rayleigh Signal [$\mu$s]')

        axs[2].errorbar(T, signal_rayl[i_channel], yerr=signal_rayl_std[i_channel], xerr=dT, label=f'Channel {i_channel}', alpha=0.5)
        if steps is not None:
            for T_ in steps:
                axs[2].axvline(T_, color='red', linestyle='--', label=f'Ladder {T_} C')
                axs[2].axvspan(T_ - resolution_T/2, T_ + resolution_T/2, color='red', alpha=0.3)
        axs[2].set_xlabel('Temperature [C]')
        axs[2].set_ylabel(r'Rayleigh Signal [$\mu$s]')

        fig.suptitle(f'Rayleigh Signal vs Temperature - Channel {i_channel}')
        plt.show()


    if steps is None:
        Ts = np.arange(T_min_fit+resolution_T/2,T_max_fit-resolution_T/2, resolution_T)
    else:
        if i_channel == 0:
            bins = np.arange(min(T)-resolution_T, max(T)+resolution_T, resolution_T)
            hist, bins, _ = plt.hist(T, bins=bins, alpha=0.5)
            plt.xlabel('Temperature [C]')
            plt.ylabel('Counts')
            for T_ in steps:
                plt.axvline(T_, color='red', linestyle='--', label=f'Ladder {T_} C')
                plt.axvspan(T_ - resolution_T/2, T_ + resolution_T/2, color='red', alpha=0.3)
            plt.show() 
        Ts = np.array(steps)

    signal = []
    signal_dev = []
    Ts_dev = []

    masks = [] 
    for i_T_val, T_val in enumerate(Ts):
        mask_T = (T_val - resolution_T/2 <= T) * (T < T_val + resolution_T/2)
        signal.append( np.mean( signal_rayl[i_channel, mask_T] ) )
        signal_dev.append( np.std( signal_rayl[i_channel, mask_T] ) )

        dT_diff_termistors = np.mean(dT[mask_T])
        dT_variability = np.std( T[mask_T] )
        # print(dT_diff_termistors, dT_variability)
        Ts_dev.append( np.sqrt( dT_diff_termistors**2 + dT_variability**2 ) )

        if np.sum(mask_T) == 0:
            masks.append(False)
        else:
            masks.append(True)
        
        if des == '_var':
            if np.sum(mask_T) > 0:
                Ts[i_T_val] = np.mean(T[mask_T])


    masks = np.array(masks)
    Ts = Ts[masks]
    signal = np.array(signal)[masks]
    signal_dev = np.array(signal_dev)[masks]
    Ts_dev = np.array(Ts_dev)[masks]

    # pearson coefficient
    pearsons.append( stats.pearsonr(T, signal_rayl[i_channel])[0] )

    # fit
    popt0, pcov0 = curve_fit(linear_func, Ts, signal, sigma=signal_dev, absolute_sigma=False)

    m0 = popt0[0]
    sigma_eff = np.sqrt( signal_dev**2 + (m0*Ts_dev)**2 )
    
    popt, pcov = curve_fit(linear_func, Ts, signal, sigma=sigma_eff, absolute_sigma=False)
    m, q = popt
    dm, dq = np.sqrt(np.diag(pcov))

    residuals = signal - linear_func(Ts, *popt)
    chi2 = np.sum( (residuals/sigma_eff)**2 )
    dof = len(Ts) - len(popt)
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    # p_val is zero if the fit is bad
    # p_val is one if the fit is perfect (errors are overestimated)

    ms.append(m)
    dms.append(dm)
    p_vals.append(p_value)

    if i_channel in show_channels:
        plt.errorbar(Ts, signal, yerr=signal_dev, xerr=Ts_dev, label=f'Channel {i_channel}', alpha=0.5, fmt='.')
        x_fit = np.linspace(T_min_fit, T_max_fit, 100)
        y_fit = linear_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, linestyle='--', label=f'Best fit')
        plt.xlabel('Temperature [C]')
        plt.ylabel(r'Rayleigh Signal [$\mu$s]')
        plt.title(f'Rayleigh Signal vs Temperature - Channel {i_channel}')
        # np.savez(f'IMGS_paper/fit_channels/{filename}_{i_channel}{des}.npz', Ts=Ts, signal=signal, signal_dev=signal_dev, Ts_dev=Ts_dev, x_fit=x_fit, y_fit=y_fit, popt=popt, pcov=pcov, channel=i_channel)
        plt.show()



plt.errorbar(range(len(ms)), ms, yerr=dms, fmt='.', capsize=1)
plt.xlabel('Channel index')
plt.ylabel(r'Slope [$\mu$s/C]')
plt.title('Rayleigh Signal vs Temperature Slope for Different Channels')
if save:
    plt.savefig(f'IMGS/response_{filename}{des}.pdf')
plt.show()

plt.plot(range(len(p_vals)), p_vals, marker='.', linestyle='')
plt.xlabel('Channel index')
plt.ylabel('p-value of Fit')
plt.title('p-value of Rayleigh Signal vs Temperature Fit for Different Channels')
if save:
    plt.savefig(f'IMGS/pval_{filename}{des}.pdf')
plt.show()


mean_pearson = np.mean(pearsons)
print(f'Mean Pearson r across channels: {mean_pearson:.4f}')
plt.axhline(mean_pearson, color='C1', linestyle='--', label=f'Mean Pearson r = {mean_pearson:.2f}')
plt.plot(range(len(pearsons)), pearsons, marker='.', linestyle='')
plt.xlabel('Channel index')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('Pearson Correlation between Rayleigh Signal and Temperature for Different Channels')
plt.show()


''' SAVE RESULTS '''
if save: 
    output_data = {
        'channels': list(range(n_channels)),
        'ms': ms,
        'dms': dms,
        'p_vals': p_vals,
    }

    output_path = f'IMGS/response_{filename}{des}.npz'
    np.savez(output_path, **output_data)