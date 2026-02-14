import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy.stats import wilcoxon, mannwhitneyu  


analisi = 4
x_multiplicator = 2.64       # mm per channel, to convert channel numbers to mm

"""
Spatial Response and Reproducibility Analysis of Rayleigh Sensitivity
======================================================================

This script post-processes the per-channel thermal sensitivities
(previously computed) to compare spatial response profiles across:

    1) Different patch sizes (1x, 1.5x, 2x)
    2) With vs without thermistor compensation
    3) Repeated trials (same replica)
    4) Different replicas (same geometry)

Input
-----
The script loads precomputed results from:

    data/response_<filename>.npz

Each file contains:
    - channels : channel indices
    - ms       : thermal slopes (µs/K)
    - dms      : slope uncertainties

Channel indices are converted to spatial position (mm) using a fixed
scale factor.


Analysis Modes
--------------

Analisi 1 - Size Scaling
    Extracts peak response for each patch size and fits a linear model
    versus size factor. A χ² goodness-of-fit test is computed.

Analisi 2 - With vs Without Thermistors
    Computes spatial skewness of the response distribution inside the
    patch region. Groups are compared using a Mann–Whitney U test.

Analisi 3 - Trial Repeatability
    Compares two trials of the same replica. Channel-wise absolute and
    relative differences are computed, and distributions are analyzed
    separately for patch and non-patch regions.

Analisi 4 - Replica Reproducibility
    Same as Analisi 3, but comparing different replicas with identical
    geometry.

Computed Metrics
----------------
Depending on the selected analysis:

    - Peak thermal response
    - Spatial skewness of sensitivity distribution
    - Absolute delta between curves
    - Relative delta (|Δ| / mean)
    - Statistical test p-values

Output
------
The script generates spatial profiles and histograms summarizing:

    - Scaling behavior with patch size
    - Symmetry effects due to temperature compensation
    - Trial-to-trial variability
    - Replica-to-replica reproducibility

This tool provides a statistical and spatial characterization of
Rayleigh-based thermal sensitivity profiles in distributed fiber-optic
e-skin systems.
"""

colors = None
linestyles = None
xlims = None
ylims = None
size_figure_factor = 1
alpha_ = None
offset_x = 0
remove_baseline = True


cmap = cm.get_cmap('tab10')


if analisi in [ 1, '1a', '1b', '1c' ]:
    legends = { 
                'Replica1_Ramp2':             'Replica 1, size 1x',
                'Replica2_Ramp2':             'Replica 2, size 1x',
                'Replica1_F150_Ramp2':        'Replica 1, size 1.5x',
                'Replica2_F150_Ramp2':        'Replica 2, size 1.5x',
                'Serie2_Replica1F200_Ramp2':  'Replica 1, size 2x',
                'Serie2_Replica2F200_Ramp2':  'Replica 2, size 2x',
        }
    
if analisi in [ 3, '3a', 4 ]:
    legends = { 
                'Replica1_Ramp2':             'Replica 1, trial 1',
                'Replica1_Ramp3':             'Replica 1, trial 2',
                'Replica2_Ramp2':             'Replica 2, trial 1',
                'Replica2_Ramp3':             'Replica 2, trial 2',
                'Replica1_F150_Ramp2':        'Replica 1, trial 1, size 1.5x',
                'Replica1_F150_Ramp3':        'Replica 1, trial 2, size 1.5x',
                'Replica2_F150_Ramp2':        'Replica 2, trial 1, size 1.5x',
                'Replica2_F150_Ramp3':        'Replica 2, trial 2, size 1.5x',
                'Serie2_Replica1F200_Ramp2':  'Replica 1, trial 1, size 2x',
                'Serie2_Replica1F200_Ramp3':  'Replica 1, trial 2, size 2x',
                'Serie2_Replica2F200_Ramp2':  'Replica 2, trial 1, size 2x',
                'Serie2_Replica2F200_Ramp3':  'Replica 2, trial 2, size 2x',
    }


if analisi == 1:
    ''' ANALISI 1: PEAK AS A FUNCTION OF SIZE '''
    savefig_name = 'IMGS_paper/size_comparison_all.pdf'
    groups = [
        [ 'Replica1_Ramp2',         # 1x
          'Replica2_Ramp2',
        ],
        [ 'Replica1_F150_Ramp2',        # 1.5 x
          'Replica2_F150_Ramp2',
        ],
        [ 'Serie2_Replica1F200_Ramp2',  # 2x
          'Serie2_Replica2F200_Ramp2',
        ]
    ]

    offsets = [0] * len(groups)
    peak_factors        = [ 1, 1.5, 2 ]
    peak_responses      = []
    peak_responses_std  = []
    colors = [ ('C0', 1), ('C1', 1), ('C2', 1) ]

elif analisi == '1a':
    savefig_name = 'IMGS_paper/size_comparison_1.pdf'
    size_figure_factor = 0.75
    groups = [
       [ 'Replica1_Ramp2', ],          
       [ 'Replica2_Ramp2', ],      
    ]
    colors = [ ('C0', 1), ('C0', 0.5) ]
    offsets = [ -19, -6 ]
    # xlims = [ 55, 55+255 ]      # + 255
    xlims = [ 60, 60+276 ]      # + 255
    ylims = [-1.5, 24]

elif analisi == '1b':    
    savefig_name = 'IMGS_paper/size_comparison_2.pdf'
    size_figure_factor = 0.75
    groups = [
       [ 'Replica1_F150_Ramp2', ],    
       [ 'Replica2_F150_Ramp2', ],  
    ]
    colors = [ ('C1', 1), ('C1', 0.5) ]
    offsets = [ 84, 65 ]
    offset_x = -200
    xlims = [ 264+offset_x, 264+276+offset_x ]    # 432
    ylims = [-1.5, 24]
    
elif analisi == '1c':
    savefig_name = 'IMGS_paper/size_comparison_3.pdf'
    size_figure_factor = 0.75
    groups = [
       [ 'Serie2_Replica1F200_Ramp2', ], 
       [ 'Serie2_Replica2F200_Ramp2', ], 
    ]
    colors = [ ('C2', 1), ('C2', 0.5) ]
    offsets = [ 143, 188 ]
    # xlims = [ 520, 775 ]      # + 255
    offset_x = -510
    xlims = [ 570+offset_x, 846+offset_x ]      # + 276
    ylims = [-1.5, 24]

elif analisi == 2:
        ''' ANALISI 2: TERMISTOR VS NO TERMISTOR, symmetry analyses '''
        savefig_name = 'IMGS_paper/with_without_term.pdf'

        groups = [
            [ 'NoTermReplica1_Gradini2', ], # without termistors
            [ 'NoTermReplica2_Gradini2', ], # without termistors
            [ 'NoTermReplica3_Gradini2', ], # without termistors

            [ 'Replica1_Ramp2', ],          # with termistors
            [ 'Replica2_Ramp2', ],          # with termistors
        ]

        colors     = [ ('grey', 1), ('grey', 0.7), ('grey', 0.4),       ('C0', 1), ('C0', 0.5),    ]
        linestyles = [  '--', '--', '--',                               '-', '-',                  ]

        offsets = [ 0-0, -2-0, -12-0, -19, -6, ]
        termistors = [ False, False, False, True, True ]
        # xlims = [ 58, 202 ]
        xlims = [ 70, 214 ]  

        if False:
            savefig_name = 'IMGS_paper/with_without_term_all.pdf'
            groups.extend ([
                [ 'Replica1_F150_Ramp2',  ],
                [ 'Replica2_F150_Ramp2' ],
                [ 'Serie2_Replica1F200_Ramp2' ],
                [ 'Serie2_Replica2F200_Ramp2' ],
            ])
            colors.extend ([
                ('C1', 1), ('C1', 0.5), 
                ('C2', 1), ('C2', 0.5), 
            ])
            linestyles.extend ([
                '-', '-',
                '-', '-',
            ])
            offsets.extend([ 84, 65, 143, 188 ])
            termistors.extend([ True, True, True, True ])
            xlims = [ 52, 850 ]

        skew_responses      = []

        legends = {
                'NoTermReplica1_Gradini2':  'Without termistor - Replica 1',
                'NoTermReplica2_Gradini2':  'Without termistor - Replica 2',
                'NoTermReplica3_Gradini2':  'Without termistor - Replica 3',

                'Replica1_Ramp2':          'With termistor - Replica 1',
                'Replica2_Ramp2':          'With termistor - Replica 2',
                'Replica1_F150_Ramp2':        'With termistor - Replica 1, size 1.5x',
                'Replica2_F150_Ramp2':        'With termistor - Replica 2, size 1.5x',
                'Serie2_Replica1F200_Ramp2':  'With termistor - Replica 1, size 2x',
                'Serie2_Replica2F200_Ramp2':  'With termistor - Replica 2, size 2x',
        }
        
elif analisi == 3:
    ''' ANALISI 3: REPETITIONS COMPARISON '''
    savefig_name = 'IMGS_paper/bell_shape_trials_all.pdf'
    groups = [
            [ 'Replica1_Ramp2', 'Replica1_Ramp3' ], 
            [ 'Replica2_Ramp2', 'Replica2_Ramp3' ],

            [ 'Replica1_F150_Ramp2', 'Replica1_F150_Ramp3' ],
            [ 'Replica2_F150_Ramp2', 'Replica2_F150_Ramp3' ],

            [ 'Serie2_Replica1F200_Ramp2', 'Serie2_Replica1F200_Ramp3' ],
            [ 'Serie2_Replica2F200_Ramp2', 'Serie2_Replica2F200_Ramp3' ],
        ]
    
    colors = [ ('C0', 'scaled'), ('C0', 'scaled'),    ('C1', 'scaled'), ('C1', 'scaled'),    ('C2', 'scaled'), ('C2', 'scaled') ]
    linestyles = [ '-', '--',           '-', '--',               '-', '--',           ]

    offsets = [ -13,0, 90,71, 149,195  ]

    all_values = []
    all_deltas = []

    patch_values = []
    patch_deltas = []

    nude_values = []
    nude_deltas = []

    remove_baseline = False
    print('remove_baseline forced to False')

elif analisi == 4:
    ''' ANALISI : REPETITIONS COMPARISON '''
    savefig_name = 'IMGS_paper/bell_shape_trials_all_analisi4.pdf'
    groups = [
            [ 'Replica1_Ramp2', 'Replica2_Ramp2' ], 
            [ 'Replica1_Ramp3', 'Replica2_Ramp3' ],

            [ 'Replica1_F150_Ramp2', 'Replica2_F150_Ramp2' ],
            [ 'Replica1_F150_Ramp3', 'Replica2_F150_Ramp3' ],

            [ 'Serie2_Replica1F200_Ramp2', 'Serie2_Replica2F200_Ramp2' ],
            [ 'Serie2_Replica1F200_Ramp3', 'Serie2_Replica2F200_Ramp3' ],
        ]
    
    colors = [ ('C0', 'scaled'), ('C0', 'scaled'),    ('C1', 'scaled'), ('C1', 'scaled'),    ('C2', 'scaled'), ('C2', 'scaled') ]
    linestyles = [ '-', '--',           '-', '--',               '-', '--',           ]

    offsets = {
        'Replica1_Ramp2': -13,
        'Replica1_Ramp3': -13,
        'Replica2_Ramp2': 0,
        'Replica2_Ramp3': 0,
        'Replica1_F150_Ramp2': 90,
        'Replica1_F150_Ramp3': 90,
        'Replica2_F150_Ramp2': 71,
        'Replica2_F150_Ramp3': 71,
        'Serie2_Replica1F200_Ramp2': 149,
        'Serie2_Replica1F200_Ramp3': 149,
        'Serie2_Replica2F200_Ramp2': 195,
        'Serie2_Replica2F200_Ramp3': 195,
    }

    all_values = []
    all_deltas = []

    patch_values = []
    patch_deltas = []

    nude_values = []
    nude_deltas = []

    remove_baseline = False
    print('remove_baseline forced to False')

elif analisi == '3a':
    savefig_name = 'IMGS_paper/bell_shape_trials.pdf'
    groups = [
            [ 'Replica1_Ramp2', 'Replica1_Ramp3' ], 
            [ 'Replica2_Ramp2', 'Replica2_Ramp3' ],
    ]

    colors = [ ('C0', 'scaled'), ('C0', 'scaled'),  ]
    linestyles = [ '-', '--',         ]
    offsets = [ -13,0, ]
    offset_x = -16

    xlims = [ 85+offset_x, 230+offset_x ]

    scatter_x = np.array([ 116,  142.4,   168.8,  198 ])
    scatter_y = np.array([ 15.64, 18.27,  17.52,  9.9 ])
    colors_scatter = [ 'C1', 'C2', 'C3', 'C4' ]
    group_filename_scatter = [ 1,0 ]


baseline_channels    = {

        'Replica1_Ramp2':   [ 40+13,  80+13 ],
        'Replica1_Ramp3':   [ 40+13,  80+13 ],
        'Replica2_Ramp2':   [ 40,     80    ],
        'Replica2_Ramp3':   [ 40,     80    ],

        'Replica1_F150_Ramp2':    [ 118-90, 178-90 ],
        'Replica1_F150_Ramp3':    [ 118-90, 178-90 ],
        'Replica2_F150_Ramp2':  [ 118-71, 178-71 ],
        'Replica2_F150_Ramp3':  [ 118-71, 178-71 ],

        'Serie2_Replica1F200_Ramp2':  [ 236-149, 316-149 ],
        'Serie2_Replica1F200_Ramp3':  [ 236-149, 316-149 ],
        'Serie2_Replica2F200_Ramp2':  [ 236-223, 316-223 ],
        'Serie2_Replica2F200_Ramp3':  [ 236-223, 316-223 ],
        'Serie2_Replica2F200_Ramp2':  [ 236-195, 316-195 ],
        'Serie2_Replica2F200_Ramp3':  [ 236-195, 316-195 ],

        'NoTermReplica1_Gradini2':  [ 34+0,    74    ],
        'NoTermReplica2_Gradini2':  [ 34+2+0,  74+2  ],
        'NoTermReplica3_Gradini2':  [ 34+12+0, 74+12 ],

}

fig = plt.figure(figsize=(6.4*size_figure_factor, 4.8*size_figure_factor))


for i_group, group in enumerate(groups):

    if colors == None:
        color = cmap((i_group+0.5) / (len(groups)))
    else:
        color, alpha = colors[i_group]
    if linestyles == None:
        linestyle = '-'
    else:
        linestyle = linestyles[i_group]

    peak_temp = []
    skew_temp = []

    for i_filename, filename in enumerate(group):
        data = np.load(f'data/response_{filename}.npz')

        marker = 'x' if 'var' in filename else '.'

        channels_orig = data['channels']
        try: 
            channels = data['channels'] + offsets[filename]
        except TypeError:
            channels = data['channels'] + offsets[i_group]
        ms = data['ms']
        dms = data['dms']
        p_vals = data['p_vals']


        baseline_mask = ( channels_orig < baseline_channels[filename][0] ) + ( channels_orig > baseline_channels[filename][1] ) 
        baseline_val = np.mean(ms[baseline_mask])
        baseline_std = np.std(ms[baseline_mask])

        # compute skewness
        patch_mask = (channels_orig >= baseline_channels[filename][0]) * (channels_orig <= baseline_channels[filename][1])
        prob = np.where(ms[patch_mask]>0, ms[patch_mask], 0)
        prob = abs(prob) / np.sum(abs(prob))
        mean = np.sum(prob * channels_orig[patch_mask])
        skew = np.sum(prob * ((channels_orig[patch_mask] - mean)**3)) / (np.sum(prob * ((channels_orig[patch_mask] - mean)**2))**(3/2))
        skew_temp.append(skew)
        # plt.axvline( (mean + offsets[i_group])*x_multiplicator+offset_x, color=color, alpha=alpha, linestyle='--')

        if analisi==3 or analisi==4:
            if i_filename==0:
                temp_r1 = ms
                temp_mask = patch_mask
            elif i_filename==1:
                temp_r2 = ms
                if analisi == 3:
                    assert np.array_equal(channels, channels) and np.array_equal(patch_mask, temp_mask), 'For Analisi 3, the two files in each group must have the same channels and patch mask'
                else:
                    first_0 = np.where(temp_mask)[0][0]
                    last_0 = np.where(temp_mask)[0][-1]
                    first_1 = np.where(patch_mask)[0][0]
                    last_1 = np.where(patch_mask)[0][-1]
                    assert last_0-first_0 == last_1-first_1, 'For Analisi 4, the two files in each group must have the same number of channels in the patch mask'

                    temp_r1 = temp_r1[first_0:last_0]
                    temp_r2 = temp_r2[first_1:last_1]
                    patch_mask = np.array([ True ] * (last_0-first_0) )
                    print(last_0, first_0, last_1, first_1, last_1-first_1)

            else:
                raise ValueError('Analisi 3/4 should have exactly 2 files per group')

        if analisi == '3a' and i_group == group_filename_scatter[0] and i_filename == group_filename_scatter[1]:
            print(f'Adding scatter points for group {i_group}, filename {filename}', baseline_val)
            scatter_y = scatter_y - baseline_val if remove_baseline else scatter_y
            plt.scatter(scatter_x, scatter_y, color=colors_scatter, s=30)
            plt.scatter(scatter_x, scatter_y, s=80, facecolors='none', edgecolors=colors_scatter)

        if remove_baseline:
            ms = ms - baseline_val
            dms = np.sqrt(dms**2 + baseline_std**2)
            baseline_val = 0
        peak_temp.append(np.max(ms))

        if colors == None or alpha == 'scaled':
            alpha_ = alpha
            alpha = max(1-0.45*i_filename, 0.01)
            print(f'Filename {filename} ({i_filename}): alpha={alpha:.2f}')
        print(alpha)
        plt.errorbar( channels*x_multiplicator+offset_x, ms, yerr=dms, linestyle=linestyle, label=f'{filename}' if filename not in legends else legends[filename] ,color=color, alpha=alpha, elinewidth=0.5)
        if alpha_ is not None:
            alpha = alpha_

    if analisi == 1:
        peak_responses.append(np.mean(peak_temp))
        peak_responses_std.append(np.std(peak_temp))
    elif analisi == 2:
        skew_responses.append(np.mean(skew_temp))
        print(f'Group {i_group}: Skewness = {skew_responses[-1]:.4f}')
    elif analisi == 3 or analisi == 4:
        # compute delta between the two replicas
        delta = np.abs(temp_r2 - temp_r1)
        mean = (temp_r1 + temp_r2) / 2
        all_values.extend(list(mean))
        all_deltas.extend(list(delta))
        patch_values.extend(list(mean[patch_mask]))
        patch_deltas.extend(list(delta[patch_mask]))
        nude_values.extend(list(mean[~patch_mask]))
        nude_deltas.extend(list(delta[~patch_mask]))
        if analisi == 3:
            plt.plot(channels*x_multiplicator+offset_x, delta, linestyle=linestyle, color=color, alpha=1)
        else:
            plt.plot(np.arange(len(patch_mask))*x_multiplicator, delta[patch_mask], linestyle=linestyle, color=color, alpha=1)

if x_multiplicator == 1:
    plt.xlabel('Gate')
else:
    plt.xlabel('Position [mm]')
if remove_baseline:
    plt.ylabel('Delta Slope ['+r'$\mu$'+'s/K]')
else:
    plt.ylabel('Slope ['+r'$\mu$'+'s/K]')
plt.legend(fontsize=8)
if xlims is not None:
    plt.xlim(xlims)
if ylims is not None:
    plt.ylim(ylims)


# plt.savefig(savefig_name, bbox_inches='tight')
plt.show()


''' analisi 1 '''
if analisi == 1:
    def linear_func(x, a, b):
        return a*x + b
    
    popt, pcov = curve_fit(linear_func, peak_factors, peak_responses, sigma=peak_responses_std)
    perr = np.sqrt(np.diag(pcov))
    res = peak_responses - linear_func(np.array(peak_factors), *popt)
    chi2_ = np.sum((res / np.array(peak_responses_std))**2)
    p_val = 1 - chi2.cdf(chi2_, df=len(peak_factors) - len(popt))
    print(f'Fitted parameters: a={popt[0]:.4f} pm {perr[0]:.4f}, b={popt[1]:.4f} pm {perr[1]:.4f}, chi2={chi2_:.2f}, p-value={p_val:.4f}')

    x_fit = np.linspace(min(peak_factors), max(peak_factors), 100)
    y_fit = linear_func(x_fit, *popt)

    fig = plt.figure(figsize=(6.4*0.75, 4.8*0.75))
    for i in range(len(peak_factors)):
        plt.errorbar(peak_factors[i], peak_responses[i], yerr=peak_responses_std[i], fmt='.', color='C'+str(i), label=f'Peak Response (size {peak_factors[i]}x)')

    plt.plot(x_fit, y_fit, '--', label=f'Best Fit (linear), p={p_val:.2f}', color='black')
    plt.xlabel('Size Factor')
    plt.ylabel(r'Peak Response [$\mu$s/K]')
    plt.xticks(peak_factors)
    plt.legend(loc='lower right', fontsize=8)
    # plt.savefig('IMGS_paper/size_comparison_fit.pdf', bbox_inches='tight')
    plt.show()


if analisi == 2:

    with_term = [ skew for skew, term in zip(skew_responses, termistors) if term ]
    without_term = [ skew for skew, term in zip(skew_responses, termistors) if not term ]

    pval = mannwhitneyu(without_term, with_term, alternative='greater').pvalue
    print(f'Mann-Whitney U test p-value: {pval:.4f} (without termistors > with termistors)')

    print(f'Average skewness with termistors: {np.mean(with_term):.4f} ± {np.std(with_term):.4f}')
    print(f'Average skewness without termistors: {np.mean(without_term):.4f} ± {np.std(without_term):.4f}')

    fig = plt.figure(figsize=([6.4, 0.6]))
    for i_group in range(len(groups)):
        plt.scatter(skew_responses[i_group], 0, c=colors[i_group][0], alpha=colors[i_group][1])
    plt.xlabel('Skewness of Response Distribution')
    plt.axhline(0, color='black', alpha=0.5, linestyle='-', linewidth=0.5)
    plt.yticks([])
    plt.ylim(-0.5, 2)
    # plt.savefig('IMGS_paper/skewness_comparison.pdf', bbox_inches='tight')
    plt.show()

if analisi == 3 or analisi == 4:
    bins = np.linspace( min(all_values), max(all_values), 50 )
    plt.hist(all_values, bins=bins,   density=True, alpha=0.5, label='All channels')
    plt.hist(patch_values, bins=bins, density=True, alpha=0.5, label='Patch channels')
    plt.hist(nude_values, bins=bins,  density=True, alpha=0.5, label='Non-patch channels')
    plt.legend()
    plt.show()

    bins = np.linspace( min(all_deltas), max(all_deltas), 50 )
    plt.hist(all_deltas, bins=bins,   density=True, alpha=0.5, label='All channels')
    plt.hist(patch_deltas, bins=bins, density=True, alpha=0.5, label='Patch channels')
    plt.hist(nude_deltas, bins=bins,  density=True, alpha=0.5, label='Non-patch channels')
    plt.legend()
    plt.show()

    all_deltas_rel = np.array(all_deltas) / np.array(all_values)
    patch_deltas_rel = np.array(patch_deltas) / np.array(patch_values)
    nude_deltas_rel = np.array(nude_deltas) / np.array(nude_values)
    bins = np.linspace( min(all_deltas_rel), max(all_deltas_rel), 50 )

    fig = plt.figure(figsize=(6.4, 2.))

    plt.hist(patch_deltas_rel, bins=bins, density=True, alpha=0.5)
    print(f'Average relative delta for patch channels: {np.mean(patch_deltas_rel)*100:.4f} %' )
    if analisi == 3:
        plt.xlabel('Relative Delta (|Trial 2 - Trial 1| / Mean)')
    else:
        plt.xlabel('Relative Delta (|Replica 2 - Replica 1| / Mean)')
    plt.ylabel('Density')
    mean = np.mean(patch_deltas_rel)
    plt.axvline(mean, linewidth=0.75)
    plt.text(mean+0.001, plt.ylim()[1]*0.9, f'Mean = {mean:.3f}', color='black', fontsize=10, ha='left')
    # plt.savefig(f'IMGS_paper/trials_delta_relative_hist_{analisi}.pdf', bbox_inches='tight')
    plt.show()