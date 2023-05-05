import os
import numpy as np
from astropy.cosmology import Planck18
from astropy import units as u
from astropy.coordinates import Distance
from astropy import constants as const
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import random
import h5py
from scipy.stats import percentileofscore
#import joypy
import json
import bilby
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit, calc_H0
from bilby.gw.conversion import luminosity_distance_to_redshift

# v = H0*D
#c = 3e5 # km/s
# z ~ H0 * D / c
# H0 = z * c /D
H_true = Planck18.H(0)
c = const.c.to('km/s').value

downsample_posterior = True
#N_draws = 100
N_draws = 50000

D_range = [0, 500]


with h5py.File('GW170817_PE.hdf5', 'r') as data:
    GW170817_samples = data['posterior_samples'][()]

# cols in inj file: idx, longitude, latitude, inclination, distance, mass1, mass2, mass1_detector, mass2_detector, lambda1, lambda2, spin1z, spin2z
inj_filename = 'O4_BNS_injections_lambdas.dat'
injections = np.loadtxt(f'posteriors/{inj_filename}')

def combine_EM_events(original_KDE, EM_event, D_inj, H_range, D_range, transform_type=logit, downsample=True, N_draws=N_draws, num=None):
    '''
    '''
    D = EM_event['luminosity_distance']
    # use injection distance
    z = np.array(luminosity_distance_to_redshift(D_inj, cosmology=Planck18))
    if not downsample_posterior:
        N_draws = len(D)
    D = np.random.choice(D, size=N_draws)
    H = calc_H0(D, z)

    plt.figure(figsize=(12,9))
    plt.hist(H, alpha=.8, bins = 30, edgecolor = 'black')
    plt.ylabel('Probability')
    plt.xlabel('H_0')
    plt.axvline(x=H_true.value, color='black', label='Planck18 H_0') # H_true
    plt.legend()
    plt.savefig(f'./output/event_hists/H0/H_individual_hist_farah_H0_{num}.png')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.hist(D, alpha=.8, bins = 30, edgecolor = 'black')
    plt.ylabel('Probability')
    plt.xlabel('Distance')
    plt.axvline(x=D_inj, color='black', label='Injected Distance')
    plt.legend()
    plt.savefig(f'./output/event_hists/distance/D_individual_hist_farah_H0_{num}.png')
    plt.close() 

    D_prior = bilby.gw.prior.UniformSourceFrame(minimum=D_range[0], maximum=D_range[1], name='luminosity_distance',latex_label='$D_L$')
    D_prior = D_prior.prob(D)

    H_transform, H_prior = logit(H, H_range, include_prior=True)
    finite_idx = np.isfinite(H_transform)
    H_transform, H_prior = H_transform[finite_idx], H_prior[finite_idx]

    H_kde_transform = scipy.stats.gaussian_kde(H_transform, weights=1/H_prior)
    D_kde = scipy.stats.gaussian_kde(D, weights=1/D_prior)
    H_kde_original = original_KDE['H0']
    D_kde_original = original_KDE['luminosity_distance']

    H_joint_kde = KDE_multiply(H_kde_transform, H_kde_original, downsample=downsample)
    D_joint_kde = KDE_multiply(D_kde, D_kde_original, downsample=downsample)

    return D_joint_kde, H_joint_kde


def run_event_combination(H_range, D_range, N_draws=N_draws):
    '''
    '''
    EM_event_files = []
    event_indices = []

    path_to_events = 'posteriors/O4'
    event_folders = os.listdir(path_to_events)

    # GW170817
    original_D_inj = 40
    D = GW170817_samples['luminosity_distance']

    z = np.array(luminosity_distance_to_redshift(original_D_inj, cosmology=Planck18))
    H = calc_H0(D, z)

    D_prior = bilby.gw.prior.UniformSourceFrame(minimum=D_range[0], maximum=D_range[1], name='luminosity_distance', cosmology = Planck18, latex_label='$D_L$') 
    D_prior = D_prior.prob(D)

    H_transform, H_prior = logit(H, H_range, include_prior=True)

    H_kde_transform = scipy.stats.gaussian_kde(H_transform, weights=1/H_prior)
    D_kde = scipy.stats.gaussian_kde(D, weights=1/D_prior)

    original_KDE = {}
    original_KDE['luminosity_distance'] = D_kde
    original_KDE['H0'] = H_kde_transform

    H_medians = []
    H_stds = []
    H_medians.append(np.median(H))
    H_stds.append(np.std(H))

    # cols in inj file: idx, longitude, latitude, inclination, distance, mass1, mass2, mass1_detector, mass2_detector, lambda1, lambda2, spin1z, spin2z 
    D_injs = injections[:,4]

    plt.figure(figsize=(12,9))
    plt.hist(D_injs, alpha=.8, bins = 10, edgecolor = 'black')
    plt.ylabel('Probability')
    plt.xlabel('Distance')
    plt.legend()
    plt.savefig(f'./output/event_hists/distance/D_inj_hist_farah.png')
    plt.close()

    N_events = 29
    N_evetns = 3
    percentiles = []
    event_idxs = random.sample(range(0,N_events), N_events)
    for n in event_idxs:
        prefix = f'run{n}_simulation'
        filename = [f for f in event_folders if f.startswith(prefix)][0]
        print(filename)
        with open(f'posteriors/O4/{filename}', 'r') as f:
            event_file = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)
        event = event_file['posterior']
        print(f'Combining events')
        percentiles.append(percentileofscore(event['luminosity_distance'], D_injs[n]))
        D_kde_result, H_kde = combine_EM_events(original_KDE, event, D_injs[n], H_range = H_range, D_range = D_range, num=n)
        original_KDE = {'luminosity_distance': D_kde_result, 'H0': H_kde}
        H_vals = logit(H_kde.resample(N_draws), H_range, inv_transform=True)
        p_value = scipy.stats.percentileofscore(event['luminosity_distance'], D_injs[n]) / 100
        p_value = min(p_value, 1-p_value)
        if p_value > 0: # no p value cut for now
            H_medians.append(np.median(H_vals))
            H_stds.append(np.std(H_vals))
    print(f'Events combined!')

    H_transform_resam = H_kde.resample(N_draws)
    D_transform_resam = D_kde_result.resample(N_draws)

    H_resam = logit(H_transform_resam, H_range, inv_transform=True)
    D_resam = D_transform_resam

    percentiles.sort()
    cdf = np.cumsum(np.ones(len(percentiles)))
    cdf = cdf/np.max(cdf)

    plt.figure(figsize=(12,9))
    plt.plot(np.array(percentiles)/100, cdf, linewidth=5)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xlabel('Percentile')
    plt.ylabel(f'Fraction of Distance values')
    plt.plot([0,1], [0, 1], color='black')
    plt.savefig(f'output/PP_plots/PP_plot_D.png')
    plt.close()
   

    plt.figure(figsize=(12,9))
    plt.hist(np.array(percentiles), alpha=.8, bins = 30, edgecolor = 'black')
    plt.ylabel('Probability')
    plt.xlabel('Percentile')
    plt.savefig('./output/PP_plots/D_per_hist.png')
    plt.close() 

    return D_resam.flatten(), H_resam.flatten(), np.array(H_medians), np.array(H_stds)



D_combined, H_combined, H_medians, H_stds = run_event_combination(H_range = [0,200], D_range = D_range)

plt.figure(figsize=(12,9))
plt.hist(H_combined, alpha=.8, bins = 30, edgecolor = 'black', density=True, label = 'Combined Events')
plt.ylabel('Probability')
plt.xlabel(r'H$_{0}$ [km/s/Mpc]')
plt.axvline(x=H_true.value, color='red', linestyle='--', linewidth=4, label=r'Planck18 H$_{0}$') # H_true
plt.legend()
plt.savefig('./output/all_events/H0_farah_hist_usf.png')
plt.close()

print(H_stds, H_medians)

f, (ax0, ax1) = plt.subplots(2, 1, figsize=(12,9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(np.arange(1,len(H_medians)+1), H_stds/H_true.value, linewidth=3, marker='o')
ax0.errorbar(np.arange(1,len(H_medians)+1), H_medians, yerr=2*H_stds, linewidth=3, capsize=6, marker='o', ls='none', label='Combined Event Prediction')
ax0.axhline(y=H_true.value, color='red', linestyle='--', linewidth=2, label=r'Planck18 H$_{0}$')
ax0.set_ylabel(r'H$_{0}$ [km/s/Mpc]')
ax1.set_ylabel(r'Fractional H$_{0}$ Error')
ax1.set_yscale('log')
ax1.set_ylim([1e-2, .3])
plt.xlabel(r'Number of Events')
ax0.legend()
plt.savefig('./output/all_events/H0_error_farah.png')
plt.close()

np.savetxt('H_medians.txt', H_medians)
np.savetxt('H_stds.txt', H_stds)
