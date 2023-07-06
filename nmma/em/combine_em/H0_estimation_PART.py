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
from combine_em_utils import logit, calc_H0, KDE_multiply
from bilby.gw.conversion import luminosity_distance_to_redshift

import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'.')

H_true = Planck18.H(0)
c = const.c.to('km/s').value

downsample_posterior = False
# number to downsample to, if True
N_draws = 1000

D_range = [0, 500]

#np.random.seed(42)

with h5py.File('GW170817_PE.hdf5', 'r') as data:
    GW170817_samples = data['posterior_samples'][()]

# cols in inj file: idx, longitude, latitude, inclination, distance, mass1, mass2, mass1_detector, mass2_detector, lambda1, lambda2, spin1z, spin2z
inj_filename = 'O4_BNS_injections_lambdas.dat'
injections = np.loadtxt(f'posteriors/{inj_filename}')

def combine_EM_events(original_KDE, EM_event, D_inj, H_range, D_range, transform_type=logit, downsample=True, N_draws=N_draws, num=None):
    '''
    '''
    D = EM_event['luminosity_distance']
   
    plt.hist(D, bins=20)
    plt.axvline(D_inj, color='k')
    plt.ylabel('Counts')
    plt.xlabel('Distance')
    plt.savefig(f'distance_posteriors/d_event_{num}.png')
    plt.close()


    # use injection distance
    z = np.array(luminosity_distance_to_redshift(D_inj, cosmology=Planck18))
    if (not downsample_posterior) or (len(D) < N_draws):
        N_draws = len(D)
        N_draws = 100000
    D = np.random.choice(D, size=N_draws).flatten()
    H = calc_H0(D, z)

    np.savetxt(f'event_files/event_{num}_H_posterior.txt', H)

    combined_posterior = eng.PART_function(original_KDE, H)   
    combined_posterior = np.asarray(combined_posterior).flatten()

    H_kde = scipy.stats.gaussian_kde(combined_posterior, weights=1/combined_posterior**4)
    H_resam = H_kde.resample(N_draws).flatten()

    bins = np.linspace(0,120,50)
    plt.figure(figsize=(12,9))
    plt.hist(H, alpha=.8, bins = bins, density = 1, edgecolor = 'black', label='New Event')
    plt.hist(original_KDE, alpha=.6, bins = bins, density = 1, edgecolor = 'black', label='Original Event(s)')
    plt.hist(H_resam, alpha=.4, bins = bins, density = 1, edgecolor = 'black', label='Combined Events')
    plt.ylabel('Probability')
    plt.xlabel('H_0')
    plt.axvline(x=H_true.value, color='black', label='Planck18 H_0') # H_true
    plt.legend()
    plt.savefig(f'./output/event_hists/PART/H_individual_hist_{num}.png')
    plt.close()

    return H_resam


def run_event_combination(H_range, D_range, N_draws=N_draws):
    '''
    '''
    EM_event_files = []
    event_indices = []

    # GW170817
    original_D_inj = 40
    D = GW170817_samples['luminosity_distance']

    z = np.array(luminosity_distance_to_redshift(original_D_inj, cosmology=Planck18))
    H_original = calc_H0(D, z)

    H_medians = []
    H_stds = []
    H_lower_cred, H_upper_cred = [], []
    H_medians.append(np.median(H_original))
    H_stds.append(np.std(H_original))
    H_lower_cred.append(np.quantile(H_original, .05))
    H_upper_cred.append(np.quantile(H_original, .95))

    # cols in inj file: idx, longitude, latitude, inclination, distance, mass1, mass2, mass1_detector, mass2_detector, lambda1, lambda2, spin1z, spin2z 
    D_injs = injections[:,4]

    N_events = 30
    #N_events = 3
    percentiles = []
    event_idxs = np.arange(0, N_events)
    #event_idxs = [0, 1, 3, 4, 5, 6, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29]
    #event_idxs = [0, 1, 3, 4, 5, 6, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 24, 26, 27]
    event_idxs = [0, 1, 3, 4, 5, 6, 11, 12, 13, 17, 18, 19, 20, 21, 24, 26, 27]
    #event_idxs = [0, 1, 3, 4, 5, 6, 11, 12, 13, 15, 17, 18, 19, 20, 21, 24, 26, 27]
    event_idxs = np.random.choice(event_idxs, len(event_idxs), replace=False)
    for n in event_idxs:
        path_to_events = f'/home/nina.kunert/02_H0/03_GWEM/01_O4/outdir_{n}'  # GW
        #path_to_events = f'/home/nina.kunert/02_H0/02_EM_H0/01_ZTF/01_O4/00_allresults/01_dummyposterior/outdir_{n}'  # EM
        event_folders = os.listdir(path_to_events)
        prefix = f'run{n}_simulation'  # GW
        #prefix = 'O4_posterior_samples'  # EM
        #filename = 'O4_result.json'
        filename = [f for f in event_folders if f.startswith(prefix)][0]
        with open(f'{path_to_events}/{filename}', 'r') as f:  # GW
            event_file = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)
        event = event_file['posterior']
        #event = pd.read_csv(f'{path_to_events}/{filename}', sep="\s+", skiprows=1, names=['KNtimeshift', 'luminosity_distance', 'KNphi', 'inclination_EM', 'log10_mej_dyn', 'log10_mej_wind Ebv', 'log_likelihood', 'log_prior'])  # EM

        print(f'Combining events')
        percentiles.append(percentileofscore(event['luminosity_distance'], D_injs[n]))
        H_vals = combine_EM_events(H_original, event, D_injs[n], H_range = H_range, D_range = D_range, num=n)
        H_original = H_vals
        H_medians.append(np.median(H_vals))
        H_stds.append(np.std(H_vals))
        H_lower_cred.append(np.quantile(H_vals, .025))
        H_upper_cred.append(np.quantile(H_vals, .975))
    print(f'Events combined!')

    H_lower_cred, H_upper_cred = np.array(H_lower_cred), np.array(H_upper_cred)

    print(percentiles)
    percentiles.sort()
    cdf = np.cumsum(np.ones(len(percentiles)))
    cdf = cdf/np.max(cdf)

    plt.hist(percentiles, bins=10)
    plt.ylabel('Counts')
    plt.xlabel('Percentile')
    plt.savefig('distance_posteriors/percentile_hist.png')
    plt.close()


    H_cred = np.vstack([H_lower_cred, H_upper_cred])

    return H_vals, np.array(H_medians), np.array(H_stds), H_cred


H_combined, H_medians, H_stds, H_cred = run_event_combination(H_range = [0,200], D_range = D_range)

plt.figure(figsize=(12,9))
plt.hist(H_combined, alpha=.8, bins = 30, edgecolor = 'black', density=True, label = 'Combined Events')
plt.ylabel('Probability')
plt.xlabel(r'H$_{0}$ [km/s/Mpc]')
plt.axvline(x=H_true.value, color='red', linestyle='--', linewidth=4, label=r'Planck18 H$_{0}$') # H_true
plt.legend()
plt.savefig('./output/all_events/H0_PART_hist_usf.png')
plt.close()

f, (ax0, ax1) = plt.subplots(2, 1, figsize=(12,9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(np.arange(1,len(H_medians)+1), (H_cred[1][:] - H_cred[0][:])/H_true.value, linewidth=3, marker='o')
#ax0.errorbar(np.arange(1,len(H_medians)+1), H_medians, yerr=2*H_stds, linewidth=3, capsize=6, marker='o', ls='none', label='Combined Event Prediction')
ax0.errorbar(np.arange(1,len(H_medians)+1), H_medians, yerr=np.abs(H_cred-H_medians), linewidth=3, capsize=6, marker='o', ls='none', label='Combined Event Prediction')
ax0.axhline(y=H_true.value, color='red', linestyle='--', linewidth=2, label=r'Planck18 H$_{0}$')
ax0.set_ylabel(r'H$_{0}$ [km/s/Mpc]')
ax1.set_ylabel(r'Fractional H$_{0}$ Error')
ax1.set_yscale('log')
ax1.set_ylim([1e-2, .3])
plt.xlabel(r'Number of Events')
ax0.legend()
plt.savefig('./output/all_events/H0_error_PART.png')
plt.close()

np.savetxt('H_medians.txt', H_medians)
np.savetxt('H_stds.txt', H_stds)
np.savetxt('H_cred.txt', H_cred)
