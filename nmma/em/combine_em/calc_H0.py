import os
import numpy as np
from astropy.cosmology import Planck18
from astropy import units as u
from astropy.coordinates import Distance
from astropy import constants as const
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import joypy
import json
import bilby
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit
from bilby.gw.conversion import luminosity_distance_to_redshift

# v = H0*D
c = 3e5 # km/s
# z ~ H0 * D / c
# H0 = z * c /D
H_true = Planck18.H(0)
c = const.c.to('km/s').value

def combine_EM_events(original_KDE, EM_event, original_D_inj, D_inj, H_range, D_range, i_range, transform_type=logit, downsample=True):
    '''
    '''
    D = EM_event['luminosity_distance']
    i = EM_event['cos_inclination_EM']
    # use injection distance
    z = np.array(luminosity_distance_to_redshift(D_inj, cosmology=Planck18))
    H = z * c / D

    print('Transforming...')
    H_transform, H_prior = logit(H, H_range, include_prior=True)
    D_transform, D_prior = logit(D, D_range, include_prior=True)
    i_transform, i_prior = logit(i, i_range, include_prior=True) 

    print('Taking KDE...')
    H_kde_transform = scipy.stats.gaussian_kde(H_transform, weights=1/H_prior)
    D_kde_transform = scipy.stats.gaussian_kde(D_transform, weights=1/D_prior)
    i_kde_transform = scipy.stats.gaussian_kde(i_transform, weights=1/i_prior)
    H_kde_original = original_KDE['H0']
    D_kde_original = original_KDE['luminosity_distance']
    i_kde_original = original_KDE['cos_inclination_EM']


    print('Taking joint KDE')
    H_joint_kde = KDE_multiply(H_kde_transform, H_kde_original, downsample=downsample)
    D_joint_kde = KDE_multiply(D_kde_transform, D_kde_original, downsample=downsample)
    i_joint_kde = KDE_multiply(i_kde_transform, i_kde_original, downsample=downsample)

    print('Done!')
    return D_joint_kde, i_joint_kde, H_joint_kde

def run_event_combination(path_to_events, H_range, D_range, i_range, save_hist=False):
    '''
    '''
    # note, right now the list dir MUST read files in the same order as the injections are in the injection file
    event_folders = os.listdir(path_to_events)
    EM_event_files = []
    for folder in event_folders:
        print(folder)
        samples = Table.read(str(path_to_events)+'/'+str(folder)+'/injection_posterior_samples.dat', format="csv", delimiter=" ")
        samples['cos_inclination_EM'] = np.cos(samples['inclination_EM'])
        EM_event_files.append(samples)

    with open('../../../injection.json', 'r') as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)
    injection_df = injection_dict["injections"]
    original_injection_parameters = injection_df.iloc[0].to_dict()
    original_D_inj = original_injection_parameters['luminosity_distance']

    original_samples = EM_event_files[0]
    D = original_samples['luminosity_distance']
    i = original_samples['cos_inclination_EM']

    z = np.array(luminosity_distance_to_redshift(original_D_inj))
    H = z * c / D

    H_transform, H_prior = logit(H, H_range, include_prior=True)
    D_transform, D_prior = logit(D, D_range, include_prior=True)
    i_transform, i_prior = logit(i, i_range, include_prior=True)

    H_kde_transform = scipy.stats.gaussian_kde(H_transform, weights=1/H_prior)
    D_kde_transform = scipy.stats.gaussian_kde(D_transform, weights=1/D_prior)
    i_kde_transform = scipy.stats.gaussian_kde(i_transform, weights=1/i_prior)

    original_KDE = {}
    original_KDE['luminosity_distance'] = D_kde_transform
    original_KDE['cos_inclination_EM'] = i_kde_transform
    original_KDE['H0'] = H_kde_transform

    for n, event in enumerate(EM_event_files[1:]):
        print('Combining events '+str(n+1)+' and '+str(n+2))
        injection_parameters = injection_df.iloc[n+1].to_dict()
        D_inj = injection_parameters['luminosity_distance']
        D_kde, i_kde, H_kde = combine_EM_events(original_KDE, event, original_D_inj, D_inj, H_range = H_range, D_range = D_range, i_range = i_range)
        original_KDE = {'luminosity_distance': D_kde, 'cos_inclination_EM': i_kde, 'H0': H_kde}
        print('Events combined!')

    N_hist = len(D)
    H_transform_resam = H_kde.resample(N_hist)
    D_transform_resam = D_kde.resample(N_hist)
    i_transform_resam = i_kde.resample(N_hist)

    H_resam = logit(H_transform_resam, H_range, inv_transform=True)
    D_resam = logit(D_transform_resam, D_range, inv_transform=True)
    i_resam = logit(i_transform_resam, i_range, inv_transform=True)

    return D_resam.flatten(), i_resam.flatten(), H_resam.flatten()

path_to_events = '../../../combine_events_outdir'
D_combined, i_combined, H_combined = run_event_combination(path_to_events, H_range = [0,150], D_range = [0,1000], i_range = [0,1], save_hist=True)

plt.hist(H_combined, alpha=.8, bins = 30, histtype='stepfilled', density=True, label = 'Combined Events')
plt.ylabel('Probability')
plt.xlabel('H_0')
plt.axvline(x=H_true.value, color='black', label='Planck18 H_0') # H_true
plt.legend()
plt.savefig('./output/H_combined_hist.png')
plt.close()
