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
# H0 ~ 70 # km/s/Mpc
c = 3e5 # km/s
# z ~ H0 * D / c
# H0 = z * c /D
H_true = Planck18.H(0)
c = const.c.to('km/s').value

def combine_EM_events(original_KDE, EM_event, D_original_inj_D, D_inj, H_range = [0,150], D_range = [0,1000], i_range = [0,1], transform_type=logit, downsample=True):
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

def run_event_combination(EM_event_files, save_hist=False, N_hist = 1000):
    '''
    '''
    original_KDE = EM_event_files[0]
    with open('../../../injection.json', 'r') as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)
    injection_df = injection_dict["injections"]
    original_injection_parameters = injection_df.iloc[0].to_dict() 
    original_injected_D = original_injection_parameters['luminosity_distance']
    for n, event in enumerate(EM_event_files[1:]):
        print('Combining events '+str(n+1)+' and '+str(n+2))
        injection_parameters = injection_df.iloc[n+1].to_dict()
        injected_D = injection_parameters['luminosity_distance']
        D_kde, i_kde, H_kde = combine_EM_events(original_KDE, event, original_injected_D, injected_D)
        original_KDE = {'luminosity_distance': D_kde, 'cos_inclination_EM': i_kde, 'H0': H_kde}
        print('Events combined!')

    H_transform_resam = H_kde.resample(N_hist)
    D_transform_resam = D_kde.resample(N_hist)
    i_transform_resam = i_kde.resample(N_hist)

    H_resam = logit(H_transform_resam, H_range, inv_transform=True)
    D_resam = logit(D_transform_resam, D_range, inv_transform=True)
    i_resam = logit(i_transform_resam, i_range, inv_transform=True)

    return D_resam.flatten(), i_resam.flatten(), H_resam.flatten()

with open('../../../injection.json', 'r') as f:
    injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)
injection_df = injection_dict["injections"]
injection_parameters = injection_df.iloc[0].to_dict()
true_D1 = injection_parameters['luminosity_distance']
injection_parameters = injection_df.iloc[1].to_dict()
true_D2 = injection_parameters['luminosity_distance']
injection_parameters = injection_df.iloc[2].to_dict()
true_D3 = injection_parameters['luminosity_distance']
injection_parameters = injection_df.iloc[3].to_dict()
true_D4 = injection_parameters['luminosity_distance']

samples1 = Table.read('../../../outdir0_custom/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples1['cos_inclination_EM'] = np.cos(samples1['inclination_EM'])

samples2 = Table.read('../../../outdir1_custom/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples2['cos_inclination_EM'] = np.cos(samples2['inclination_EM'])

samples3 = Table.read('../../../outdir2_custom/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples3['cos_inclination_EM'] = np.cos(samples3['inclination_EM'])

D = samples1['luminosity_distance']
i = samples1['cos_inclination_EM']

D2 = samples2['luminosity_distance']
i2 = samples2['cos_inclination_EM']

D3 = samples3['luminosity_distance']
i3 = samples3['cos_inclination_EM']

D_range = [0,1000] 
i_range = [0,1]
H_range = [0,150]

z = np.array(luminosity_distance_to_redshift(true_D1))
H = z * c / D
z2 = np.array(luminosity_distance_to_redshift(D2))
H2 = z2 * c / D2
z3 = np.array(luminosity_distance_to_redshift(D3))
H3 = z3 * c / D3

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

#em_list = ['', '1', '2', '3', '4', '10']
event_list = [original_KDE, samples2, samples3]
#event_list = [original_KDE, samples1, samples2]

D_combined, i_combined, H_combined = run_event_combination(event_list, N_hist=len(D), save_hist=True)
#tab = Table([D_combined, D], names=['Combined Events', 'Event1'])

plt.hist([D, D2, D3], alpha=.8, bins = 30, histtype='stepfilled', density=True, label = ['Event1', 'Event2', 'Event3', 'Event4'])
plt.ylabel('Probability')
plt.xlabel('Luminosity Distance')
plt.axvline(x=true_D1, color='blue', label='True D1')
plt.axvline(x=true_D2, color='orange', label='True D2')
plt.axvline(x=true_D3, color='green', label='True D3')
plt.legend()
plt.savefig('./output/comparison_hist_D.png')
plt.close()

#H = z * c / D
z = np.array(luminosity_distance_to_redshift(D, cosmology=Planck18))
z2 = np.array(luminosity_distance_to_redshift(D2, cosmology=Planck18))
z3 = np.array(luminosity_distance_to_redshift(D3, cosmology=Planck18))
H_test1 = z * c / D
H_test2 = z2 * c / D2
H_test3 = z3 * c / D3

plt.hist([H, H2, H3, H_combined], alpha=.8, bins = 30, histtype = 'stepfilled', density=True, label = ['Event1', 'Event2', 'Event3', 'Combined'])
plt.ylabel('Probability')
plt.xlabel('H_0')
plt.axvline(x=H_true.value, color='black', label='Planck18 H_0') # H_true
plt.legend()
plt.savefig('./output/comparison_stacked_hist_H.png')
plt.close()

plt.hist([H, H2, H3, H_combined], alpha=.8, bins=30, histtype='step', density=True, linewidth=2, label = ['Event1', 'Event2', 'Event3', 'Combined'])
plt.ylabel('Probability')
plt.xlabel('H_0')
plt.axvline(x=H_true.value, color='black', label='Planck18 H_0') # H_true
plt.legend()
plt.savefig('./output/comparison_line_hist_H.png')
plt.close()
