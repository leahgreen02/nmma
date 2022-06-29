import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import joypy
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit
from bilby.gw.conversion import luminosity_distance_to_redshift

# v = H0*D
# H0 ~ 70 # km/s/Mpc
c = 3e5 # km/s
# z ~ H0 * D / c
# H0 = z * c /D

def combine_EM_events(original_KDE, EM_event, H_range = [0,150], D_range = [0,200], i_range = [0,1], transform_type=logit, downsample=True):
    '''
    '''
    D = EM_event['luminosity_distance']
    i = EM_event['cos_inclination_EM']

    z = np.array(luminosity_distance_to_redshift(D))

    H = z * c / D

    print('Transforming...')
    H_transform, H_prior = logit(H, H_range, include_prior=True)
    D_transform, D_prior = logit(D, D_range, include_prior=True)
    i_transform, i_prior = logit(i, i_range, include_prior=True) 

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
    for n, event in enumerate(EM_event_files[1:]):
        print('Combining events '+str(n+1)+' and '+str(n+2))
        D_kde, i_kde, H_kde = combine_EM_events(original_KDE, event)
        original_KDE = {'luminosity_distance': D_kde, 'cos_inclination_EM': i_kde, 'H0': H_kde}
        print('Events combined!')

    H_transform_resam = H_kde.resample(N_hist)
    D_transform_resam = D_kde.resample(N_hist)
    i_transform_resam = i_kde.resample(N_hist)

    H_resam = logit(H_transform_resam, H_range, inv_transform=True)
    D_resam = logit(D_transform_resam, D_range, inv_transform=True)
    i_resam = logit(i_transform_resam, i_range, inv_transform=True)

    if save_hist:
        plt.hist(H_resam.flatten(), bins=20)
        plt.xlabel('H_0')
        plt.ylabel('Probability')
        plt.savefig('./output/H_downsample_hist.png')
        #plt.savefig('./output/H_hist.png')
        plt.close()
        plt.hist(D_resam.flatten(), bins=20)
        plt.xlabel('Luminosity Distance')
        plt.ylabel('Probability')
        plt.savefig('./output/D_downsample_hist.png')
        #plt.savefig('./output/D_hist.png')
        plt.close()
        plt.hist(i_resam.flatten(), bins=20)
        plt.xlabel('Cos Inclination Angle')
        plt.ylabel('Probability')
        plt.savefig('./output/i_downsample_hist.png')
        #plt.savefig('./output/D_hist.png')
        plt.close()
    return D_resam.flatten(), i_resam.flatten(), H_resam.flatten()

samples1 = Table.read('../../../outdir/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples1['cos_inclination_EM'] = np.cos(samples1['inclination_EM'])

samples2 = Table.read('../../../outdir2/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples2['cos_inclination_EM'] = np.cos(samples2['inclination_EM'])

samples3 = Table.read('../../../outdir3/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples3['cos_inclination_EM'] = np.cos(samples3['inclination_EM'])

samples4 = Table.read('../../../outdir4/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples4['cos_inclination_EM'] = np.cos(samples4['inclination_EM'])

D = samples1['luminosity_distance']
i = samples1['cos_inclination_EM']

D2 = samples2['luminosity_distance']
i2 = samples2['cos_inclination_EM']

D3 = samples3['luminosity_distance']
i3 = samples3['cos_inclination_EM']

D4 = samples4['luminosity_distance']
i4 = samples4['cos_inclination_EM']

D_range = [0,200] 
i_range = [0,1]
H_range = [0,150]

z = np.array(luminosity_distance_to_redshift(D))
H = z * c / D
z2 = np.array(luminosity_distance_to_redshift(D2))
H2 = z2 * c / D2
z3 = np.array(luminosity_distance_to_redshift(D3))
H3 = z3 * c / D3
z4 = np.array(luminosity_distance_to_redshift(D4))
H4 = z4 * c / D4

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
event_list = [original_KDE, samples2, samples3, samples4]
#event_list = [original_KDE, samples1, samples2]

D_combined, i_combined, H_combined = run_event_combination(event_list, N_hist=3619, save_hist=True)
tab = Table([D_combined, D], names=['Combined Events', 'Event1'])

plt.hist(D, alpha=.5, label = 'Event1')
plt.hist(D2, alpha=.5, label = 'Event2')
plt.hist(D3, alpha=.5, label = 'Event3')
plt.hist(D4, alpha=.5, label = 'Event4')
plt.hist(D_combined, label = 'Combined')
plt.ylabel('Probability')
plt.xlabel('Luminosity Distance')
plt.legend()
plt.savefig('./output/comparison_hist_D.png')
plt.close()

plt.hist(i, alpha=.5, label = 'Event1')
plt.hist(i2, alpha=.5, label = 'Event2')
plt.hist(i3, alpha=.5, label = 'Event3')
plt.hist(i4, alpha=.5, label = 'Event4')
plt.hist(i_combined, label = 'Combined')
plt.ylabel('Probability')
plt.xlabel('Cos Inclination Angle')
plt.legend()
plt.savefig('./output/comparison_hist_i.png')
plt.close()

plt.hist(H, alpha=.5, label = 'Event1')
plt.hist(H2, alpha=.5, label = 'Event2')
plt.hist(H3, alpha=.5, label = 'Event3')
plt.hist(H4, alpha=.5, label = 'Event4')
plt.hist(H_combined, label = 'Combined')
plt.ylabel('Probability')
plt.xlabel('H_0')
plt.legend()
plt.savefig('./output/comparison_hist_H.png')
plt.close()

plt.hist([D, D2, D3, D4], alpha=.8, bins = 30, stacked = True, density=True, label = ['Event1', 'Event2', 'Event3', 'Event4'])
plt.ylabel('Probability')
plt.xlabel('Luminosity Distance')
plt.legend()
plt.savefig('./output/comparison_hist2_D.png')
plt.close()

plt.hist([H, H2, H3, H4, H_combined], alpha=.8, bins = 30, stacked = True, density=True, label = ['Event1', 'Event2', 'Event3', 'Event4', 'Combined'])
plt.ylabel('Probability')
plt.xlabel('H_0')
plt.legend()
plt.savefig('./output/comparison_hist2_H.png')
plt.close()

plt.hist([H, H2, H3, H4, H_combined], alpha=.8, bins=30, histtype='step', density=True, label = ['Event1', 'Event2', 'Event3', 'Event4', 'Combined'])
plt.ylabel('Probability')
plt.xlabel('H_0')
plt.legend()
plt.savefig('./output/comparison_hist3_H.png')
plt.close()
