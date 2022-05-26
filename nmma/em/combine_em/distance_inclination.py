import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit


def combine_EM_events(original_KDE, EM_event, D_range = [0,1000], i_range = [0,1], transform_type=logit, save_hist=False):
    '''
    '''
    D = EM_event['luminosity_distance']
    i = EM_event['inclination_EM']

    D_original = original_KDE['luminosity_distance']
    i_original = original_KDE['inclination_EM']

    print(np.min(i), np.max(i), np.mean(i), np.median(i))
    print('-------------')
    print(np.min(i_original), np.max(i_original), np.mean(i_original), np.median(i_original))

    D_transform, D_prior = logit(D, D_range, include_prior=True)
    i_transform, i_prior = logit(i, i_range, include_prior=True) 
    D_original_transform = logit(D_original, D_range)
    i_original_transform = logit(i_original, i_range)

    D_kde_transform = scipy.stats.gaussian_kde(D_transform, weights=1/D_prior)
    D_kde_original = scipy.stats.gaussian_kde(D_original_transform)
    i_kde_transform = scipy.stats.gaussian_kde(i_transform, weights=1/i_prior)
    i_kde_original = scipy.stats.gaussian_kde(i_original_transform)

    D_jkde = KDE_multiply(D_kde_transform, D_kde_original)
    i_jkde = KDE_multiply(i_kde_transform, i_kde_original)

    D_transform_resam = D_jkde.resample(len(D))
    i_transform_resam = i_jkde.resample(len(D))

    D_resam = logit(D_transform_resam, D_range, inv_transform=True)
    i_resam = logit(i_transform_resam, i_range, inv_transform=True) 
    
    if save_hist:
        D_hist = plt.hist(D_resam)
        plt.savefig('./output/D_hist.png')
        plt.close()
        i_hist = plt.hist(i_resam)
        plt.savefig('./output/i_hist.png')
        plt.close()

    return D_resam.flatten(), i_resam.flatten()

def run_event_combination(EM_event_files, save_hist=False):
    '''
    '''
    original_KDE = EM_event_files[0]
    for n, event in enumerate(EM_event_files[1:]):
        print('Combining events '+str(n+1)+' and '+str(n+2))
        D, i = combine_EM_events(original_KDE, event)
        original_KDE = {'luminosity_distance': D, 'inclination_EM': i}

    if save_hist:
        D_hist = plt.hist(D)
        plt.savefig('./output/D_hist.png')
        plt.close()
        i_hist = plt.hist(i)
        plt.savefig('./output/i_hist.png')
        plt.close()

samples1 = Table.read('../../../outdir/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples1['inclination_EM'] = np.cos(samples1['inclination_EM'])

samples2 = Table.read('../../../outdir2/injection_posterior_samples.dat', format="csv", delimiter=" ")
samples2['inclination_EM'] = np.cos(samples2['inclination_EM'])

em_list = ['', '1', '2', '3', '4', '10']

run_event_combination([samples1, samples2], save_hist=True)
