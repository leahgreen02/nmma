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
from scipy.interpolate import interp1d
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit, calc_H0, R_from_M, q2eta, mc2ms, load_eos_posterior
from bilby.gw.conversion import luminosity_distance_to_redshift

# v = H0*D
#c = 3e5 # km/s
# z ~ H0 * D / c
# H0 = z * c /D
H_true = Planck18.H(0)
c = const.c.to('km/s').value

downsample_posterior = True
N_draws = 500
#N_draws = 50000

method = 'R14'

#D_range = [0, 500]
prior_range = [1, 2.05]

with h5py.File('GW170817_PE.hdf5', 'r') as data:
    GW170817_samples = data['posterior_samples'][()]

ALL_EOS_DRAWS = load_eos_posterior()


# cols in inj file: idx, longitude, latitude, inclination, distance, mass1, mass2, mass1_detector, mass2_detector, lambda1, lambda2, spin1z, spin2z
inj_filename = 'O4_BNS_injections_lambdas.dat'
injections = np.loadtxt(f'posteriors/{inj_filename}')

def combine_EM_events(original_KDE, EM_event, post_range, prior_range, D_inj = 'None', transform_type=logit, downsample=True, N_draws=N_draws, num=None):
    '''
    '''
    method = 'R14'
    if method == 'H0':
        D = EM_event['luminosity_distance']
        z = np.array(luminosity_distance_to_redshift(D_inj, cosmology=Planck18))
        if not downsample_posterior:
            N_draws = len(D)
        D = np.random.choice(D, size=N_draws)
        H = calc_H0(D, z)

    elif method == 'R14':
        EM_event = event
        mc, q = EM_event['chirp_mass'], EM_event['mass_ratio']
        eta = q2eta(q)
        m1, m2 = mc2ms(mc, eta)
        if not downsample_posterior:
            N_draws = len(m1)
        rand_idx = np.random.choice(N_draws, size=N_draws)
        m1, m2 = m1[rand_idx], m2[rand_idx]
        #m1, m2 = m1[m1 < 2.05], m2[m2 < 2.05]
        ms = np.hstack((m1, m2))

        eos_seed = 42
        num_eos_draws = len(ms)
        np.random.seed(eos_seed)
        prediction_nss, prediction_ems = [], []

        rand_subset = np.random.choice(
            len(ALL_EOS_DRAWS), num_eos_draws if num_eos_draws < len(ALL_EOS_DRAWS) else len(ALL_EOS_DRAWS))  # noqa:E501
        subset_draws = ALL_EOS_DRAWS[rand_subset]
        M, R = subset_draws['M'], subset_draws['R']
        max_masses = np.max(M, axis=1)
        f_Ms = [interp1d(m, r, bounds_error=False) for m, r in zip(M, R)]

        rs = []
        for f_M, m in zip(f_Ms, ms):
            rs.append(f_M(m))

        #f_M_eos = interp1d(ms,rs)
        #R14s = f_M_eos(1.4)
  
        R14 = rs[(rs > 13.5) & (rs < 14.5)] 
 
    plt.figure(figsize=(12,9))    
    plt.scatter(ms, rs)
    plt.ylabel('R')
    plt.xlabel('M')
    plt.savefig(f'./output/R14/m_vs_r_{num}.png')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.hist(R14)
    plt.xlabel('R')
    plt.xlabel('Counts')
    plt.savefig(f'./output/R14/R14_hist_{num}.png')
    plt.close()

    '''
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
    '''


    if method == 'H0':
        input_prior = bilby.gw.prior.UniformSourceFrame(minimum=prior_range[0], maximum=prior_range[1], name='luminosity_distance',latex_label='$D_L$')
        input_prior = input_prior.prob(D)
        H_transform, H_prior = logit(H, post_range, include_prior=True)
        finite_idx = np.isfinite(H_transform)
        H_transform, H_prior = H_transform[finite_idx], H_prior[finite_idx]

        H_kde_transform = scipy.stats.gaussian_kde(H_transform, weights=1/H_prior)
        D_kde = scipy.stats.gaussian_kde(D, weights=1/input_prior)
        H_kde_original = original_KDE['H0']
        D_kde_original = original_KDE['luminosity_distance']

        joint_kde = KDE_multiply(H_kde_transform, H_kde_original, downsample=downsample)
        #D_joint_kde = KDE_multiply(D_kde, D_kde_original, downsample=downsample)
    if method == 'R14':
        input_prior = bilby.gw.prior.Uniform(minimum=prior_range[0], maximum=prior_range[1])
        input_prior = input_prior.prob(ms)
        post_prior = bilby.gw.prior.Uniform(minimum=post_range[0], maximum=post_range[1])
        post_kde = scipy.stats.gaussian_kde(R14, weights=1/post_prior)
        joint_kde = KDE_multiply(post_kde, original_kde, downsample=downsample)

    return joint_kde


def run_event_combination(post_range, prior_range, N_draws=N_draws):
    '''
    '''
    method = 'R14'
    EM_event_files = []
    event_indices = []

    path_to_events = 'posteriors/O4'
    event_folders = os.listdir(path_to_events)

    # GW170817
    original_D_inj = 40
    D = GW170817_samples['luminosity_distance']
    m1, m2 = GW170817_samples['m1_detector_frame_Msun'], GW170817_samples['m2_detector_frame_Msun']
    ms = np.hstack((m1, m2))

    if method == 'R14':
        eos_seed = 42
        num_eos_draws = len(ms)
        np.random.seed(eos_seed)
        prediction_nss, prediction_ems = [], []

        rand_subset = np.random.choice(
            len(ALL_EOS_DRAWS), num_eos_draws if num_eos_draws < len(ALL_EOS_DRAWS) else len(ALL_EOS_DRAWS))  # noqa:E501
        subset_draws = ALL_EOS_DRAWS[rand_subset]
        M, R = subset_draws['M'], subset_draws['R']
        max_masses = np.max(M, axis=1)
        f_Ms = [interp1d(m, r, bounds_error=False) for m, r in zip(M, R)]

        rs = []
        for f_M, m in zip(f_Ms, ms):
            rs.append(f_M(m))

        R14 = rs[(rs > 13.5) & (rs < 14.5)]
        input_prior = bilby.gw.prior.Uniform(minimum=prior_range[0], maximum=prior_range[1])
        input_prior = input_prior.prob(ms)
        post_prior = bilby.gw.prior.Uniform(minimum=post_range[0], maximum=post_range[1])
        original_kde = scipy.stats.gaussian_kde(R14, weights=1/post_prior)

        R14_medians = []
        R14_stds = []
        R14_medians.append(np.median(R14))
        R14_stds.append(np.std(R14))

        N_events = 29
        N_events = 3
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
            #percentiles.append(percentileofscore(event['luminosity_distance'], D_injs[n]))
            joint_kde = combine_EM_events(original_KDE, event, post_range = post_range, prior_range = prior_range, num=n)
            #original_KDE = {'luminosity_distance': D_kde_result, 'H0': H_kde}
            #H_vals = logit(H_kde.resample(N_draws), post_range, inv_transform=True)
            #p_value = scipy.stats.percentileofscore(event['luminosity_distance'], D_injs[n]) / 100
            #p_value = min(p_value, 1-p_value)
            #if p_value > 0: # no p value cut for now
            R14_vals = joint_kde.resample(N_draws)
            R14_medians.append(np.median(R14_vals))
            R14_stds.append(np.std(R14_vals))
        print(f'Events combined!')

        #H_transform_resam = H_kde.resample(N_draws)
        #D_transform_resam = D_kde_result.resample(N_draws)

        #H_resam = logit(H_transform_resam, post_range, inv_transform=True)
        #D_resam = D_transform_resam
        return R14, np.array(R14_medians), np.array(R14_stds)


    if method == 'H0':
        z = np.array(luminosity_distance_to_redshift(original_D_inj, cosmology=Planck18))
        H = calc_H0(D, z)
        D_prior = bilby.gw.prior.UniformSourceFrame(minimum=prior_range[0], maximum=prior_range[1], name='luminosity_distance', cosmology = Planck18, latex_label='$D_L$')
        D_prior = D_prior.prob(D)
 
        H_transform, H_prior = logit(H, post_range, include_prior=True)

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

        '''
        plt.figure(figsize=(12,9))
        plt.hist(D_injs, alpha=.8, bins = 10, edgecolor = 'black')
        plt.ylabel('Probability')
        plt.xlabel('Distance')
        plt.legend()
        plt.savefig(f'./output/event_hists/distance/D_inj_hist_farah.png')
        plt.close()
        '''

        N_events = 29
        N_events = 3
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
            D_kde_result, H_kde = combine_EM_events(original_KDE, event, D_injs[n], post_range = post_range, prior_range = prior_range, num=n)
            original_KDE = {'luminosity_distance': D_kde_result, 'H0': H_kde}
            H_vals = logit(H_kde.resample(N_draws), post_range, inv_transform=True)
            p_value = scipy.stats.percentileofscore(event['luminosity_distance'], D_injs[n]) / 100
            p_value = min(p_value, 1-p_value)
            if p_value > 0: # no p value cut for now
                H_medians.append(np.median(H_vals))
                H_stds.append(np.std(H_vals))
        print(f'Events combined!')

        H_transform_resam = H_kde.resample(N_draws)
        D_transform_resam = D_kde_result.resample(N_draws)

        H_resam = logit(H_transform_resam, post_range, inv_transform=True)
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



if method == 'R14':
    post_range = [8,18]
    R14, R14_medians, R14_stds = run_event_combination(post_range = post_range, prior_range = prior_range)

    plt.figure(figsize=(12,9))
    plt.hist(R14, alpha=.8, bins = 30, edgecolor = 'black', density=True, label = 'Combined Events')
    plt.ylabel('Probability')
    plt.xlabel('R14')
    #plt.xlabel(r'H$_{0}$ [km/s/Mpc]')
    #plt.axvline(x=H_true.value, color='red', linestyle='--', linewidth=4, label=r'Planck18 H$_{0}$') # H_true
    plt.legend()
    plt.savefig('./output/R14/R14_hist_all.png')
    plt.close()

    print(H_stds, H_medians)

    f, (ax0, ax1) = plt.subplots(2, 1, figsize=(12,9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(np.arange(1,len(R14_medians)+1), R14_stds/R14_true.value, linewidth=3, marker='o')
    ax0.errorbar(np.arange(1,len(R14_medians)+1), R14_medians, yerr=2*R14_stds, linewidth=3, capsize=6, marker='o', ls='none', label='Combined Event Prediction')
    #ax0.axhline(y=H_true.value, color='red', linestyle='--', linewidth=2, label=r'Planck18 H$_{0}$')
    #ax0.set_ylabel(r'H$_{0}$ [km/s/Mpc]')
    #ax1.set_ylabel(r'Fractional H$_{0}$ Error')
    ax0.set_ylabel(r'R14')
    ax1.set_ylabel(r'Fractional R14 Error')
    ax1.set_yscale('log')
    ax1.set_ylim([1e-2, .3])
    plt.xlabel(r'Number of Events')
    ax0.legend()
    plt.savefig('./output/R14/R14_error.png')
    plt.close()

    np.savetxt('H_medians.txt', H_medians)
    np.savetxt('H_stds.txt', H_stds)

else:
    D_combined, H_combined, H_medians, H_stds = run_event_combination(post_range = post_range, prior_range = prior_range)

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
