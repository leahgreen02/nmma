import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import joypy
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit


def combine_EM_events(original_KDE, EM_event, D_range = [0,1000], i_range = [0,1], transform_type=logit, downsample=True):
    '''
    '''
    D = EM_event['luminosity_distance']
    i = EM_event['cos_inclination_EM']

    # samples, not KDE???
    #D_original = original_KDE['luminosity_distance']
    #i_original = original_KDE['cos_inclination_EM']

    #print(np.min(i), np.max(i), np.mean(i), np.median(i))
    #print('-------------')
    #print(np.min(i_original), np.max(i_original), np.mean(i_original), np.median(i_original))
    print('Transforming...')
    D_transform, D_prior = logit(D, D_range, include_prior=True)
    i_transform, i_prior = logit(i, i_range, include_prior=True) 
    #D_original_transform = logit(D_original, D_range)
    #i_original_transform = logit(i_original, i_range)

    D_kde_transform = scipy.stats.gaussian_kde(D_transform, weights=1/D_prior)
    #D_kde_original = scipy.stats.gaussian_kde(D_original_transform)
    i_kde_transform = scipy.stats.gaussian_kde(i_transform, weights=1/i_prior)
    #i_kde_original = scipy.stats.gaussian_kde(i_original_transform)

    D_kde_original = original_KDE['luminosity_distance']
    i_kde_original = original_KDE['cos_inclination_EM']
    print(D_kde_original, i_kde_original)

    print('Taking joint KDE')
    D_joint_kde = KDE_multiply(D_kde_transform, D_kde_original, downsample=downsample)
    i_joint_kde = KDE_multiply(i_kde_transform, i_kde_original, downsample=downsample)

    '''
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
    '''
    print('Done!')
    return D_joint_kde, i_joint_kde

def run_event_combination(EM_event_files, save_hist=False, N_hist = 1000):
    '''
    '''
    original_KDE = EM_event_files[0]
    for n, event in enumerate(EM_event_files[1:]):
        print('Combining events '+str(n+1)+' and '+str(n+2))
        D_kde, i_kde = combine_EM_events(original_KDE, event)
        original_KDE = {'luminosity_distance': D_kde, 'cos_inclination_EM': i_kde}
        print('Events combined!')

    D_transform_resam = D_kde.resample(N_hist)
    i_transform_resam = i_kde.resample(N_hist)

    D_resam = logit(D_transform_resam, D_range, inv_transform=True)
    i_resam = logit(i_transform_resam, i_range, inv_transform=True)

    if save_hist:
        plt.hist(D_resam.flatten(), bins=20)
        plt.savefig('./output/D_downsample_hist.png')
        #plt.savefig('./output/D_hist.png')
        plt.close()
        plt.hist(i_resam.flatten(), bins=20)
        plt.savefig('./output/i_downsample_hist.png')
        #plt.savefig('./output/D_hist.png')
        plt.close()
    return D_resam.flatten(), i_resam.flatten()

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

D_range = [0,1000] 
i_range = [0,1]

D_transform, D_prior = logit(D, D_range, include_prior=True)
i_transform, i_prior = logit(i, i_range, include_prior=True)

D_kde_transform = scipy.stats.gaussian_kde(D_transform, weights=1/D_prior)
i_kde_transform = scipy.stats.gaussian_kde(i_transform, weights=1/i_prior)

original_KDE = {}
original_KDE['luminosity_distance'] = D_kde_transform
original_KDE['cos_inclination_EM'] = i_kde_transform

#em_list = ['', '1', '2', '3', '4', '10']
event_list = [original_KDE, samples2, samples3, samples4]
#event_list = [original_KDE, samples1, samples2]


D_combined, i_combined = run_event_combination(event_list, N_hist=3619, save_hist=True)
#D = D[0:10]
#D2 = D2[0:10]
#D_combined = D_combined[0:10]
print(D_combined, D_combined.shape)
tab = Table([D_combined, D], names=['Combined Events', 'Event1'])
#print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).shape)
#print(np.array([D_combined, D]).shape)
#print(np.arange(0,10))
#D_combined = np.array([[0]*3619, D_combined])
#D = np.array([[1]*3619, D])
#D2 = np.array([[2]*3619, D2])
#D3 = np.array([[3]*3619, D3])
#D4 = np.array([[4]*3619, D4])

#D_clabel = np.array([['Event1']*10, D_combined])
#D_label = np.array([['Event1']*10, D])
#D2_label = np.array([['Event2']*10, D2])
print(D)
#df = pd.DataFrame(np.array([tab['Combined Events'], tab['Event1']]), columns=['Combined Events', 'Event1'])
#df = pd.DataFrame(np.vstack([D_combined, D, D2]).T, columns=['Combined Events', 'Event1', 'Event2'], index=np.arange(0,10))
#df = pd.DataFrame(np.hstack([D_combined, D, D2, D3, D4]).T, columns=['Name', 'Distance'], index=np.arange(0,3619))
#print(df)
#df['Distance'] = np.float(df['Distance'])
#fig, axes = joypy.joyplot(df, by='Name', bins=20, overlap=0)
#plt.savefig('comparison_hist_D.png')
#plt.close()
plt.hist(D, alpha=.5, label = 'Event1')
plt.hist(D2, alpha=.5, label = 'Event2')
plt.hist(D3, alpha=.5, label = 'Event3')
plt.hist(D4, alpha=.5, label = 'Event4')
plt.hist(D_combined, label = 'Combined')
plt.ylabel('Probability')
plt.xlabel('Luminosity Distance')
plt.legend()
plt.savefig('comparison_hist2_D.png')
plt.close()

plt.hist(i, alpha=.5, label = 'Event1')
plt.hist(i2, alpha=.5, label = 'Event2')
plt.hist(i3, alpha=.5, label = 'Event3')
plt.hist(i4, alpha=.5, label = 'Event4')
plt.hist(i_combined, label = 'Combined')
plt.ylabel('Probability')
plt.xlabel('Cos Inclination Angle')
plt.legend()
plt.savefig('comparison_hist2_i.png')
plt.close()
