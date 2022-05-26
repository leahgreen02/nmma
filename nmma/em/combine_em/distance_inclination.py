import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
from gwpy.table import Table
from combine_em_utils import KDE_multiply, logit

samples1 = Table.read('../../../outdir/injection_posterior_samples.dat', format="csv", delimiter=" ")
D1 = samples1['luminosity_distance']
i1 = np.cos(samples1['inclination_EM'])

samples2 = Table.read('../../../outdir2/injection_posterior_samples.dat', format="csv", delimiter=" ")
D2 = samples2['luminosity_distance']
i2 = np.cos(samples2['inclination_EM'])

em_list = ['', '1', '2', '3', '4', '10']

def combine_em(original_KDE, EM_event, D_range = [0,1000], i_range = [0,1], transform_type=logit)
    '''
    '''
    D = EM_event['luminosity_distance']
    i = np.cos(EM_event['inclination_EM'])

    D_transform, D_prior = logit(D, D_range, include_prior=True)
    i_transform, i_prior = logit(i, i_range, include_prior=True) 





for n in em_list:
    samples1 = Table.read('../../../outdir/injection_posterior_samples.dat', format="csv", delimiter=" ")
D1 = samples1['luminosity_distance']
i1 = np.cos(samples1['inclination_EM'])

# add config file
Dmin = 0
Dmax = 1000
imin = 0
imax = 1

il1 = np.log((i1)/(1-i1))
il2 = np.log((i2)/(1-i2))
Dl1 = np.log((D1-Dmin)/(Dmax-D1))
Dl2 = np.log((D2-Dmin)/(Dmax-D2))

prior_i = np.exp(il1)/(1+np.exp(il1))*(1-np.exp(il1)/(1+np.exp(il1)))
prior_D = np.exp(Dl1)/(1+np.exp(Dl1))*(1-np.exp(Dl1)/(1+np.exp(Dl1)))

D_kde1 = scipy.stats.gaussian_kde(Dl1,weights=1/prior_D)
D_kde2 = scipy.stats.gaussian_kde(Dl2)
i_kde1 = scipy.stats.gaussian_kde(il1,weights=1/prior_i)
i_kde2 = scipy.stats.gaussian_kde(il2)

D_jkde = KDE_multiply(D_kde1, D_kde2)
i_jkde = KDE_multiply(i_kde1, i_kde2)

resam_size = 10000
il_resam = i_jkde.resample(resam_size)
Dl_resam = D_jkde.resample(resam_size)

i_resam = (imax*np.exp(il_resam.flatten())+imin)/(1+np.exp(il_resam.flatten()))
D_resam = (Dmax*np.exp(Dl_resam.flatten())+Dmin)/(1+np.exp(Dl_resam.flatten()))

i_hist = plt.hist(i_resam) 
plt.savefig('./output/i_hist.png')
plt.close()
D_hist = plt.hist(D_resam) 
plt.savefig('./output/D_hist.png')
plt.close()
