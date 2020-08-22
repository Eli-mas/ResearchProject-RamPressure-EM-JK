"""basic quantities used for computation across this project."""

#try: TERMINAL_ROWS,TERMINAL_COLUMNS=(int(v) for v in check_output(['stty', 'size']).split())
#except CalledProcessError: TERMINAL_ROWS=TERMINAL_COLUMNS=None
"""debug=False
debug_multi=False
debug_multi_verbose=False
debug_keys={
'qscore offset':False,
'beam smear correction':False,
'score_gmean':False,
'tail spectrum':False,
'points':False,
'total_ar lexsort':False,
'njit extentlist':False,
'moving centroid radial':False,
'moving centroid cutoff':False,
'flux histograms':False,
'fr gradient':False,
'outer pixel test':False,
'optimizing extentlist':True
}
debug_master_keys={
'ratio_calc':any(debug_keys[k] for k in {'tail spectrum',}),
'm2_calc':any(debug_keys[k] for k in {'beam smear correction','score_gmean'}),
'moving centroid':any(debug_keys[k] for k in {'moving centroid radial','moving centroid cutoff'})
}
debug_master_keys['extra flux analyze']=bool(sum((
	debug_master_keys['moving centroid'], debug_keys['fr gradient'],
	debug_keys['flux histograms']
)))
debug_plot=False"""

#smoothing_test=True

#enable_numba_funcs=False if debug else False

#asy_noise=False
#noise_check=False
#noise_trials=1000


comp_plot_m=[1,2]#

#multi=(not debug)

######## {plotting}
#if asy_noise: plotting=False
#else: plotting=True

#if multi: parallel_plotting=True
#else: parallel_plotting=False
########\{plotting}

import numpy as np
tau = 2*np.pi

al=int(100) #al must be divisible by 4; 180-deg range
a2l=2*al #360-deg range
ahl=int(.5*al) #90-deg range
a1hl=int(1.5*al) #270-deg range
aql=int(.25*al) #45-deg range

deg_to_rad=tau/360
rad_to_deg=360/tau
deg_to_index=a2l/360
index_to_deg=360/a2l
rad_to_index=a2l/tau
index_to_rad=tau/a2l

sim_back_cutoff={'x':1,'y':2}#{'x':.24,'y':.24}

deg=u'\N{DEGREE SIGN}'

deny_m2_high_i=False
high_i_threshold=75 #i>75 --> high inclination

sig_ext_cutoff=1.2
sig_flux_cutoff=.1#.05
sig_m1_m2_ext_cutoff=1#1.2
sig_m1_m2_flux_cutoff=0#.1
sig_angle_cutoff = 20

from numpy import nan #, ndarray
np.warnings.filterwarnings('ignore')

__all__=('nan', 'comp_plot_m','tau','al','a2l','ahl','a1hl','aql','deg_to_rad',
'rad_to_deg','deg_to_index','index_to_deg','rad_to_index','index_to_rad',
'sim_back_cutoff','deg','deny_m2_high_i','high_i_threshold','sig_ext_cutoff',
'sig_flux_cutoff','sig_m1_m2_ext_cutoff','sig_m1_m2_flux_cutoff','sig_angle_cutoff',)


