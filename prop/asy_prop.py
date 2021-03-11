"""basic quantities used for computation across this project."""

from collections import OrderedDict
from common import Struct

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

deny_m2_high_i = True
high_i_threshold = HIGH_I_THRESHOLD = 75 #i>75 --> high inclination

sig_ext_cutoff=1.2
sig_flux_cutoff=.1#.05
sig_m1_m2_ext_cutoff=1#1.2
sig_m1_m2_flux_cutoff=0#.1
sig_angle_cutoff = 20
ASY_CUTOFFS = Struct(ER=sig_ext_cutoff, FR=sig_flux_cutoff)

M2_WEIGHT_SMOOTH = 2

from numpy import nan #, ndarray
np.warnings.filterwarnings('ignore')

INC_CUTOFFS = OrderedDict((('low', 35), ('med',65), ('high',90)))

__all__=('nan', 'tau','al','a2l','ahl','a1hl','aql','deg_to_rad',
'rad_to_deg','deg_to_index','index_to_deg','rad_to_index','index_to_rad',
'sim_back_cutoff','deg','deny_m2_high_i','high_i_threshold','sig_ext_cutoff',
'sig_flux_cutoff','sig_m1_m2_ext_cutoff','sig_m1_m2_flux_cutoff','sig_angle_cutoff',
'INC_CUTOFFS','ASY_CUTOFFS', 'HIGH_I_THRESHOLD', 'M2_WEIGHT_SMOOTH')


