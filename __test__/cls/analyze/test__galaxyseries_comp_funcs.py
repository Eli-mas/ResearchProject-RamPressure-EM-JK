import numpy as np
from cls.adc import *
from comp.polar_functions import *
from asy_io.asy_io import print_update

def test__mean_offsets_across_angles(series_type):
	s = series_type()
	ANGLES = ('EA', 'FA', 'EA_trig', 'FA_trig', 'HTA', 'qEA', 'qFA', 'TA')
	m = s.mean_offsets_across_angles(*ANGLES)
	a = np.array([getattr(s,a) for a in ANGLES])
	a = polar_reduction(np.nanmean,a,360,axis=0)
	a = polar_offset(a,0,360)
	assert np.allclose(a,m)
	
	a = np.array([getattr(s,a) for a in ANGLES])
	a = [polar_offset(polar_reduction(np.nanmean,a[:,i],360),0,360) for i in range(a.shape[1])]
	assert np.allclose(a,m)

if __name__=='__main__':
	for s in VollmerSeriesCollection():
		print_update(s.keywords['galaxy'],s.keywords['inclination'])
		test__mean_offsets_across_angles(s)
	print_update('')