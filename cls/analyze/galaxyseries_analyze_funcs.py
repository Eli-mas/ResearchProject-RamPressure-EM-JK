import numpy as np
from numpy import ma
import pandas as pd
from matplotlib import pyplot as plt
from asy_io import asy_paths, asy_io
from plot.plotting_functions import keepax
from prop.asy_prop import *
from comp import computation_functions
#from cls import adc

# INITIAL = 

def evaluate_ratio_cutoffs_condition(self,attrs,cutoffs):
	#AXES: 0 ratio type, 1 timestep
	return (self.getattrs(*attrs,asarray=True) >= np.vstack(cutoffs)).all(axis=0)

def accuracy(self,*attr,
			 ext_cutoff=sig_ext_cutoff,flux_cutoff=sig_flux_cutoff,
			 m1_m2_ext_cutoff = sig_m1_m2_ext_cutoff,
			 m1_m2_flux_cutoff = sig_m1_m2_flux_cutoff,
			 angle_cutoff = sig_angle_cutoff,
			 #m2=True
			 ):
	#AXES: 0 angle type, 1 timestep
	offsets = ma.masked_array(self.offset(*self.getattrs(*attr)))
	abs_offsets = np.abs(offsets)

	#AXES:	0 angle type
	accuracy_raw = np.mean(abs_offsets <= angle_cutoff,axis=1)
	mean_offset = np.mean(offsets,axis=1)
	mean_abs_offset = np.mean(abs_offsets,axis=1)

	condition = ~self.evaluate_ratio_cutoffs_condition(('ER','FR'), (ext_cutoff,flux_cutoff))
	offsets.mask = condition
	abs_offsets.mask = condition
	accuracy_by_m1 = np.mean(abs_offsets <= angle_cutoff,axis=1)
	mean_offset_by_m1 = np.mean(offsets,axis=1)
	mean_abs_offset_by_m1 = np.mean(abs_offsets,axis=1)

	try:#if m2:
		condition *= ~self.evaluate_ratio_cutoffs_condition(
			( (self.ER-1)/(self.m2ER-1), self.FR-self.m2FR),
			(m1_m2_ext_cutoff,m1_m2_flux_cutoff)
		)
		offsets.mask = condition
		abs_offsets.mask = condition
		accuracy_by_full = np.mean(abs_offsets <= angle_cutoff, axis=1)
		mean_offset_by_full = np.mean(offsets, axis=1)
		mean_abs_offset_by_full = np.mean(abs_offsets,axis=1)
	except AttributeError:
		accuracy_by_full = mean_offset_by_full = mean_abs_offset_by_full = [nan]*len(attr)

	#AXES:	0 sig asy criterion
	#		1 angle type
	accuracy_results = np.array((accuracy_raw, accuracy_by_m1, accuracy_by_full))
	offset_results = np.array((mean_offset, mean_offset_by_m1, mean_offset_by_full))
	abs_offset_results = np.array((mean_abs_offset, mean_abs_offset_by_m1, mean_abs_offset_by_full))

	#AXES:	0 aggregation type <fraction accurate, average offset>
	#		1 sig asy criterion
	#		2 angle type
	return np.array((accuracy_results,offset_results,abs_offset_results))


def tabulate_angle_results(self):
	columns=['ER','m2ER','FR','m2FR','ER::m2ER','FR::m2FR','sig_m1','sig_m1, m1/m2']
	df = pd.DataFrame(index=self.time_index, columns=columns)
	for attr in columns[:4]:
		try: df[attr] = getattr(self,attr)
		except AttributeError: df[attr]='/'
	try:
		df['ER::m2ER'] = (df['ER']-1)/(df['m2ER']-1)
		df['FR::m2FR'] = df['FR'] - df['m2FR']
	except TypeError:
		df['ER::m2ER'] = df['FR::m2FR'] = '/'
	sig = computation_functions.sig_asy_mask(df['ER'],df['FR'])
	df['sig_m1'][sig] = '*'
	df['sig_m1'][~sig] = ''
	try:
		sig = computation_functions.sig_asy_full_mask(df['ER'], df['FR'],df['m2ER'], df['m2FR'])
		df['sig_m1, m1/m2'][sig] = '*'
		df['sig_m1, m1/m2'][~sig] = ''
	except TypeError:
		df['sig_m1, m1/m2'] = '/'
	with open(asy_io.makepath(asy_paths.OUTPUT_DATA_PATH+'vsim sig asy results/')+
			  f'{self.directory} sig asy table.file','w') as f:
		f.write(df.to_string())

__all__ = ('evaluate_ratio_cutoffs_condition','accuracy','tabulate_angle_results')

if __name__ == '__main__':
	#AXES:	0 simulation
	#		1 aggregation type <fraction accurate, average offset, average abs offset>
	#		2 sig asy criterion <NONE, m1, m1+m1::m2>
	#		3 angle type
	from common.common.cls import Proxy
	from cls.adc import VollmerSeriesCollection
	series = Proxy(tuple(s() for s in VollmerSeriesCollection()))
	attrs = series[0].EWA
	results = [s.accuracy(*attrs) for s in series]
	results = np.array(results)
	print(results.shape)
	for i,a in enumerate(attrs):
		ax = plt.subplot(title=a)
		attr__full_mask__accuracy = results[:,0,1,i]
		#ax.scatter([s.DWA for s in series],[s.inc for s in series],s=500*EA__full_mask__accuracy)
		keepax([s.DWA for s in series],[s.inc for s in series],ax=ax)
		for ac,d,inc in zip(attr__full_mask__accuracy.round(2),[s.DWA for s in series],[s.inc for s in series]):
			ax.text(d,inc,ac,ha='center',va='center')
		plt.show()