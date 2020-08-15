import functools
import numpy as np
from numpy import ma
from comp import computation_functions, polar_functions, array_functions
from comp.array_functions import broadcast_by, get_region_inds
from comp.polar_functions import polar_std_offset, polar_mean_offset
from prop.asy_prop import *

# INITIAL = 

@functools.wraps(computation_functions.sig_asy_mask)
def sig_asy_mask(self):
	return computation_functions.sig_asy_mask(self.ER, self.FR)

def sig_asy_m2_mask(self):
	try: return computation_functions.sig_asy_m2_mask(self.ER, self.FR, self.m2ER, self.m2FR)
	except AttributeError: return computation_functions.sig_asy_mask(self.ER, self.FR)

def sig_asy_full_mask(self):
	try: return computation_functions.sig_asy_full_mask(self.ER, self.FR, self.m2ER, self.m2FR)
	except AttributeError: return computation_functions.sig_asy_mask(self.ER, self.FR)

def sig_angle_mask(self):
	return computation_functions.sig_angle_mask(self.EA, self.FA)

def asy_mask(self, m2):
	return self.sig_asy_full_mask() if m2 else self.sig_asy_mask()

def offset(self, *attr, alternate_angle = None, squeeze=True, **k):
# 	print('galaxyseries_comp_funcs.offset: attr:',attr)
# 	for a in attr: print(a)
	res = polar_functions.polar_offset(
		[getattr(self,a) if isinstance(a,str) else a for a in attr],
		self.WA if (alternate_angle is None) else alternate_angle,
		**k
	)
	res *= self.rot
# 	print('GalaxySeries.offset: res:',res)
	return res.squeeze() if squeeze else res
	

def abs_offset(self, *a, **k):
	return np.abs(self.offset(*a, **k))

def time_evolving_side_mean(self, center=None, ax=None):
	window=180
	start_ind = int((center - window / 2) * deg_to_index)
	result = array_functions.reindex(self.extentlist_graph[:, :, 1], start_ind, axis=1)[:, :int(window * deg_to_index)]
	return (result / result[0]).mean(axis=1)

def _attr_sig(self,attr,m2=True):
	"""
	Return a masked array which will leave visible only data at times when asymmetry is significant
	:param self: _GalaxySeries instance
	:param attr: attribute name
	:param m2: whether asymmetry mask incorporates m1::m2 values
	:return: masked array
	"""
	if isinstance(attr,str): attr = getattr(self,attr)
	return ma.masked_array(attr, mask = broadcast_by(~self.asy_mask(m2=m2), attr, 0))

def attr_sig(self,*attr,m2=True,squeeze=False):
	"""
	call _attr_sig on attributes specified in varargs `attr`parameter

	Parameters
	----------
	:param self:
	:param attr:
	:param m2:
	:param squeeze: squeeze result (fit to lowest possible dimensionality)
	:return:
	"""
# 	res = ma.masked_array([self._attr_sig(a, m2=m2) for a in attr])
	attr = tuple(getattr(self, a) if isinstance(a, str) else a for a in attr)
	res = ma.masked_array(
		attr,
		mask = np.broadcast_to(~self.asy_mask(m2=m2), (len(attr), len(self)))
	)
	return res.squeeze() if squeeze else res

def sig_offset_std(self, *attrs, m2=True):
	"""return mean offset for attributes over times when asymmetry is significant"""
	return polar_std_offset(
		self.attr_sig(*self.offset(*attrs, squeeze=False), m2=m2),
	axis=1, mod=360)

def sig_delta_std(self, *attrs, m2=True):
	"""return mean displacement for attributes over times when asymmetry is significant"""
	return self.attr_sig(
		*self.abs_offset(*attrs, squeeze=False),m2=m2
	).std(axis=1).data.squeeze()

def sig_offset_mean(self, *attrs, m2=True):
	"""return mean offset for attributes over times when asymmetry is significant"""
	return self.attr_sig(
		*self.offset(*attrs, squeeze=False), m2=m2
	).mean(axis=1).data.squeeze()

def sig_delta_mean(self, *attrs, m2=True):
	"""return mean abs offset for attributes over times when asymmetry is significant"""
	
	# print(f'sig_delta_mean: attr={attr}, m2={m2}')
	#data = self.attr_sig(*(self.abs_offset(a) for a in attr),m2=m2,squeeze=False)
	#try: assert np.allclose(data,self.abs_offset(self.attr_sig(*attr,m2=m2,squeeze=False)))
	#except AssertionError:
	#	print(data - self.abs_offset(self.attr_sig(*attr,m2=m2,squeeze=False)))
	#	raise
	#return data.mean(axis=1).squeeze()
	
	### print('sig_delta_mean:',self.directory,self.attr_sig(*self.abs_offset(*attrs),m2=m2),sep='\n')
	
	return self.attr_sig(
		*self.abs_offset(*attrs, squeeze=False),m2=m2
	).mean(axis=1).data.squeeze()

def sig_delta_max(self, *attrs, m2=True):
	"""return max abs offset for attributes over times when asymmetry is significant"""
	return self.attr_sig(
		*self.abs_offset(*attrs, squeeze=False), m2=m2
	).max(axis=1).data.squeeze()

def sig_delta_min(self, *attrs, m2=True):
	"""return min abs offset for attributes over times when asymmetry is significant"""
	return self.attr_sig(
		*self.abs_offset(*attrs, squeeze=False), m2=m2
	).min(axis=1).data.squeeze()

def angle_sig_accuracy(self, *attr, threshold=sig_angle_cutoff, m2=True):
	return (self.attr_sig(*self.abs_offset(*attr),m2=m2,squeeze=False)<=threshold).mean(axis=1).squeeze()

def sig_attr_mean(self, *attrs, m2=True):
	"""return max abs offset for attributes over times when asymmetry is significant"""
	return self.attr_sig(*attrs, m2=m2).mean(axis=1).data

def sig_attr_max(self, *attrs, m2=True):
	"""return max abs offset for attributes over times when asymmetry is significant"""
	return self.attr_sig(*attrs, m2=m2).max(axis=1).data

def sig_attr_min(self, *attrs, m2=True):
	"""return min abs offset for attributes over times when asymmetry is significant"""
	return self.attr_sig(*attrs, m2=m2).min(axis=1).data

def get_sig_asy_primary_start(self, m2):
	r = get_region_inds(self.asy_mask(m2=m2))
	r = max(r, key = len)
	return self.time_index[r[0]]

def gas_content_at_time(self, t):
	return self.time(t).zdata.sum()

def initial_gas_content(self):
	return self.instances[0].zdata.sum()

def gas_fraction(self):
	return np.array([g.zdata.sum() for g in self.instances]) / self.initial_gas_content()

def mean_offsets_across_angles(self,*angles):
	return polar_mean_offset(self.getattrs(*angles),axis=0,ref=self.WA)

def std_offsets_across_angles(self,*angles):
	return polar_std_offset(self.getattrs(*angles),axis=0)

__all__ = ('sig_asy_mask','sig_asy_m2_mask','sig_asy_full_mask','sig_angle_mask',
'asy_mask','offset','abs_offset','time_evolving_side_mean','_attr_sig','attr_sig',
'sig_offset_std','sig_offset_mean','sig_delta_mean','sig_delta_max','sig_delta_min',
'angle_sig_accuracy','sig_attr_mean','sig_attr_max','sig_attr_min','gas_fraction',
'get_sig_asy_primary_start','gas_content_at_time','initial_gas_content',
'mean_offsets_across_angles','std_offsets_across_angles','sig_delta_std')