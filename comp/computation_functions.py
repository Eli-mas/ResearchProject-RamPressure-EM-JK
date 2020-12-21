import operator

import numpy as np

from prop.asy_prop import *
from prop.asy_defaults import *
from .array_functions import reindex, ensure_elementwise, ensure_columnwise
from common.arrays.roll_wrap_funcs import *
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from numpy import ndarray

from common.decorators import add_doc

def reflect_reciprocate(ar, mod=al, **kw):
	"""Divide an array by its reindexing over a specified window."""
	return ar/reindex(ar, mod, **kw)

def cdf_(sorted_data, return_sum=False):
	"""
	as implied by name, 'sorted_data' must be sorted for this to work
	for unsorted data, use the function 'make_cdf'
	"""
	cdf=np.add.accumulate(sorted_data)
	
	if return_sum: asum=cdf[-1]
	
	cdf/=cdf[-1]
	
	if return_sum: return cdf, asum
	return cdf

def make_cdf(d):
	"""
	take any sortable array d (need not be sorted already), 
	and return the x and y for the cdf of the data in an ndarray
	"""
	vals, counts=np.unique(d, return_counts=True)
	return np.array([vals, np.add.accumulate(counts, dtype=float)/d.shape[0]])

"""def smooth(ar, indices, axis=0):
	alen=len(ar)
	return np.array([np.average(ar[(i+indices)%alen], axis=axis) for i in range(alen)])"""


def quadrature_sum(a, b):
	"""Quadrature sum for two arguments."""
	return np.sqrt((a**2)+(b**2))
def quadrature_sub(a, b):
	"""Quadrature difference for any number of arguments."""
	return np.sqrt((a**2)-(b**2))
def qsum_n(*args):
	"""Quadrature sum for any number of arguments."""
	return np.sqrt(np.sum(np.array(args)**2))

def sort_arrays_by_array(sorter, arrays_to_sort, reverse=False):
	"""Given a `sorter` array and an iterable of other arrays.
	sort the other arrays by the sorting order used to sort the
	sorter array. If `reverse`, sort in descending order."""
	sort_inds=np.argsort(sorter)
	if reverse: sort_inds=sort_inds[::-1]
	return [a[sort_inds] for a in arrays_to_sort]

def c_to_p(x_coor, y_coor, xc, yc, sorted=False):
	'''
	converts cartesian (x_coor, y_coor) to polar coordinates
	(x_center, y_center) is treated as (0, 0)
	returns (angle, radius)
	if sorted=True and the input data are coordinate arrays, the results will be returned sorted by angle (ascending)
	-- -- --
	the original [unvectorized] algorithm(s) may be found in "adf non_decommented 9-6-18.py"
	'''
	#eta measured from North
	r_sky=quadrature_sum(x_coor-xc, y_coor-yc)
	eta=(np.arctan2(y_coor-yc, x_coor-xc)-tau/4)%tau
	if sorted and isinstance(eta, ndarray):
		return sort_arrays_by_array(eta, (eta, r_sky))
	return eta, r_sky

def p_to_c(theta, radius, xc, yc):
	"""
	converts polar to cartesian coordinates
	tau/4 is added because N is typically 0 degrees in astronomy
	returns [x, y]
	"""
	theta=theta+tau/4
	x=xc+radius*np.cos(theta)
	y=yc+radius*np.sin(theta)
	return x, y

def ellipse(a, d_beam_arcsec, inclination, PA, b=None, normal=False, m2=False, eplot=False, polar_axis=None):
	"""
	`inclination`, `PA` provided in degrees
	"""
	inc=inclination*deg_to_rad
	if b is None:
		b=a*np.cos(inc) #review this--see how calculation is done in aj_138_6_1741.pdf
		#if debug: print 'ellipse axial ratio: %.1f/%.1f = %.3f --> %.3f'%(a, b, a/b, b/a)
		"""if debug and debug_master_keys['m2_calc']:
			print('\nellipse: original a =', a)
			print('ellipse: original b =', b)"""
		if m2:
			"""if debug: ab_ar=[[a, b]]"""
			r_beam_arcsec=d_beam_arcsec/2
			a=quadrature_sum(a, r_beam_arcsec)
			b=quadrature_sum(a*np.cos(inc), r_beam_arcsec*np.sin(inc))
			"""if debug and debug_master_keys['m2_calc']:
				print('ellipse: m=2 a =', a)
				print('ellipse: m=2 b =', b)
				ab_ar.append([a, b])
				print(np.array(ab_ar))"""
	phi=PA*deg_to_rad
	#print '%s ellipse: a=%.2f, b=%.2f, phi=%.2f, d_beam_arcsec=%.2f, ' %(filename, a, b, phi, d_beam_arcsec)
	#print "ellipse: R25(''), inclination %g %g" %(R25*60, inclination)
	e_angles=np.linspace(0, tau, a2l, endpoint=False)
	e_radii=(a*b)/np.sqrt((b*np.cos(e_angles-phi))**2+(a*np.sin(e_angles-phi))**2)
	#		np.array([(a*b)/np.sqrt((b*np.cos(t-phi))**2+(a*np.sin(t-phi))**2) for t in e_angles])
	if eplot:
		if polar_axis is None: polar_axis=plt.gca(polar=True)
		polar_axis.plot(e_angles, e_radii, ls='--')
	if normal: return e_angles, e_radii
	else:
		e_deg=e_angles*rad_to_deg
		return e_deg[(range_a2l-ahl)%a2l]-360, e_radii[(range_a2l-ahl)%a2l]

def deproject_graph(theta, r_sky, inclination, PA):
	'''the original algorithm(s) may be found in "adf non_decommented 9-6-18.py"'''
	inc=inclination*np.pi/180
	pos=PA*np.pi/180
	eta=theta*deg_to_rad-pos
	conv_fact=np.sqrt(np.power(1/np.cos(inc), 2)-(np.power(np.tan(inc)*np.cos(eta), 2)))
	r_gal=r_sky*conv_fact
	return r_gal, conv_fact

#def deproject(x_coor, y_coor, xc, yc, inclination, PA):
#	inc=inclination*np.pi/180
#	pos=PA*np.pi/180
#	r_sky=quadrature_sum(x_coor-xc, y_coor-yc)
#	eta=np.arctan2(y_coor-yc, x_coor-xc)-pos
#	conv_fact=np.sqrt(np.power(1/np.cos(inc), 2)-(np.power(np.tan(inc)*np.cos(eta), 2)))
#	#the only reason we are allowed to use eta as defined above in place of a more nuanced version of angular distance from
#	#PA (call it d_ep) in the np.cos term is because np.cos(d_ep)=abs(np.cos(eta)), thus np.cos^2(d_ep)=np.cos^2(eta)
#	r_gal=r_sky*conv_fact
#	return [r_gal, r_sky, 360*eta/tau, conv_fact]

def coorsgen(ar2d, null=0, t=True, as_columns=True):
	"""Given a matrix (array) representing a 2-d map of a galaxy,
	generate the coordinates of that galaxy as the cells
	where the matrix has non-null values. By default the `null`
	value is 0. `t` means 'take the transpose'. Results are returned
	as an (Nx2) array if `as_columns` is True, else as a (2xN) array."""
	ar2d = np.atleast_1d(ar2d)
	if ar2d.ndim<2 or ar2d.ndim>2:
		raise ValueError("'coorsgen': `ar2d` parameter should have 2 dimensions, has {ar2d.ndim}")
	res=np.array(np.where((ar2d.T if t else ar2d)!=null))
	return np.column_stack(res) if as_columns else res

def znan(z, remove_nan=False, rep=None, inplace=False):
	"""
	Converts 0 to nan -- OR -- if remove_nan=True, converts nan to 0
	
	CONSIDER: np.nan_to_num
	"""
	
	if inplace: zc=z
	else: zc=np.copy(z)
	if remove_nan:
		if rep is None: rep=0
		zc[np.isnan(zc)]=rep
	else:
		zc[zc==0]=nan
		if rep is not None: zc[~np.isnan(zc)]=rep
	return zc

def setnan(data, where, inplace=False):
	if not inplace: data=np.copy(data)
	"""if len(where)==1: where=slice(*where[0])
	else: where=tuple(slice(*w) for w in where)
	print(where)"""
	data[where]=nan
	if not inplace: return data

def zfloor(data, threshold=0, rep=None):
	"""Return modified version of array with values<threshold set to threshold.
	By default the threshold is 0, hence the name 'zfloor'."""
	z=np.copy(data)
	if rep is None: rep=threshold
	z[z<threshold]=rep
	return z

def find_containing_angle(theta, mod, window):
	"""
	find the angle which maximizes the amount of theta contained within +-window
	
	returns (center_windows, max_count):
		'max_count' is the maximum number of cases in `theta` that can be made to fit within `window`
		center_windows=list of arrays
			each array has shape (2, ) and contains (low, high)
			low, high are both angles
			for all angles that fall within these angles, 
				a region of width `window` contains `max_count` angles in theta
	"""
	
	tlen=theta.shape[0]
	theta=np.sort(theta%mod)
	
	low_inds=np.searchsorted(theta, (theta-window)%360, 'left')%tlen
	high_inds=(np.searchsorted(theta, (theta+window)%360, 'right')-1)%tlen
	
	highlow_inds_dif=(high_inds-low_inds)%tlen
	
	max_count=np.max(highlow_inds_dif)
	loc=np.where(highlow_inds_dif==max_count)[0]
	
	theta_spans=(theta[high_inds[loc]]-theta[low_inds[loc]])%mod
	theta_halves=theta_spans/2
	theta_centers=(theta[low_inds[loc]]+theta_halves)%mod
	
	ret=np.unique((theta_centers, window-theta_halves), axis=1)
	
	return ret, max_count+1

from common.arrays import round_to_nearest
"""def round_to_nearest(value, rounder, mode='round'):
	'''NOTE: floating-point errors can occur here, the resolution is not yet implmenented'''
	if mode=='round': func=np.round
	elif mode=='floor': func=np.floor
	elif mode=='ceil': func=np.ceil
	result=rounder*func(value/rounder)
	return result#np.round(result, int(np.floor(np.abs(np.log10(rounder%1)))))"""
	
def polar_invert(t, reference_point=0):
	"""subtract 360 from positive values; add 360 to negative values"""
	return t-np.sign(t-reference_point)*360

def polar_stabilize(theta, reference_point=0):
	"""at the moment, used for making neater-looking and (more importantly) more interpretable plots"""
	result=np.copy(theta)
	large_offset=(np.abs(result-reference_point)>=180)
	if isinstance(reference_point, ndarray): reference_point=reference_point[large_offset]
	result[large_offset]=polar_invert(result[large_offset], reference_point)
	return result

def sigfig(x, d=1):
	"""round number to specified precision."""
	try:
		return round(x, d-int(np.log10(np.abs(x))))
	except (ValueError, OverflowError):
		return x
	except TypeError:
		if isinstance(x, complex):
			return (sigfig(x.real, d)+1j*sigfig(x.imag, d))
		else: raise

def sig_asy_mask(ER, FR):
	"""Tell whether asymmetry is significant on the basis of `ER`, `FR` values.
	
	I.e., return (ER >= sig_ext_cutoff) & (FR >= sig_flux_cutoff).
	"""
	return (ER >= sig_ext_cutoff) & (FR >= sig_flux_cutoff)

def sig_asy_m2_mask(ER, FR, m2ER, m2FR):
	"""Return whether the m=1 asymmetry given by `ER`,`FR` is significant
	by comparing with the values given by `m2ER`, `m2FR`."""
	return (((ER-1)/(m2ER-1)) > sig_m1_m2_ext_cutoff) & ((FR-m2FR) > sig_m1_m2_flux_cutoff)

@add_doc(sig_asy_mask, sig_asy_m2_mask)
def sig_asy_full_mask(ER, FR, m2ER, m2FR):
	"""Return whether given asymmetry is significant on the basis of
	m=1 values and on comparison of these with m=2 values."""
	return sig_asy_mask(ER, FR) & sig_asy_m2_mask(ER, FR, m2ER, m2FR)

def sig_angle_mask(EA, FA):
	"""Given asymmetry angles `EA`, `FA` *measured relative to a known,
	true wind angle, tell whether the angles fall sufficiently close
	to the wind angle as determined by `sig_angle_cutoff`."""
	return (np.abs(EA) <= sig_angle_cutoff) & (np.abs(FA) <= sig_angle_cutoff)

def regress_linear(x,y, predict=None, round=-1):
	"""Linear regression via sklearn.LinearRegression,
	If `predict` is None, return (coef, intercept, score(x,y)) of the model.
	Otherwise, specify `predict` as input values for which to predict outputs,
	and return the same data but with the predictions appended to the tuple."""
	x = ensure_columnwise(x)
	reg = LinearRegression().fit(x,y)
	
	if predict is None:
		results = reg.coef_.squeeze(), reg.intercept_, reg.score(x,y)
	else:
		predict = ensure_columnwise(ensure_elementwise(predict))
		results = reg.coef_.squeeze(), reg.intercept_, reg.score(x,y), reg.predict(predict)
	if round >= 0: return tuple(np.round(v,round) for v in results)
	return results

def intersect_lines(m1b1, m2b2, eq=operator.eq):
	"""Given slope,intercept pairs for two lines,
	return the (x,y) point of their intersection.
	
	If both lines have the same slope, return a tuple of nan values.
	
	Determination of whether the lines have equal slope is by default
	performed by the == operator. If you want to use another *binary*
	operator, such as np.isclose, pass it to the `eq` parameter."""
	(m1,b1) = m1b1
	(m2,b2) = m2b2
	if eq(m1, m2): return (nan, nan)
	
	x = (b2-b1)/(m1-m2)	# x coordinate of intersection
	return x, m1*x+b1	# (x,y) coors. of intersection

def intersect_vline(m, b, vx):
	"""Given m = slope, b=y-intercept, vx = x-coordinate of vertical line,
	return the y-coordinate at which line with (m, b)
	intersects the vertical line."""
	return m*vx+b

def intersect_hline(m, b, hy):
	"""Given m = slope, b=y-intercept, hy = y-coordinate of horizontal line,
	return the x-coordiante at which line with (m, b)
	intersects the horizontal line."""
	return (hy-b)/m

__all__ = ('reflect_reciprocate','cdf_','make_cdf','quadrature_sum',
'quadrature_sub','qsum_n','sort_arrays_by_array','c_to_p','p_to_c','ellipse',
'deproject_graph','coorsgen','znan','setnan','zfloor','find_containing_angle',
'round_to_nearest','polar_invert','polar_stabilize','sigfig','sig_asy_mask',
'sig_asy_m2_mask','sig_asy_full_mask','sig_angle_mask','regress_linear',
'intersect_lines','intersect_vline','intersect_hline')
