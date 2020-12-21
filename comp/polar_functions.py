"""
This module handles computations involving polar quantities,
i.e. periodic/modular quantities where 0 = <some other value>
in a cyclic numeric domain. In particular, it encapsulates the
logic for computing averages, medians, and other aggregating
statistics for such quantities.
"""
import numpy as np
from numpy.ma import masked_array

from common.arrays.iterate import access_1d_strands, rollaxes, recreate_from_1d_strands

from asy_io.asy_io import lprint,print_update
from .array_functions import *
from functools import partial
from prop.asy_prop import *

# INITIAL=



def polar_offset(t,ref,mod=360):
	"""
	get the polar difference between two inputs, i.e.
	the angular difference between the two, bounded between (-mod/2, mod/2)
	:param t: input angle (numeric or array-like)
	:param ref: reference angle (numeric or array-like)
	:param mod: modular factor
	:return: t-ref, in the polar sense
	"""
	try:
		d = (t - ref) % mod
	except TypeError:
		d = (np.atleast_1d(t) - np.atleast_1d(ref)) % mod
		d[d > mod/2] -= mod
		return d
	
	try:
		# assumes numpy-like iterable
		d[d > mod/2] -= mod
		return d
	except TypeError:
		# if not iterable, we are working with a numeric type
		return d if (d <= mod/2) else d-mod

def rolling_polar_offset(ar,mod):
	"""
	return the element-wise polar offsets between pairs of values in a 1-d array
	:param ar:
	:param mod:
	:return:
	"""
	ar = ensure_elementwise(ar)
	return polar_offset(ar[1:],ar[:-1],mod)

def polar_reduce(ar,*a,**kw):
	"""see docs for polar_reduce and polar_reduce_2d"""
	if ar.ndim==1: return _polar_reduce(ar,*a,**kw)
	elif ar.ndim==2: return polar_reduce_2d(ar,*a,**kw)
	else: raise ValueError(f'polar_reduce is not configured to handles arrays of dimension {ar.ndim}')

def _polar_reduce(ar,mod,pr=False,return_start=False,return_index=False):#,modulate=True
	"""
	Given a 1-d array,
	sort the data in a polar sense--i.e., arrange the data into a new array in such a way that:
		the array is montonically increasing
		the distance between the first and last elements of the array is maximized,
			i.e., the array is confined to the smallest region possible of angular space

	DEPRECATED:
		-- By default, this function does not preserve the original values.
		-- Instead, the resultant polar-sorted array is modulated by the 'mod' value.
		-- If this behavior is not desired, set the keyword modulate=False, and original
		-- values will be preserved.
	UPDATED:
		** Arrays are always modulated. To get back modularly-equivalent
		** values to the originals, add the start value as returned by
		** `return_start` to the reduced array.

	Notes:
		axis keyword argument was removed; see 2-10-18 version
			reintroducing it would require a much cleaner implementation
	
		4-15-19: 2d cases can now be handled by `polar_reduce_2d` in 'polar_functions'
	"""
	if isinstance(ar,masked_array):
		# transform the original array and mask so that values correspond
		# between the two, then return a new masked array from this
		(data, mask), true_start, (inds,b) = _reduce_arrays_by_polar_1d(ar.data,ar.mask,mod)
		ar = masked_array(data, mask)
	
	else:
		ar=ensure_elementwise(ar)
		
		slice=0
		if pr: print('initial shape:',ar.shape)
		inds=np.argsort(ar%mod)
		ar=ar[inds]
		
		dif=(reindex(ar,1)-ar)%mod
		
		b=(1+np.argmax(dif))%ar.shape[0]
		
		ar=reindex(ar,b)
		
		true_start=ar[slice]
		
		ar=(ar-true_start)%mod
		#if modulate: ar = ar
	
	if return_start:
		if return_index: return ar, true_start, (inds,b)
		else: return ar, true_start
	elif return_index: return ar,(inds,b)
	else: return ar

def _reduce_arrays_by_polar_1d(polar_array,other,mod):
	"""
	Say values in two arrays correspond to one another,
	and one is a polar array that is reduced via _polar_reduce.
	This function transforms the other array so that values
	in both arrays still correspond after the reduction.
	
	"""
	reduced_polar_array, true_start, (inds,b) = \
		_polar_reduce(polar_array,360,return_index = True, return_start = True)
	
	return (reduced_polar_array, reindex(other[inds],b)), true_start, (inds,b)

def polar_reindex(a,mod):
	"""
	polar-sort an array, preserving original values
	NOTE--same functionality present in polar_reduce by setting keyword modulate=False
	but that functionality has become deprecated
	
	:param a: input array
	:param mod: modular factor
	:return: polar-sorted array
	"""
	return np.add(*polar_reduce(a,mod,return_start=True))%mod

def polar_reindex_multi(a,mod,*arrays): #use this for polar_fill in plotting_functions
	"""Reindex an arbitrary number of input arrays
	to maintaing alignment with the result of reducing
	`a` by polar reduction given `mod`."""
	reduced,(inds,start)=polar_reduce(a,mod,return_index=True)
	for ar in arrays:
		if np.atleast_1d(ar).shape!=np.atleast_1d(a).shape:
			raise ValueError('`polar_reindex_multi`: all arrays must have equal shape')
	return tuple(reindex(ar[inds],start) for ar in ((a,)+arrays))

def polar_reduction(reducing_function,ar,*a, nomask=True, **k):
	"""Apply an aggregating function over 1-d or 2-d polar data.
	See '_polar_reduction', 'polar_reduce_2d' for details."""
	ar = np.atleast_1d(ar)
	if ar.ndim==1:
		try: k.pop('squeeze')
		except KeyError: pass
		result = _polar_reduction(reducing_function,ar,*a,**k)
	elif ar.ndim==2:
		result = polar_reduction_2d(reducing_function,ar,*a,**k)
	else:
		raise ValueError(f'polar_reduction is not configured to handles arrays of dimension {ar.ndim}')
	
	if isinstance(result, masked_array) and nomask:
		return result.data
	return result

def _polar_reduction(reducing_function,*args,start=True,**kwargs):
	"""Apply an aggregating function over 1-d polar data.
	If 'start' is true, the aggregation is applied after
	true_start from '_polar_reduce' is added back to the
	original array; otherwise the reduction is performed
	on the reduced array. Note, for some statistics we
	do not want to add the starting value back in--
	e.g., standard deviation and variance.
	"""
	fkwargs={'axis':kwargs['axis']} if 'axis' in kwargs else {}
	kwargs['return_start']=True
	#kwargs.setdefault('modulate',True)
	reduction,true_start=polar_reduce(*args,**kwargs)
	if len(args)>1: mod=args[1]
	else: mod=kwargs['mod']
	result=reducing_function(reduction,**fkwargs)
	#print("initial result in polar_reduction:",result)
	if start: result=(np.squeeze(true_start)+result)%mod
	return result

def polar_reduce_2d(ar,mod,axis=0,pr=False, return_start=False):#,modulate=True
	"""
	performs the `polar_reduce` algorithm independently
	on the rows or columns (-->`axis`) of an array.
	This function is configured to properly handle nan values,
	which is not yet the case for the original algorithm.
	
	former version of this algorithm (change to nan handling)
	in polar_reduce_2d_tests.py
	
	returns: ar, true_start, inds, b (the roll amounts)
	"""
	
	if isinstance(ar,masked_array):
		(data, mask), *extra = reduce_arrays_by_polar(ar.data, ar.mask, mod, axis=axis)
		return (masked_array(data, mask), *extra)
	
	ar=ensure_elementwise(ar)
	if pr: lprint('ar:',ar)
	
	if ar.shape[axis]<=1:
# 		if return_start: return ar,np.zeros(ar.shape[1-axis])
# 		else: return ar
		return (ar, np.zeros(ar.shape[1-axis], dtype=int),
				np.argsort(ar,axis=axis),
				np.zeros(ar.shape[1-axis], dtype=int))
	
	ar=ar%mod
	sort_inds=np.argsort(ar,axis=axis)
	ar=np.sort(ar,axis=axis)
	
	if pr: lprint('ar (sorted):',ar)
	
	if np.any(np.isnan(ar)):
		nans=True
		nancheck=np.isnan(ar)
		ar=nanrep_filter(ar,axis=axis,pr=pr)
	else: nans=False
	
	dif=(reindex(ar,1,axis=axis)-ar)%mod
	
	b=(1+np.argmax(dif,axis=axis))%ar.shape[axis]
	ar=complex_reindex_2d(ar,b,axis=axis)
	true_start=np.copy(ar[:,0:1] if axis==1 else ar[0:1,:])
	#take a copy because 'ar' may subsequently be modified in-place
	
	if pr: lprint('dif:',dif,'b:',b,'ar (reindexed):',ar,'true_start:',true_start)
	
	if nans:
		nancheck_reindex=complex_reindex_2d(nancheck,b,axis=axis)
		nancheck_reindex_float=complex_reindex_2d(nancheck.astype(float),b,axis=axis)
		ar[nancheck_reindex]=nan
		if pr: lprint(	'nancheck',nancheck,'nancheck_reindex',nancheck_reindex,
						'nancheck_reindex (float)',nancheck_reindex_float,
						'ar (with nans re-filled)',ar)
		del nancheck
	
	if True:#modulate
		ar=(ar-true_start)%mod
		if pr: lprint('ar (modulated):',ar)
	
	return ar, true_start, sort_inds, b

def polar_reduce_nd(ar, mod, axis=-1, return_extra = False):
	"""not yet complete: the array is properly reduced, but
	true_start has to be reshaped such that numpy's broadcasting
	will properly add it back to the original array."""
	# funnel to lower-dimensional cases if relevant
	if ar.ndim==1:
		ar, true_start, (inds, b) = _polar_reduce(ar, mod, return_start = True, return_index = True)
		return (ar, (true_start, inds, b)) if return_extra else ar
	if ar.ndim==2:
		new, (true_start, inds, b) = polar_reduce_2d(ar, mod=mod, axis=axis)
		return (ar, (true_start, inds, b)) if return_extra else ar
	
	# if higher-dimensional case, use `access_1d_strands` to make the array 2-d
	# then funnel to polar_reduce_2d and reshape the result to match original
	shape = np.array(ar.shape)
	new, true_start, inds, b = polar_reduce_2d(access_1d_strands(ar, axis), mod=mod, axis=1)
	new = recreate_from_1d_strands(new, axis, shape)

	# not necessary to reshape inds,b
	# however, probably want to make sure true_start has the right shape
	"""# make sure what follows is correct!
	shape = [*(s for i,s in enumerate(shape) if i!=axis), shape[axis]]
	shape[-1] = 1
	b = b.reshape(ar.shape[axis])
	true_start = recreate_from_1d_strands(ar.shape[axis])
	inds = recreate_from_1d_strands(inds, axis, shape)"""
	true_start = ...
	
	if return_extra: return (new, (true_start, inds, b))
	return new

def polar_reduction_2d(reducing_function,*args,fkwargs={},modulate=True,squeeze=False,start=True,**kwargs):
	"""Similar to '_polar_reduction', apply an aggregating function over
	the rows or columns of a 2-d array."""
	reduction, true_start, sort_inds, reindexer=polar_reduce_2d(*args,**kwargs)#modulate=modulate,
	
	try: mod=args[1]
	except IndexError: mod=kwargs['mod']
	
	result=reducing_function(reduction,axis=kwargs['axis'],keepdims=True,**fkwargs)
	if start: result=(true_start+result)%mod
	
	if squeeze: return result.squeeze()
	return result

def reduce_arrays_by_polar(polar_array,other,mod,axis=0,test=False):
	"""Run the polar_reduce algorithm on one array,
	and then rearrange another array `other` so that
	values between the two arrays correspond after
	the transformation. Both arrays should be 2-d
	and have the same shape."""
	if test: comp1=np.sort(np.array([polar_array.ravel(),other.ravel()]).T,axis=0)
	polar_reduced, true_start, sort_inds, b = polar_reduce_2d(polar_array,mod,axis)
	#print((polar_reduced+true_start)%mod)
	#print(sort_inds)
	other=np.take_along_axis(other,sort_inds,axis)
	other=complex_reindex_2d(other,b,axis)
	if test:
		comp2=np.sort(np.array([((polar_reduced+true_start)%mod).ravel(),other.ravel()]).T,axis=0)
		assert np.allclose(comp1,comp2)
	return (polar_reduced, other), true_start, sort_inds, b

def polar_mean_offset(angles,mod=360,handle_nan=True,ref=0,**kw):
	"""Get the average polar offset between a set of angles
	and some reference angle."""
	assert angles.size != 0, "no angles were passed to polar_mean_offset"
	meanfunc = np.nanmean if handle_nan else np.mean
	return polar_offset(
		polar_reduction(meanfunc, angles, mod, squeeze=True, **kw),
		ref,
		mod
	)

def polar_std_offset(angles,mod=360,handle_nan=True,**kw):
	"""Get the standard deviation in the polar offset
	between a set of angles and some reference angle."""
	assert angles.size != 0, "no angles were passed to polar_std_offset"
	stdfunc = np.nanstd if handle_nan else np.std
	return polar_reduction(stdfunc,angles,mod, start=False, squeeze=True, **kw)

def polar_stats(ar,ref,mod=360,axis=0):
	if ar.ndim==2:
		reduction = partial(polar_reduction_2d, axis=axis, mod=mod, squeeze=True)
	else:
		reduction = partial(polar_reduction,mod=mod)
	#reducer=lambda f,v: polar_offset(reduction(f,v),0,mod=mod)
	offset=polar_offset(ar,ref,mod=mod)
	"""
	check to make sure reducer behavior is correct
	"""
	#reducer=lambda f,v: f(v,axis=axis)
	reducer = lambda f,v,**k: polar_offset(reduction(f,v,**k),0,mod)
	mean=reducer(np.nanmean,offset)
	stdev=reducer(np.nanstd,offset,start=False)
	median=reducer(np.nanmedian,offset)
	
	abs_offset=np.abs(offset)
	mean_abs=reducer(np.nanmean,abs_offset)
	median_abs=reducer(np.median,abs_offset)
	max_abs=reducer(np.nanmax,abs_offset)
	
	return np.array((mean,median,stdev,mean_abs,median_abs,max_abs))

__all__ = ('polar_offset','rolling_polar_offset','polar_reduce','polar_reduction',
'_reduce_arrays_by_polar_1d','polar_reindex','polar_reindex_multi','polar_reduce_2d',
'polar_reduction_2d','reduce_arrays_by_polar','polar_mean_offset','polar_std_offset',
'polar_stats', 'polar_reduce_nd')

if __name__=='__main__':
	#from computation_functions import polar_reduce
	for i in range(1000):
		print_update(i)
		angles=np.random.randint(0,360,np.random.randint(20,200,2))
		check1=np.allclose(polar_reduce_2d(angles,360,1)[0],np.array([polar_reduce(row,360) for row in angles]))
		check2=np.allclose(polar_reduce_2d(angles,360,0)[0],np.array([polar_reduce(col,360) for col in angles.T]).T)
		if not (check1 and check2): raise RuntimeError('unequal arrays resulted')
		print_update('')