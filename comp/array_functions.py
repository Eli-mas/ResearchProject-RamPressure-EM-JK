from common.collections import consume
from common.arrays import make_object_array, minmax
import numpy as np
from asy_io.asy_io import merge_dicts,lprint
from skimage.util import view_as_windows as vw_
nan=np.nan
from itertools import chain, product, takewhile
import operator
import heapq







allslice=np.s_[:]

def nd_padder(a,axis):
	shape=np.array(a.shape)
	axis=np.atleast_1d(axis)
	pad_axis=np.zeros([a.ndim,2],int)
	pad_axis[axis,1]=shape[axis]-1
	p=np.pad(a,pad_axis,'wrap')
	return p

def complex_reindex_2d(a, roll_indices, axis, err_raise=True):
	"""adapted from https://stackoverflow.com/a/51613442"""
	roll_indices=np.atleast_1d(roll_indices)
	if roll_indices.ndim>1:
		raise ValueError('too many dimensions for `roll_indices`: must be 1-d')
	if err_raise and (roll_indices.shape[0]!=a.shape[1-axis]):
		raise ValueError('`roll_indices` has length incompatible with the array\'s other axis')
	n=a.shape[axis]
	w=np.ones(a.ndim)
	w[axis]=n
	s=(np.arange(len(roll_indices)),(roll_indices-n)%n,0)
	if axis==1: s=s+(allslice,)
	elif axis==0: s=(allslice,)+s
	return vw_(nd_padder(a,axis),w)[s]

def take_along_axis_by_grid(array,indices,axis):
	shape=np.array(array.shape)
	reduced_shape=list(shape)
	reduced_shape.pop(axis)
	if not np.array_equal(indices.shape,reduced_shape):
		raise ValueError(
		"`take_along_axis_by_grid`: shape of 'indices' must have shape equal to"
		" shape of the array with the specified 'axis' removed"
		)
	grid_inds=np.meshgrid(*(range(i) for i in reduced_shape),indexing='ij')
	indexer=[None for i in range(array.ndim)]
	reduced_indices=(i for i in range(array.ndim) if i!=axis)
	for pos,gi in zip(reduced_indices,grid_inds):
		indexer[pos]=gi
	indexer[axis]=indices
	return array[tuple(indexer)]

def nanrep_filter(ar,axis,pr=False):
	"""
	this function supersedes 'nanrep_filter_2d'
		see array_functions_4_19_19 for that version
	
	the prior function worked only on 2d arrays
	this version should generalize to any dimensionality,
		though testing is not yet conducted for higher-dimensional cases
	"""
	nancheck=np.isnan(ar)
	if not np.any(nancheck): return ar
	
	numloc=np.argmax(nancheck,axis=axis)-1
	
	values=take_along_axis_by_grid(ar,numloc,axis)
	
	rep=np.broadcast_to(
		np.expand_dims(values,axis=axis),
		ar.shape
	)
	
	new=np.copy(ar)
	new[nancheck]=rep[nancheck]
	
	if pr: lprint('nancheck',nancheck,'numloc',numloc,'values',values,'rep',rep,'-- -- --')
	
	return new

def reindex(ar, starts_at, *, axis=0, extend=None, **kw):#,axis=-1
	"""Equivalent to np.roll(ar, -starts_at, ...)"""
	ar=np.roll(ar,-np.atleast_1d(starts_at),axis=axis,**kw)
	if extend is not None: return np.concatenate((ar,ar[:extend]), axis=axis)
	return ar

def in_sorted(array, value):
	"""
	this assumes that 'array' is sorted and increasing
	"""
	# only one element and matching?
	if len(array.shape)==0: return array==value
	# target value out of bounds?
	if (value<array[0]) or (value>array[-1]): return False
	# delegate to np.searchsorted, implementing binary search
	# using keyword 'right' means that the value 1 behind the
	# returned index has to be equal to the target value
	# if the value is located in the array
	return (array[np.searchsorted(array,value,'right')-1]==value)

def get_regions(ar, threshold):#,period=0
	"""
	find subarrays in a 1d sorted array that are related,
	i.e. where the differences between consecutive elements
	of the subarray do not surpass `threshold`
	"""
	return np.split(ar, np.where(np.ediff1d(ar) > threshold)[0]+1)

def get_regions_from_other(ar, threshold, *arrays):
	s = np.where(np.ediff1d(ar) > threshold)[0] + 1
	return tuple(np.split(a,s) for a in ((ar,) + arrays) )

def get_region_inds(bool_ar, threshold = 1):
	w = np.where(bool_ar)[0]
	return get_regions(w, threshold)

def find_largest_bool_ind_region(*a, **kw):
	return max(get_region_inds(*a, **kw), key=len)

def nanfilter(ar, dtype=None):
	return np.array(ar[~np.isnan(ar)],dtype=dtype)

def take_by_condition(condition,*arrays):
	return tuple(np.atleast_1d(ar[condition]) for ar in arrays)

def ensure_elementwise(iterable):
	"""
	ensure that an iterable can handle vectorized operations
	if the passed argument does not support iteration, error is raised
	"""
	iter(iterable)
	try:
		# if d is a numpy array, pandas Series, or similar,
		# this raises an exception
		bool(iterable)
		# if no exception raised, d is a collection that is not expected to provide
		# element-wise operations, so convert to array
		return np.array(iterable)
	except:
		return iterable

def broadcast_by(array, source, axis):
	"""
	Given an input array `array`, reshape and broadcast `array`
	so that it has a new shape equal to the shape of `source`.
	The `axis` argument specifies the axis or axes along `source`
	along which `array` is to be broadcasted. Therefore,
	the shape of `source` along the specified `axis`
	must correspond to the shape of `array`.

	This function does not handle transposing--thus another requirement
	is that `axis` be ordered, e.g. monotonically increasing,
	if it is array-like.

	The `axis` argument should also have no duplicate values, if it is
	array-like.


	:param array: input array to be broadcasted; elementwise array-like
	:param source: source array; elementwise array-like
	:param axis: the axis along the source array into which the input array is to be broadcasted; int or array-like
	:return: broadcasted input array
	"""
	#print('source:',source)
	shape = np.array(np.atleast_1d(source).shape)
	axis = np.atleast_1d(axis)
	axis[axis<0] += len(shape)
	
	if np.any(axis[:-1]>=axis[1:]):
		raise ValueError(f"the `axis` argument must be ordered (monotnically increasing): axis = {axis}")

	if not np.array_equal(array.shape, shape[axis]):
		raise ValueError(
			f"the source array has shape {shape} --> "
			f"{shape[axis]} along axis {axis}, "
			f"while the input array has shape {array.shape}"
		)

	new_array_shape = np.ones(len(shape),dtype=int)
	new_array_shape[axis] = array.shape
	array = np.broadcast_to(array.reshape(new_array_shape), shape)
	return array

def find_minimal_keys(keys, values, apply=None, sort_keys=True, verbose=False):
	"""Given iterables of keys and values, find all the keys
	that correspond to a minimal value among all keys."""
	if verbose:
		print('keys, values:')
		for k,v in zip(keys,values): print(f'\t{k}, {v}')
	if apply: values = [apply(v) for v in values]
	data = [*zip(values,keys)]
	
	# data becomes a heap
	heapq.heapify(data)
	
	#grab smallest value/corresponding key
	minimum,k = heapq.heappop(data)
	minima = [k]
	
	# pop keys/values from the heap while values == minimum; store keys in minima
	minima[1:] = map(lambda s: s[1], takewhile(lambda s: s[0]==minimum, (heapq.heappop(data) for k in keys)))
	#the result is all keys whose corresponding values are equal to 'minimum'
	
	if verbose:
		res = sorted(minima) if sort_keys else list(minima)
		print(res)
		return res
	return sorted(minima) if sort_keys else list(minima)

def find_closest_keys(keys, values, target_value, **k):
	"""Given a set of keys and values, find the keys whose
	corresponding values are closest to the target value."""
	return find_minimal_keys(keys, values, apply=lambda v: np.abs(v-target_value))

def bin_keys(keys, values, bins, apply=None, sort_keys=True, verbose=False, to_string=False):
	if apply: values = [apply(v) for v in values]
	data = [*zip(values,keys)]
	
	binned_keys=[[] for _ in range(len(bins)+1)]
	data.sort()
	data = np.array(data,dtype=object)
	divisions = [0, *np.searchsorted(data[:,0],bins,'right'), len(data)]
	if verbose: print(f'bin_keys:\ndata: {data}\nbins: {bins}\ndivisions:{divisions}')
	for bin, (i0,i1) in zip(binned_keys, zip(divisions[:-1],divisions[1:])):
		bin[:]=data[i0:i1,1]
	
	if sort_keys:
		for b in binned_keys: b.sort()
	
	if to_string:
		binned_keys = [f"({', '.join(b)})" if b else '--' for b in binned_keys]
	
	return binned_keys

def ensure_columnwise(x):
	if x.ndim==1:
		x = x.reshape(len(x),1)
	elif x.ndim>2:
		raise ValueError(
			"'x' argument to 'ensure_columnwise' cannot have more than 2 dimensions"
		)
	return x

__all__ = ('complex_reindex_2d','nanrep_filter','reindex','in_sorted','get_regions',
'get_regions_from_other','get_region_inds','find_largest_bool_ind_region',
'take_by_condition','minmax','nanfilter','ensure_elementwise','make_object_array',
'broadcast_by','find_minimal_keys','find_closest_keys','bin_keys','ensure_columnwise',
)










if __name__=='__main__':
	# a = np.random.randint(4,size=[2, 3])
	# source = np.random.randint(20, size=(5, a.shape[0], 10, a.shape[1]))
	# b = broadcast_by(a,source,[1,3])
	# print(b[0:1,:,0:1,:])

	size=10
	a = np.arange(size)
	source = np.random.randint(5,size=[size,3,4])
	b = broadcast_by(a,source,0)

	print('a:',a,sep='\n')
	print('source.shape:',source.shape)
	print('broadcast_by(a,source,...).shape:',b.shape)
	print('broadcasted array (a):',b,sep='\n')
	print(b[:,:1,:1])
