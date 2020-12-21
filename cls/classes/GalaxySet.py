import re

from common import getattrs, reparse, Struct, EmptyDict, MultiIterator
from common.collections import windowed_chunked
from common.decorators import add_doc_constants

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from prop.galaxy_file import *
from prop.simprop import *
from prop.sim_fn_gen import rsim_file_groups
from prop.ref_rel_data import ref_dict

from prop.asy_prop import INC_CUTOFFS

from plot.plotting_functions import makeax, stabilize_plot
from comp.polar_functions import polar_offset

from .Galaxy.Galaxy import Galaxy
from .Galaxy.galaxy_attribute_information import non_arrays, all_attributes

from sklearn.neighbors import KernelDensity

# from core.core import AttributeAbsent

# in frequency histograms involving OTHER, if there is only 1 element in OTHER,
# this variable sets the height of the bar pertaining to that element
HIST_SINGULAR_HEIGHT = .5
SCATTER_MAX_SIZE = 200
SCATTER_MIN_SIZE = 40

"""def iter_multi_dict_common(*dicts):
	# require all dictionaries to have same keys
	assert all(d1.keys()==d2.keys()
			   for d1,d2 in windowed_chunked(dicts, 2))
	
	keys = dicts[0].keys()
	
	return ((d[k] for d in dicts) for k in keys)"""

def iter_dict_keys(d, keys):
	return ((k,d[k]) for k in keys)

def KDE_fit(data, **kw):
	if data.ndim==1: data = data[:, np.newaxis]
	kde = KernelDensity(**kw).fit(data)
	return np.exp(kde.score_samples(data))

class _GalaxySet:
	"""
	Galaxy aggregation where order is arbitrary.
	
	Must be subclassed in order to function properly.
	"""
	
	def __init__(self):#, globals
# 		"""'globals' is the namespace this instance will use when
# 		evaluating expressions passed to it via `reparse`. If you do
# 		not provide a series of specific names, pass the global namespace
# 		from the module where this instance is initialized."""
		assert hasattr(self, '_sample'), \
			f"{type(self)} must have '_sample' attribute"
	
	def __getattr__(self, attr, *, non_arrays = non_arrays):
		"""Tell Galaxy to load data from disk (HDF5) corresponding
		to this instance's sample and the attributes passed."""
		if attr in non_arrays:
			return Galaxy.read_scalars(self._sample, attr)
		else:
			return Galaxy.read_arrays(self._sample, attr)
	
	def __getitem__(self, attr):
		"""defers to getattr(self, attr)"""
		return getattr(self, attr)
	
	def get(self, q, pattern=None):
		"""Given an expression (string) containing names of quantities
		defined on the Galaxy class and operations performed on these, replace
		references to these names to calls to getattr(self, attr) for each
		name, and return the result of eval() called on this expression."""
		if isinstance(q,str):
			expr = reparse(q, 'self', pattern=pattern, attributes=all_attributes)
			return eval(expr)#, self.globals
		
		return q
	
	@makeax
	def scatter(self, q1, q2, ax=None,
				time_map = False, asy_size=None, stabilize=False, **kw):
		"""plt.scatter of quantities q1, q2 (strings or plot data).
		
		If `time_map`, scatter points are colored by time.
		
		If `asy_size` is specified, it should be a string telling the attribute
		(or expression) to be used to set the size of the markers, or a
		container of sizes (it will be passed to `self.get`).
		
		Returns the result of the matplotlib plotting call."""
		if time_map:
			kw.setdefault('c',kw.setdefault('color',None))
			kw.pop('color')
			kw['c'] = np.arange(len(self))
			kw.setdefault('cmap', 'viridis')
		
		x, y = self.get(q1), self.get(q2)
		if asy_size:
			kw['s'] = Normalize()(self.get(asy_size)) * (
					SCATTER_MAX_SIZE -SCATTER_MIN_SIZE
				) + SCATTER_MIN_SIZE
		
		if stabilize:
			return stabilize_plot(y, X_array=x, scatter=True, ax=ax,
								  label=self.__class__.__name__, **kw)
		
		return ax.scatter(x, y, label=self.__class__.__name__, **kw)
	
	@makeax
	def plot(self, q1, q2, ax=None, ls='none', marker='o', stabilize=False, **kw):
		"""plt.plot of quantities q1, q2 (strings or plot data).
		Returns the result of the matplotlib plotting call."""
		x, y = self.get(q1), self.get(q2)
		
		if stabilize:
			return stabilize_plot(y, X_array=x, **kw, ax=ax,
								  label=self.__class__.__name__)
		return ax.plot(x, y, ls=ls, marker=marker,
					   label=self.__class__.__name__, **kw)
	
	def aplot(self, *a, **kw): self.plot(*a, stabilize=True, **kw)
	def ascatter(self, *a, **kw): self.scatter(*a, stabilize=True, **kw)
	
	@makeax
	def hist(self, q, ax=None, **kw):
		"""plt.hist of quantity q (string or plot data).
		Returns the result of the matplotlib plotting call."""
		data = self.get(q)
		n,b,p = ax.hist(data, **kw, label=self.__class__.__name__,
					   weights = np.full(len(self),1/len(self)))
		
		# not used at the moment
# 		kde = kw.get('kde')
# 		if kde is not None:
# 			ax.plot(data, KDE_fit(data), c=p[0].get_color())
		
		return n,b,p
	
	def __hash__(self): return id(self)
	
	def __eq__(self, other): return self is other
	
	def __len__(self): return len(self.sample)

def singleton_new(cls, *a, **kw):
	"""Singleton pattern: use this for the __new__ method of
	a class to be made a singleton."""
	if cls._instance is None:
		cls._instance = _GalaxySet.__new__(cls, *a, **kw)
	return cls._instance

class GalaxySetMaker(type):
	"""Metaclass responsible for defining the derivatives of _GalaxySet.
	
	Each _GalaxySet derivative has these qualities:
		* A class-level '_sample' attribute giving the names
		  of the galaxies in the associated sample. The sample may
		  be ANY kind of iterator; in particular, some derivates in
		  this module use `common.common.cls.MultiIterator` for the
		  _sample attribute.
		* The derived class name is guaranteed to be title-cased.
		* The derived class is a singleton class.
		* If the derived class' sample corresponds to real galaxies,
		  from a single sample as identified in `prop.galaxy_file`,
		  its name is the title-case of the all-uppercase name
		  used to identify the sample in `prop.galaxy_file`.
		* It is allowed to inherit from other classes in addition
		  to _GalaxySet and may define other optional class-level
		  attributes (such functionality has not yet been needed).
	
	Derived classes currently defined in this script:
		- Cluster
		- Atlas3d
		- Other
		- Reference (galaxies with reference data, see prop.ref_rel_data)
		- R_0_90 (Roediger)
		- R_90_0 (Roediger)
		- V{galaxy}_{inclination}_{dwa} for all defined combinations (Vollmer)
		- V{ig}, for each ig = an inclination group (low, med, high)
		- V{g}, for each g = a Vollmer galaxy model (4522, 4388, 4501, 4330)
	"""
	__new__ = singleton_new # GalaxySetMaker itself is a singleton
	
	@staticmethod
	def get_sim_params(clsname, num_params, _str=()):
		split = clsname.split('_')
		result = [int(''.join(filter(str.isdigit, p))) for p in split[-num_params:]]
		for i in _str: result[i] = str(result[i])
		return tuple(result)
	
	# define this as so so that __new__ can submit any arguments without issues
	def __init__(self, *a, **kw):
		...
	
	def __new__(cls, clsname, bases=(), attr=EmptyDict, sample = None):
# 		print('GalaxySetMaker.__new__: sample:',sample)
		
		if sample is None:
			if clsname.startswith('R'): # assign Roediger sample
				sample = rsim_file_groups[cls.get_sim_params(clsname, 2)]
			elif clsname.startswith('V'): # assign Vollmer sample
				sample = vsim_file_groups[cls.get_sim_params(clsname, 3, (0,))[:-1]]
			else: #
				sample = eval(clsname.upper())
		
		bases = (_GalaxySet, *bases)
		attr = {'_sample':sample, '_instance':None, '__new__':singleton_new, **attr}
		
		return super().__new__(cls, clsname, bases, attr)

GalaxySets = Struct()
for clsname in ('Cluster','Atlas3d','Other','R_0_90','R_90_0'):
	GalaxySets[clsname] = GalaxySetMaker(clsname)()

GalaxySets['Reference'] = GalaxySetMaker('Reference', sample=tuple(ref_dict.keys()))()

VollmerSet = {}
VollmerISet = {f'V{ig}':GalaxySetMaker(f'V{ig}', sample=MultiIterator())()
			   for ig in INC_CUTOFFS}
VollmerGSet = {f'V{g}':GalaxySetMaker(f'V{g}', sample=MultiIterator())()
			   for g in vollmer_groups}

for g,dwa in iter_dict_keys(vollmer_disk_wind_inclinations, vollmer_groups):
	group = vollmer_groups[g]
	VollmerGSet[f'V{g}']._sample.extend(group.values())
	for i,sample in group.items():
		clsname = f'V{g}_{i}_{dwa}'
		cls = GalaxySetMaker(clsname)()
		GalaxySets[clsname] = VollmerSet[(g,i)] = cls
		VollmerISet[f'V{get_inc_group(i)}']._sample.add_target(cls._sample)

del g, dwa, group, i, sample
for g in vollmer_groups: GalaxySets[f'V{g}'] = VollmerGSet[f'V{g}']
for ig in INC_CUTOFFS: GalaxySets[f'V{ig}'] = VollmerISet[f'V{ig}']

@add_doc_constants(globals(), 'HIST_SINGULAR_HEIGHT')
def adjust_singular_bar_height(hist, height = HIST_SINGULAR_HEIGHT):
	"""Given hist = the return values of plt.hist, if there is only one element
	in the set used to make the histogram, adjust the corresponding bar in the
	histogram to have a new height teh fraction of the original height given by
	the `height` parameter (default HIST_SINGULAR_HEIGHT).
	"""
	if np.sum(hist[0])==1: # only one element
		for p in hist[2]: # plt.hist returns counts, bins, patches
			p.set_height(p.get_height()*HIST_SINGULAR_HEIGHT)


REAL_GROUP = getattrs(GalaxySets, 'Cluster', 'Atlas3d', 'Other')
RSIM_GROUP = getattrs(GalaxySets, 'R_0_90', 'R_90_0')
VSIM_I_GROUP = tuple(VollmerISet.values())
VSIM_G_GROUP = tuple(VollmerGSet.values())

__all__ = ('GalaxySetMaker', 'GalaxySets', 'REAL_GROUP', 'RSIM_GROUP',
'VSIM_I_GROUP', 'VSIM_G_GROUP', 'VollmerISet', 'VollmerGSet', 'vollmer_groups',
'VollmerSet' )