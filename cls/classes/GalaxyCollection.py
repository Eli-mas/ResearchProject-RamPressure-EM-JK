import operator #, itertools
from functools import partial

from common.common.cls import Proxy

import numpy as np
from matplotlib import pyplot as plt

from prop import galaxy_file
from prop.asy_prop import *
from prop.simprop import vollmer_data

from asy_io.asy_paths import COLLECTION_NPY_PATH
from asy_io.asy_io import load_array, save_array
from plot.plotting_functions import makeax

from comp import polar_functions
from comp.array_functions import make_object_array

from cls.classes.Galaxy.Galaxy import Galaxy
from cls.classes.Galaxy.galaxy_attribute_information import non_arrays

from cls import plotters

from cls.adc_prep import computable_quantities, nonmembers
import cls.analyze

from core.core import AttributeAbsent

class _GalaxyCollection:
	
	operators={
		'+' :	operator.add, 
		'-' :	operator.sub, 
		'*' :	operator.mul, 
		'/' :	operator.truediv, 
		'**':	operator.pow
	}
	
	def __init__(self, galset, coll_type, regenerate=False, _zip = True, **kw):
		from prop.galaxy_file import galaxy_association_samples, recognized_samples
		
		#print('_GalaxyCollection: galaxies:', galset, sep='\n')
		galaxies=np.array(galset)
		self.galaxies=galaxies
		self.count=len(galaxies)
		self.instances=[Galaxy(g, **kw) for g in galaxies]#, start_empty=True
		self.galindex={g:i for i, g in enumerate(galaxies)}
		#print('coll_type', coll_type)
		if coll_type in recognized_samples: self.directory=coll_type
		else: self.directory=galaxy_association_samples[coll_type]
		self.compute=kw.setdefault('compute', False)
		self.regenerate=regenerate
		self.P_=Proxy(self.instances)
		#self.NULL=NULL
		self.__INITIAL_ARGS__=(galset, coll_type)
		self.__INITIAL_KWARGS__=kw
		#print self.__INITIAL_ARGS__
		#print self.__INITIAL_KWARGS__
		self._zip = _zip

	def get_instance(self, name):
		return self.instances[self.galindex[name]]

	def __len__(self): return len(self.galaxies)
	
	def array(self, attr): return np.atleast_1d(getattr(self, attr))
	
	def __getattr__(self, attr): return self.get_attr(attr)
	
	def get(self, attr, fill=True):
		try: return getattr(self,attr)
		except AttributeError: return np.full(len(self), nan) if fill else nan
	
	def get_attr(self, attr, _zip=None):
		# print(f'get_attr ({attr}): compute={self.compute}, regenerate={self.regenerate}, in computable_quantities: {attr in computable_quantities}')
		if not isinstance(attr, str): return attr
# 		if attr in nonmembers: return
		if (not (self.compute or self.regenerate)) and (attr in non_arrays):
			#print(self.directory+': loading %s from memory'%attr)
			try:
				result = self._load_array(attr)
				if np.all(np.isnan(result)):
					raise AttributeError(f"nan array loaded: {attr} appears to be invalid attribute for {self.directory}")
				return result
			except IOError: pass

		result=self._getmissing(attr, _zip=_zip)
		if result is None:
			raise ValueError(
				"_GalaxyCollection.__getattr__: '%s' not obtained"%attr
				#+the stack is as follows:
				#+print_stack(inspect.stack(), item=3)
			)
		if attr in non_arrays:
			if np.all(np.isnan(result)):
				raise AttributeError(f"attempted to save nan array: {attr} appears to be invalid attribute for {self.directory}")
			return self._save_array(attr, result)
		return result
		
	def get_from_instance(self, g, attr, default=nan):
		try: return getattr(g, attr)
		except: return default
	
	def _getmissing(self, attr, _zip=None):
		try:
			if _zip is None: _zip = self._zip
			#print(f'_getmissing: _zip: {_zip}')
			#print(f'{self.directory}::_getmissing : attr={attr}, _zip={_zip}')
			genfunc=self.get_from_instance if _zip else getattr
			ret=[genfunc(g, attr) for g in self.instances]
			try: return np.array(ret)
			except ValueError: return ret
			"""return arrays by default, as types are expected to be homogenous"""
		except AttributeAbsent:
			raise
		except Exception as e:
			print(f'error in retrieving attribute {attr} for {self}:',repr(e),sep='\n')
			raise AttributeError(
				"`_GalaxyCollection`:'%s' the attribute '%s' could not be loaded"
				" for the collected galaxies"%(self.directory, attr)
			)
			#traceback.print_exc()
			return NULL
	
	def getattrs(self, *attrs, asarray=True):#, **k
		# _zip=k.setdefault('_zip', None)
		# #print 'getattrs: _zip=%s'%_zip
		# res=tuple((self._getmissing(v, _zip=_zip) if isinstance(v, str) else v) for v in attrs)
		res = tuple(getattr(self,attr) if isinstance(attr,str) else attr for attr in attrs)
		if asarray:
			try: return np.array(res)
			except ValueError: return make_object_array(res)
		else:
			return res
	
	def __getitem__(self, gal):
		if isinstance(gal, int): return self.instances[gal]
		return self.instances[self.galindex[gal]]
	
	#def clear_loaded_data(self):
	#	for g in self.instances: g.clear_loaded_data()
	
	def __delattr__(self, attr):
		for g in self.instances: delattr(g, attr)

	@staticmethod
	def format_collection_path(directory, artype):
		return COLLECTION_NPY_PATH.format(directory=directory, artype=artype)
	
	def _save_array(self, attr, data):
		print(
			f'{self}: _save_array: saving attribute "{attr}" to {self.format_collection_path(self.directory, attr)}',
			str(data).replace('\n','\n\t'),sep='\n'
			)
		loaded=save_array(self.format_collection_path(self.directory, attr), data)
		return loaded
	
	def _load_array(self, attr):
		loaded=load_array(self.format_collection_path(self.directory, attr))
		return loaded
	
	"""def remove(self, attr, indices=None):
		'''
		A more conservative form of `clear_loaded_data`:
		this version allows for deleting of particular attributes from particular Galaxy instances, 
		rather than removing all data from all galaxies
		'''
		attr=listify2(attr)
		if indices is None:
			for attribute in attr:
				delattr(self, attribute)
		else:
			indices=np.array([i if isinstance(i, int) else self.galindex[i] for i in listify2(indices)])
			for index in indices:
				Ginstance=self.instances[index]
				for attribute in attr:
					delattr(Ginstance, attribute)"""
	
	@makeax
	def plotter(self, pfunc, attr1, attr2, *a, ax=None, alter_x=None, alter_y = None, **kw):
		x, y=self.getattrs(attr1, attr2, _zip=True)
		if alter_x:
			if isinstance(alter_x,str): alter_x = getattr(self,alter_x)
			x = alter_x(x)
		if alter_y:
			if isinstance(alter_y,str): alter_y = getattr(self,alter_y)
			y = alter_y(y)
		getattr(ax, pfunc)(x, y, *a, **kw)
	
	def plot(self, *a, **k): self.plotter('plot', *a, **k)
	def scatter(self, *a, **k): self.plotter('scatter', *a, **k)
	
	def apply(self, func, *a, **k):
		if callable(func): return np.array(tuple(func(g, *a, **k) for g in self.instances))
	
	#def apply_attr_multi(self, func)
	
	
	def apply_attr(self, func, *attrs, **kw):
		if isinstance(func, str):
			for o in _GalaxyCollection.operators:
				if o in func: break
			attrs=func.split(o)
			func=_GalaxyCollection.operators[o]
			kw={}
		
		asets=self.getattrs(*attrs, _zip=True)
		
		return np.array([func(*v, **kw) for v in zip(*asets)])
	
	def compute_all(self):
		raise NotImplementedError('`compute_all` not yet configured for this class')
		"""_=parmap.map(compute_all, self.galaxies)
		
		self.__init__(**merge_dicts(self.__INITIAL_KWARGS__, compute=True))#*self.__INITIAL_ARGS__, 
		for attr in computable_quantities:
			self._save_array(self, attr)"""
	
	def m2_weights_range_plot(self):
		m2w=np.array([g.m2_weights for g in self.instances if g.inclination<=high_i_threshold])
		m2w_low, m2w_high=m2w.min(axis=1), m2w.max(axis=1)
		g_low=[g for g in self.instances if g.inclination<=high_i_threshold]
		for i, (g, l, h) in enumerate(zip(g_low, m2w_low, m2w_high)):
			plt.bar(i, h-l, bottom=l)
			plt.text(i, h, g.filename, va='bottom', ha='center', fontsize=14)
		
		plt.gca().tick_params(labelright=True, right=True)
		print(np.column_stack(g_low, m2w_low.round(2), m2w_high.round(2)))
		plt.show()
	
	
class ClusterCollection(_GalaxyCollection):
	def __init__(self, **k):
		_GalaxyCollection.__init__(self, galaxy_file.cluster, 'cluster', **k)

class ATLAS3DCollection(_GalaxyCollection):
	def __init__(self, **k):
		_GalaxyCollection.__init__(self, galaxy_file.ATLAS3D, 'ATLAS3D', **k)

class OTHERCollection(_GalaxyCollection):
	def __init__(self, **k):
		_GalaxyCollection.__init__(self, galaxy_file.OTHER, 'OTHER', **k)



'''def itercall(func, args=None, iterargs=None, iterkwargs=None, **kw):
	"""
	functionalities:
		(
		func(	*constant_args_across_calls, *variable_args_between_calls, 
				*constant_kwargs_across_calls, *variable_kwargs_between_calls
			)
		for variable_arg_iterable, varaible_kwarg_iterable in zip (varargs, varkwargs))
	"""
	
	try: iterkw=kw.pop('iterkw')
	except KeyError: iterkw=None
	
	func()'''

class _GalaxySeries(_GalaxyCollection):
	
	for _module in cls.analyze.__all__:
		if _module.startswith('galaxyseries'):
			_m = getattr(cls.analyze,_module)
			for func in _m.__all__:
				exec (f'{func}=getattr(_m, func)')
	
	for _module in plotters.__all__:
		if _module.startswith('galaxyseries'):
			_m = getattr(plotters,_module)
			for func in _m.__all__:
				#setattr(_GalaxySeries, func, getattr(_m, func))
				exec (f'{func}=getattr(_m, func)')
	
	
	def __iter__(self): return iter(self.instances)
	
	def sig_indices(self, m2=False):
		"""
		Get the indices, corresponding to the **instances** array, 
		when the asymmetry is significant. If m2=False, this considers
		only the m=1 ER, FR ratios; if True, it also considers
		ER/m2ER & FR-m2FR
		"""
		cond=(self.ER>=sig_ext_cutoff) & (self.FR>=sig_flux_cutoff)
		if m2 and self.inc<90:
			cond &= ((self.ER/self.m2ER>sig_m1_m2_ext_cutoff) & (self.FR-self.m2FR>sig_m1_m2_flux_cutoff))
		return np.where(cond)[0]
	
	@property
	def sig_indices_(self):
		"""Equivalent to calling sig_indices(m2=True)"""
		return self.sig_indices(m2=True)
	
	def __init__(self, galset, coll_type, time_index=None, rot=None, **kw):
		_GalaxyCollection.__init__(self, galset, coll_type, _zip = False, **kw)
# 		print('{self}: self.sig_indices_:',self.sig_indices_)
		self.ind_sig=self.sig_indices(m2=True)#int(self.sig_indices_[0])
		for c in ('xpix', 'ypix', 'PA'): setattr(self, c, getattr(self.instances[0], c))
		if rot is None:
			raise ValueError("specify 'rot' parameter to describe rotation")
		self.rot = -1 if rot==-1 else 1
	
	def time(self, t):
		"""
		get the Galaxy instance closest to the argument t

		:param t: a time in Myr
		:return: the instance at tindex(t)
		"""
		return self.instances[self.tindex(t)]

	def tindex(self, t):
		"""
		get the index of the time contained in the series' time array
		that is closest to the argument t

		:param t: a time in Myr
		:return: index of the closest time in the series' time array
		"""
		return np.argmin(np.abs(self.time_index-t))
	
	def __str__(self): return f'series<{self.directory}>'
	def __repr__(self): return f'series<{self.directory}>'

	def max(self, *a):
		return np.atleast_2d(self.getattrs(*a)).max(axis=1).squeeze()
	
	"""def mean(self,*attrs,m2=True,mask=True):
		
		if mask:
			mask = self.mask_full if m2 else self.mask_m1_only
			mask = broadcast_by(mask, self, 0)
			
			return ma.masked_array"""

	"""def __numpy_apply(self, func_name, attr, start=None, end=None, inds=None):
		#assumed to apply only to 1-d arrays
		data=getattr(self, attr)
		if inds is not None:
			data=data[inds]
		else:
			if end is not None: data=data[:end]
			if start is not None: data=data[start:]
		return getattr(np, func_name)(data)
	
	def average(self, *a, **k): return self.__numpy_apply('average', *a, **k)
	def std(self, *a, **k): return self.__numpy_apply('std', *a, **k)
	
	def sig_mean(self, attr): return self.average(attr, inds=self.sig_indices_)
	def sig_std(self, attr): return self.average(attr, inds=self.sig_indices_)"""
	
	@property
	def PA_diameters(self): return self.get_PA_radii().sum(axis=1)
	@property
	def PA_diameters_norm(self):
		d = self.get_PA_radii().sum(axis=1)
		d /= d[0]
		return d
	
	@property
	def gas_content(self): return self.get_contained_gas()
	@property
	def gas_content_norm(self):
		g = self.get_contained_gas()
		g /= g[0]
		return g
	
	def get_PA_radii(self):
		"""get the radii at PA, PA+180 for all galaxies in the series"""
		pos=int(self.PA*deg_to_index)
		return np.array([
			Ginstance.extentlist_graph[[pos, (pos+al)%a2l], 1]
			for Ginstance in self.instances
		])
	
	def get_contained_gas(self):
		zdata_original=self.instances[0].zdata
		#coors=np.where(zdata_original!=0)
		
		"""
		#the unit of the pixels is M_Sol/pc^2
		#each pixel has side length D = 1kpc * self.psa = 1000pc * self.psa
		#Each pixel is (D pc x D pc) in area
		#so multiply by D^2 to get the total gas mass
		
		IMPORTANT
		the values of self.psa for Vollmer files are confirmed
		confirm them for Roediger files
		"""
		#return adjust_mass_for_pixel_area * np.array([
		#	fsum(Ginstance.zdata[coors[0], coors[1]])
		#	for Ginstance in self.instances
		#])

		adjust_mass_for_pixel_area=(1000 * self.psa)**2

		return adjust_mass_for_pixel_area * np.sum(
			self.zdata*np.expand_dims(zdata_original!=0, 0), 
			axis=(1, 2)
		)
	
	@makeax
	def WAline(self, ax=None): return hline(self.WA, ax=ax)
	@makeax
	def antiWAline(self, ax=None): return hline(self.antiWA, ax=ax)
	
	@property
	def peak_angle(self):
		return np.argmax(self.extentratio[:, :, 1], axis=1)*index_to_deg+180
	
	@property
	def sig_ind_linspace(self):
		return np.linspace(self.ind_sig[0], self.count-1, 6).astype(int)
	#
	
	def reverse(self,attr):
		if isinstance(attr,str): attr = getattr(self,attr)
		return self.WA+(self.WA-attr)

	@property
	def ext_windside_means(self): return self.time_evolving_side_mean(self.WA)
	@property
	def ext_leeside_means(self): return self.time_evolving_side_mean(self.antiWA)
	
	def wind_vector_lateral_proportion(self):
		wv2 = np.array(self.wind_vector)**2
		return np.sqrt(wv2[:2].sum()) / np.sqrt(wv2.sum())
	
	def wind_vector_lateral_angle(self):
		# dot product of wind vector with z unit vector is equivalent
		# to z-component of wind vector
		# z-unit vector has magnitude=1
		# so angle = cos^-1((w <dot> i_z)/(||w|| ||z||)) = cos^-1(w_z/||w||)
		a = np.arccos(self.wind_vector[2] / np.sqrt((self.wind_vector**2).sum()))
		return 90-np.abs(polar_functions.polar_offset(np.round(rad_to_deg*a,1),90))
	
	@property
	def time_diff(self):
		return (self.time_index[:-1] + self.time_index[1:]) / 2

class RoedigerSeries(_GalaxySeries):
	
	WA = 90
	antiWA = 270
	PA = nan
	
	def __init__(self, sim_id, **kw):
		from prop import sim_fn_gen
		from prop.simprop import roediger_disk_wind_inclinations, roediger_wind_vectors
		
		#self.WA=90
		#self.antiWA=270
		if sim_id=='x':
			self.inc=0
			self.aplot_band=np.array((-40, 40))
			self.rotation=-1 # clockwise
			rot = -1
			self.psa = .1
			#self.PA = nan
		elif sim_id=='y':
			self.inc=90
			self.aplot_band=np.array((-30, 30))
			self.rotation=0 # edge-on
			rot = 0
			self.psa = .2
			#self.PA = nan
		else:
			raise ValueError("sim_id '%s' does not exist for the roediger galaxies"%sim_id)

		self.aplot_ylim = self.WA + self.aplot_band
		self.title = 'r%i' % self.inc

		self.sim_type = 'hydro'
		self.DWA = roediger_disk_wind_inclinations[self.title]
		self.wind_vector = np.array(roediger_wind_vectors[self.title])

		_GalaxySeries.__init__(self, sorted(getattr(sim_fn_gen,'sim_fn_'+sim_id)), self.title, rot=rot, **kw)
		self.time_index=getattr(sim_fn_gen,'simtime_'+sim_id)
		#self.=np.average(view_as_windows(), axis=1)
		self.galaxy=None
		
	def __getitem__(self, frame):
		try: return self.instances[frame-208]
		except TypeError: return self.instances[int(frame[1:])-208]
	
	#

"""vsim_aplot_bands=np.array([
	[-50, 50], 
	[-110, 40], 
	[40, 180]
])"""

class VollmerSeries(_GalaxySeries):
	
	#def add_RP(self, function): #FIGURE OUT HOW TO USE THIS AS A DECORATOR
	#def plotter(self, function): #DECORATOR TO ENABLE SELECTIVE CALLING OF <plt.show>
	
	def __init__(self, galaxy, inclination, *, time_index=None, **kw):
		from plot.voll_rp_profile import voll_time_array, vollmer_p_ram_df
		from prop.simprop import vollmer_wind_angles, vollmer_disk_wind_inclinations, vollmer_PA
		from prop.galaxy_file import vollmer_groups
		
		self.title='v{gal}_{inc}_{dw}'.format(gal=galaxy, inc=inclination, dw=vollmer_disk_wind_inclinations[galaxy])
		self.inc=inclination
		self.galaxy = galaxy
# 		print('VollmerSeries.__init__: about to initialize _GalaxySeries')
		self.rotation = 1 if inclination!=90 else 0 # counterclockwise
		_GalaxySeries.__init__(self, vollmer_groups[galaxy][inclination], self.title, rot = self.rotation, **kw)
		self.psa = self.instances[0].pix_scale_arcsec

		self.WA=vollmer_wind_angles[galaxy][inclination]
		self.DWA = vollmer_disk_wind_inclinations[galaxy]
		self.wind_vector = np.round(vollmer_data[galaxy][inclination],2)
		peak,t_HW = vollmer_p_ram_df[galaxy]
		self.peak = peak/t_HW**2
		self.t_HW = t_HW
		self.PA = vollmer_PA[galaxy]

		self.sim_type = 'sticky n-body'
		
		attributes = ['EA', 'FA', 'EA_trig', 'FA_trig', 'tail_A', 'HTA', 'qEA', 'qFA']
		self.EWA = dict(zip(attributes, Galaxy(galaxy).getattrs(*attributes)))
		self.antiWA=(self.WA+180)%360
		# self.aplot_ylim=self.WA+vsim_aplot_bands[vindex]
		
		self.time_index = voll_time_array(galaxy) #voll_time(galaxy, time_index=time_index)
		#self.time_diff=np.average(view_as_windows(self.time_index, 2), axis=1)
		
# 		if galaxy=='4522':
# 			self.rp_trace_zshow_indices=np.linspace(-270, 280, 6, dtype=int)
		self.rp_trace_zshow_indices=np.linspace(self.time_index[len(self.time_index)//4], self.time_index[-1], 6, dtype=int)
		
		self.DWA = vollmer_disk_wind_inclinations[galaxy]
	
	def __getitem__(self, frame):
		try: return self.instances[frame]
		except TypeError: return self.instances[int(frame[1:])]
	#

R0Series=partial(RoedigerSeries, sim_id='x')
R90Series=partial(RoedigerSeries, sim_id='y')
"""
V4522_80_30Series=partial(VollmerSeries, galaxy='4522', inclination=80)
V4522_60_30Series=partial(VollmerSeries, galaxy='4522', inclination=60)
V4522_30_30Series=partial(VollmerSeries, galaxy='4522', inclination=30)
"""

def VollmerSeriesCollection(source = vollmer_data, **k):
	if isinstance(source, str):
		return tuple(
			partial(VollmerSeries, galaxy=source, inclination=i, **k)
			for i in sorted(vollmer_data[source])
		)
	return tuple(
		partial(VollmerSeries, galaxy=g, inclination=i, **k)
		for g in source for i in sorted(vollmer_data[g])
	)



class SeriesProxy(Proxy):
# 	def __init__(self, *targets, **kw):
# 		super().__init__(*targets, **kw)
# 		self.simprop = [(s.directory, s.inc, s.DWA, s.galaxy) for s in targets]
	
	def angles_with_min_offset_over_sig_asy(self, **kw):
		angles = np.array(('EA', 'FA', 'EA_trig', 'FA_trig', 'HTA', 'qEA', 'qFA'))
		deltas = self.sig_delta_mean(*angles, **kw)
		amin = np.argmin(deltas, axis=1)
		optimizing_angles = angles[amin]
		optimized_offsets = deltas[range(deltas.shape[0]), amin]
		return optimizing_angles, optimized_offsets

	def most_accurate_angles_over_sig_asy(self, **kw):
		angles = np.array(('EA', 'FA', 'EA_trig', 'FA_trig', 'HTA', 'qEA', 'qFA'))
		accuracies = self.angle_sig_accuracy(*angles, **kw)
		# AXES: 0 series, 1 angle type
		
		amax = np.argmax(accuracies, axis=1) # AXES: 0 series
		# indices of highest accuracies for each simulation
		
		optimized_accuracies = accuracies[range(accuracies.shape[0]), amax] # AXES: 0 series
		# highest accuracies for each simulation
		
		#optimizing_angles = angles[amax] # AXES: 0 series
		## angles resulting in highest accuracies for each simulation
		optimizing_angles = np.array([
			', '.join(angles[np.where(row==opt)[0]]) if not np.isclose(opt,0) else '--'
			for i,(opt,row) in enumerate(zip(optimized_accuracies,accuracies))
		])
		#iterate over series indices, highest accuracies
			# get all angles that match this accuracy for this series
		
		return optimizing_angles, optimized_accuracies
	
	def remove(self,galaxy,inc):
		size=len(self.targets)
		#self.targets[:] = (s for s in self.targets if not (s.galaxy==galaxy and s.inc==inc))
		self.targets = self.targets[[not (s.galaxy==galaxy and s.inc==inc) for s in self.targets]]
		if len(self.targets)==size:
			raise ValueError('SeriesProxy.remove: no targets were removed')
	
	def index(self, directory):
		return [
			i for i in range(len(self.targets))
			if self.targets[i].directory==directory
		][0]
	
	def groupby(self,results,how):
		if how=='inc':
			bins = np.array([30,60])
			key = lambda s: np.digitize(s.inc,bins,right=True)
		elif how=='gal':
			key = lambda s: s.galaxy
		else:
			raise ValueError(f'SeriesProxy.groupby: unknown value passed to `how`: {how}')
		
		#argsort = np.argsort(self.apply(key))
		#return itertools.groupby([results[i] for i in argsort],key=key)
		
		... # finish/fix this

class RoedigerProxy(SeriesProxy):
	def __init__(self, *a, **k):
		super(RoedigerProxy,self).__init__((R0Series(),R90Series()))

class VollmerProxy(SeriesProxy):
	def __init__(self, *a, **k):
		super(VollmerProxy,self).__init__(tuple(s() for s in VollmerSeriesCollection(*a, **k)))

class AllSeriesProxy(SeriesProxy):
	def __init__(self, *a, drop=None, **k):
# 		print('called AllSeriesProxy.__init__')
		super(AllSeriesProxy,self).__init__(
			(R0Series(),R90Series(), *(s() for s in VollmerSeriesCollection(*a, **k)))
		)
# 		print('finished initializing AllSeriesProxy')

__all__ = ('RoedigerSeries','VollmerSeries','R0Series','R90Series',
		   'VollmerSeriesCollection','Proxy','VollmerProxy','AllSeriesProxy')