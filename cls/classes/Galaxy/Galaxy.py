"""The Galaxy class, the centerpiece of this module, is at the
center of all computations performed for this project. It maintains
a list of attributes describing its gas distribution, from which
relevant asymmetry quantities are calculated.

In addition, Galaxy (really its base, _Galaxy) is an example of a
class that takes advantage of overriding Python's special methods.
One can call an attribute on a Galaxy instance, even if it is not
yet defined; if it is a valid attribute, that attribute will be
computed and set. This applies even when the attribute recursively
depends on other such attributes; in this way, the class defines an
architecture for automatically exploring the graph (tree) that
defines the calculation dependencies of any given attribute.
_Galaxy never has to be accessed directly."""

from multiprocessing import Pool, cpu_count
import numpy as np
from matplotlib import pyplot as plt

from common import MultiIterator, getattrs, print_update, consume
from common.decorators import add_doc

from core import AttributeAbsent
from prop.asy_prop import *
from prop.galaxy_file import OTHER, cluster, ATLAS3D, external_references, galaxy_samples
from prop.asy_defaults import *

from asy_io.asy_io import fits_data

from comp.asymmetry_functions import (m2_calc, EA_calc, galaxy_outer_configure, get_m2_ext_quantities,
	ratio_angle_calc, get_galaxy_inner_regions, compute_region_f_per_t)
from comp.computation_functions import coorsgen, c_to_p
from comp.array_functions import reindex

from cls.plotters import galaxy_plot_funcs, galaxy_fig_funcs

from .galaxy_attribute_information import (
	all_attributes, dynamic_attributes, get_file_info, m2_attributes, arrays,
	observational_only_attributes, centroid_angle_attrs, centroid_radius_attrs,
	nonsaveable_arrays, other_attributes
)
from . import Galaxy_getters

from ..h5 import Galaxy_H5_Interface, H5_INTERFACE

from plot.plot_classes import PaperFigure
from plot.plotting_functions import ax_0N



class Null:
	"""
	A singleton to indicate the lack of a value for an attribute in other
	classes. `None` doesn't work because some attributes may validly be
	set to None.
	
	useful: https://python-patterns.guide/gang-of-four/singleton/
	"""
	_null=None
	
	def __new__(cls):
		if cls._null is None:
			cls._null = super().__new__(cls)
		return cls._null
	
	def __bool__(self): return False




class _Galaxy:
	"""Base class that tells how to compute data for a galaxy."""
# 	from .galaxy_attribute_information import *
	# imported at the module level
	
	# don't use more memory than needed
	__slots__ = all_attributes
	
	__getters__ = { # dict mapping attribute names to retrieving functions
		a: getattr(Galaxy_getters,a)
		for a in dynamic_attributes
	}
	
	# access to the HDF5 interface
	h5_interface = H5_INTERFACE
	
	# this will keep track of what data is loaded for different galaxies
	__loaded_data__ = None
	
	# used for missing data and error handling
	NULL = Null()
	
	def __str__(self): return f'<{self.filename}>'
	def __repr__(self): return f'<Galaxy{{{self.filename}}}>'
	
	
	################################
	###
	###		Initialization
	###
	################################
	
	def __init__(self, filename, *, compute = False, debug = False, reload = False, save=False):
		self.filename = filename
		try:
			#print(f'{filename}: setting directory to {galaxy_samples[filename]}')
			self.directory = galaxy_samples[filename]
			#print(f'{filename}: directory set to {self.directory}')
		except KeyError:
			raise NameError(f"the galaxy '{filename}' is not recognized")
		
		self.compute = compute
		# compute attributes from scratch or load from disk?
		
		self.debug = debug
		# for testing
		
		self.absent = MultiIterator()
		# attributes that are not allowed on this instance
		# empty by default
		
		self.save = save and (not debug)
		# allow computations to save to disk?
		
		"""if reload:
			self.retrieve_baseline_information()
		else:
			self.load_names_and_paths()  ######## {name/path parameters}
			self.load_fixed_quantities()  ######## {fixed quantities}"""
		
		self.retrieve_baseline_information()
	
	def set_galtype(self):
		"""Tell what kind of galaxy this is: real or simulated,
		and from there what subgroup it belongs to."""
		if self.filetype == 'r':
			self.is_real = True
			self.is_rsim = self.is_vsim = False
			self.is_other = (self.filename in OTHER)
			self.is_atlas3d = (self.filename in ATLAS3D)
			self.is_cluster = (self.filename in cluster)
			self.is_ref = (self.filename in external_references)
		else:
			self.is_real = self.is_cluster = self.is_atlas3d = self.is_other = self.is_ref = False
			self.is_vsim = (self.filename[0] == 'v')
			self.is_rsim = (not self.is_vsim)
	
	def retrieve_baseline_information(self):
		"""Loads underlying identification information about the galaxy,
		as well as the gas data from the corresponding FITS file."""
		self.openname, self.filetype = get_file_info(self.filename)
		self.set_galtype()
		#print(f'inside `retrieve_baseline_information`: self.directory={self.directory}')
		fits_data(self)
	
	def deny_m2(self):
		"""Mark that this galaxy does not support m=2-only attributes."""
		self.absent.add_target(m2_attributes)
	
	def deny_observational(self):
		"""Mark that this galaxy does not support observational-only attributes."""
		self.absent.add_target(observational_only_attributes)
	
	
	
	################################
	###
	###		Attribute retrieval and setting
	###
	################################
	
	def __getattr__(self, attr):
		"""
		process:
			check to see if the attribute is denied; if so, raise AttributeError
			set the attribute to NULL
			then see if the Galaxy class knows a way of retrieving it
			if so,
				try retrieving it
				if it cannot be retrieved, delete attribute and raise AttributeError
			if not, delete the attribute, then raise AttributeError
		"""
		
		#print(f'__getattr__ called on attribute <{attr}>')
		
		# check to see if the attribute is known to the Galaxy class
		# but denied for this particular Galaxy instance
		if attr in super().__getattribute__('absent'):
			raise AttributeAbsent(self,attr)
		
		# note, as seen above, be careful to use super().__getattribute__
		# instead of a straight attribute access, which would cause
		# infinite recursion in the event I overlook setting the
		# desired attribute. Another option is to access the attribute
		# through __class__ if pertinent, as below.
		
		# set the attribute to NULL to avoid infinite recursion
		setattr(self, attr, self.__class__.NULL)
		
		getters = self.__class__.__getters__
		
		# does the Galaxy class know how to retrieve this?
		if attr in getters:
			"""
			two occasions where we do not want to try loading data:
			(1) Galaxy instance is set to compute rather than load
			(2) the attribute is present in 'other_attributes', meaning it is
			    not meant to be computed or loaded as a numerical data type,
			    but is set by some other means; e.g. `ratio_process_plotter`
			"""
			if super().__getattribute__('compute') or attr in other_attributes:
# 				print('computing attribute:',attr)
				result = getters[attr](self)
			else:
# 				print('loading attribute:',attr)
				if attr not in nonsaveable_arrays:
					result = self.load_attr(attr)
					setattr(self, attr, result)
					#print(f'loaded attribute {repr(attr)}: type is {type(result)}')
				else:
					result = getters[attr](self)
					#print(f'attempted to load but had to compute attribute {repr(attr)}: type is {type(result)}')
			
			if getattr(self,attr) is self.__class__.NULL:
				delattr(self,attr)
				raise RuntimeError(f"Galaxy.__getattr__: attribute '{attr}': this attribute was not properly set")
# 			if attr in super().__getattribute__('storable'):
# 				setattr(self,attr,result)
			
			return result
		
		else: # no, it doesn't
			delattr(self,attr) # delete to avoid lingering nonce attribute
			return super().__getattribute__(attr) # raises AttributeError
	
	def load_attr(self,attr):
		"""load an attribute from the HDF5 interface"""
		if attr in arrays:
			return self.h5_interface.read_array_for_instance(self,attr)
		else:
			return self.h5_interface.read_scalar_for_instance(self,attr)
	
	def _setattrs_unzip(self, attrs, container):
		"""
		Set attrs based on a list of attribute names and container of values;
		the container is expected to support len(container)
		"""
		if len(attrs) != len(container):
			raise ValueError(
				'`_setattrs_unzip`: attribute list and value container have unequal lengths '
				'(%i, %i)' % (len(attrs), len(container))
			)
		for attr, v in zip(attrs, container):
			setattr(self, attr, v)
	
	def tryattr(self, attr, default=nan):
		"""
		Try getting an attribute, return `default` (default nan) if invalid.
		"""
		return getattr(self, attr, default)
	
	def tryattrs(self, *attrs, array=False, default=nan):
		"""Map tryattr across multiple attributes."""
		if (not array) and (set(attrs) & arrays):
			return tuple(self.tryattr(attr, default) for attr in attrs)
		return np.array(tuple(self.tryattr(attr, default) for attr in attrs))
	
	def getattrs(self, *vals):
		"""
		Return multiple attributes; if any values passed
		are not strings, return them directly, otherwise return
		the corresponding attribute on this instance.
		"""
		return [(getattr(self, v)) if isinstance(v, str) else v for v in vals]
	
	def populate(self):
		"""
		load all attributes that can be loaded
		"""
# 		consume(map(self.tryattr, all_attributes))
		for a in all_attributes:
		    try:
		    	getattr(self,a)
		    except AttributeAbsent as e:
		    	pass
	
	
	
	################################
	###
	###		other computations
	###
	################################

	def get_total_xy(self):
		"""Establish x and y coordinates for all pixels."""
		self._setattrs_unzip(
			('total_x', 'total_y'),
			coorsgen(self.zdata, as_columns=False)
		)

	@add_doc(galaxy_outer_configure)
	def isolate_outer_galaxy(self):
		"""Isolate and generate arrays for outer & wrongside pixels."""
		galaxy_outer_configure(self)

	def compute_total_theta_radii(self):
		"""Establish theta and radial coordinates for all pixels."""
		self._setattrs_unzip(
			('total_t', 'total_r'),
			c_to_p(self.total_x, self.total_y, self.xpix, self.ypix)
		)

	def arrange_total_f_by_t(self):
		"""Digitizes total flux analogous to `nc_f_per_t`."""
		theta_int, flux = self.total_t_int_tsort, self.total_f_tsort
		self.total_f_per_t = compute_region_f_per_t(theta_int, flux)
		self.total_f_per_t_rw = compute_region_f_per_t(theta_int, flux * self.total_r_tsort)

	def get_extent_arrays(self):
		"""
		generate:
			* extentlist_graph, columns: angles in degrees, extents in arcsec at these angles
			* plot_list_adjust: as above, save that units are radians // pixels
		
		Note to self: check whether the extents in extentlist_graph are
		in arcsec for simulated galaxies or only for real galaxies
		"""
		extentlist_graph = reindex((self.extentlist - [tau / 4, 0]) * [rad_to_deg, self.pix_scale_arcsec], ahl)
		extentlist_graph[:, 0] %= 360
		self.extentlist_graph = extentlist_graph
		plot_list_adjust = reindex(self.extentlist - [tau / 4, 0], ahl)
		plot_list_adjust[:, 0] %= tau
		self.plot_list_adjust = plot_list_adjust

	@add_doc(get_galaxy_inner_regions)
	def compute_inner(self):
		"""Isolate and generate arrays for inner pixels."""
		get_galaxy_inner_regions(self)
	
	@add_doc(ratio_angle_calc)
	def get_m1_flux_quantities(self):
		"""Get flux asymmetry quantities with all possible weighting schemes."""
		ratio_angle_calc(self)
		ratio_angle_calc(self, trig=True)
		ratio_angle_calc(self, rweight=True)
		ratio_angle_calc(self, trig=True, rweight=True)

	@add_doc(get_m2_ext_quantities)
	def get_m2_extent_data(self):
		"""Get raw m=2 extent-related quantities if permitted on this galaxy;
		note this precedes the calculation of the m=2 extent ratio and other
		m=2 asymmetry data."""
		if deny_m2_high_i and self.inclination > high_i_threshold:
			consume(setattr(self, attr, nan) for attr in m2_attributes)
		else:
			get_m2_ext_quantities(self)

	@add_doc(m2_calc)
	def compute_m2_asymmetry_arrays(self):
		"""Compute all m=2 asymmetry quantities."""
		if deny_m2_high_i and self.inclination > high_i_threshold:
			consume(setattr(self, attr, nan) for attr in m2_attributes)
		else:
			m2_calc(self)

	@add_doc(compute_m2_asymmetry_arrays)
	def get_m2_quantities(self):
		"""Compute all m=2 asymmetry quantities via 'compute_m2_asymmetry_arrays'."""
		if deny_m2_high_i and self.inclination > high_i_threshold:
			consume(setattr(self, attr, nan) for attr in m2_attributes)
		else:
			self.compute_m2_asymmetry_arrays()

	def get_centroid_data(self):
		"""compute centroid angles/radii and tail angle"""
		self._setattrs_unzip(centroid_angle_attrs, self.centroids[:, 0])
		self._setattrs_unzip(centroid_radius_attrs, self.centroids[:, 1])

		self.tail_A, self.tail_r = self.weighted_centroid_angle, self.weighted_centroid_radius
		self.TA = (self.tail_A+180)%360

	def compute_m1_ext_asymmetry(self):
		"""Compute extent ratios and asymmetries (quadrant, half-disk)"""
		self.EA, self.ER = EA_calc(self.score)
		self.qEA, self.qER = EA_calc(self.qscore)

		self.EA_trig, self.ER_trig = EA_calc(self.score_trig)
		self.qEA_trig, self.qER_trig = EA_calc(self.qscore_trig)
		
	def get_short_long_sides(self):
		"""Compute shortside_list_graph and longside_list_graph"""
		extentlist_graph=self.extentlist_graph
		extg_cind=np.where(np.isclose(extentlist_graph[:, 0], self.EA))[0][0]
		self.shortside_list_graph=np.copy(extentlist_graph[(extg_cind+birange_ahl)%a2l])
		self.longside_list_graph=np.copy(extentlist_graph[(extg_cind+al+birange_ahl)%a2l])
	
	def get_deprojected_sides(self):
		"""Compute shortside_list_graph_deproject and longside_list_graph_deproject"""
		extentlist_graph_deproject=self.extentlist_graph_deproject
		extg_cind=np.where(np.isclose(extentlist_graph_deproject[:, 0], self.EA))[0][0]
		self.shortside_list_graph_deproject=np.copy(extentlist_graph_deproject[(extg_cind+np.arange(-ahl, ahl+1))%a2l])
		self.longside_list_graph_deproject=np.copy(extentlist_graph_deproject[(extg_cind+al+np.arange(-ahl, ahl+1))%a2l])

	def get_shortsums(self, lin_weights=np.linspace(1, 0, 101)):
		"""Compute shortsum (fraction of shortside angles where radius is
		shorter than on opposite side) and weighted variants."""
		shortlong_dif = self.shortside_list_graph[:, 1] - self.longside_list_graph[:, 1]
		short_shorter = (shortlong_dif <= self.pix_scale_arcsec)
		self.shortsum = np.average(short_shorter)
		self.shortsum_lin = np.average(short_shorter, weights=lin_weights)
		self.shortsum_trig = np.average(short_shorter, weights=trig_weights)

	def setattr_argsort(self, internal_sorter_array, sort_str):
		"""Get and set as an attribute an array sorted by another array."""
		inds = np.argsort(internal_sorter_array)
		setattr(self, sort_str + 'sort_inds', inds)
		

class Galaxy(_Galaxy):	
	"""Supplements computational instructions from _Galaxy with instructions
	on how to load attributes via the HDF5 interface."""
	
	# again, keep memory usage down
	__slots__ = ()
	
	from prop.galaxy_file import galaxy_noise_data
	
	@classmethod
	@add_doc(Galaxy_H5_Interface.write_files_to_h5)
	def write_to_h5(cls, *filenames):
		"""Write data for galaxies to hdf5 file."""
		from ..h5 import write_h5_file_contents
		recomputed_galaxies = cls.recompute(*filenames)
		cls.h5_interface.write_files_to_h5(recomputed_galaxies)
		write_h5_file_contents()
		return recomputed_galaxies
	
	@classmethod
	def read_scalars(cls, filenames, attributes):
		"""Retrieve scalar data as pandas DataFrame with filenames
		as index, attributes as columns. Allows for reading data
		for particular galaxies without establishing Galaxy instances
		by directly interfacing with the HDF5 interface.
		
		`filenames`, `attributes` may both be strings or containers of
		strings. <None> is not permitted for either parameter.
		"""
		if filenames is None:
			raise ValueError("pass non-None value for `filenames`")
		if attributes is None:
			raise ValueError("pass non-None value for `attributes`")
		
		d = cls.h5_interface.get_scalars_for_filenames(filenames, attributes)
		
		return d.values.squeeze()
	
	@classmethod
	@add_doc(Galaxy_H5_Interface.read_arrays)
	def read_arrays(cls, filenames, attributes):
		"""Retrieve array data from the HDF5 interface.
		The result of `Galaxy_H5_Interface.read_arrays` is returned.
		
		`filenames`, `attributes` may both be strings or containers of
		strings. <None> is not permitted for either parameter."""
		if filenames is None:
			raise ValueError("pass non-None value for `filenames`")
		if attributes is None:
			raise ValueError("pass non-None value for `attributes`")
		
		d = cls.h5_interface.read_arrays(filenames, attributes, permit_absent = True)
		
		"""test:
		results = Galaxy.read_arrays(filenames, attrs)
		all(np.allclose(r,ar, equal_nan=True)
			for a,f in zip(results, filenames)
				for r,ar in zip(a, Galaxy(f).tryattrs(*attrs))
		)
		
		# so far verified on cluster, ATLAS3D, OTHER, sim_fn
		"""
		
		return d
	
	@classmethod
	def get_noise_std(cls, filenames, quantities, reduce=True):
		"""Get the standard deviation in a quantity as computed in
		the Monte Carlo noise simulations."""
		filenames, quantities = map(np.atleast_1d, (filenames, quantities))
		r = cls.galaxy_noise_data.loc[filenames, quantities]
		if reduce: return r.values.squeeze()
		return r
	
	@classmethod
	def compute_all_values(cls, filename, attribute_set=None):
		"""
		create a new Galaxy instance and compute values on it from scratch
		if attribute_set is not specified, all attributes are computed
		"""
		print_update('recomputing',filename)
		g = Galaxy(filename,compute=True)
		
		# the computational machinery is written into the __getattr__ method
		# so simply getting all permitted attributes ensures they are computed
		if attribute_set is None:
			attribute_set = all_attributes
		
		for attr in attribute_set:
			try: getattr(g,attr)
			except AttributeAbsent: pass
		return g
	
	@classmethod
	@add_doc(compute_all_values)
	def recompute(cls, *filenames, attribute_set=None):
		"""
		Call 'compute_all_values' on provided filenames.
		"""
		if attribute_set is None:
			attribute_set = all_attributes
		
		if len(filenames)==0:
			print('no filenames passed to _Galaxy.recompute; returning')
			return
		
		if len(filenames)==1:
			res = (cls.compute_all_values(filenames[0]),)
			print_update('')
			return res
		
		p = Pool(min(cpu_count(),len(filenames)))
		galaxies = p.map(cls.compute_all_values,filenames)
		
		print_update('closing pool...')
		p.close()
		p.join()
		
		print_update('')
		
		return galaxies
	
	def get_imshow_lims(self):
		"""Get the axis limits when calling plt.imshow(self.zdata)"""
		return (
			np.array((self.xpix, self.xpix, self.ypix, self.ypix))+
			np.array((-1, 1, -1, 1))*(self.edge_radius/self.pix_scale_arcsec)
		)

	def m1_process_plots(self):
		"""Make plots relating to m=1 asymnmetry calculation process."""
		self.m1_flux_process_plot()
		self.m1_extent_process_plot()
	
	def m2_process_plots(self):
		"""Make plots relating to m=2 asymnmetry calculation process."""
		self.deprojection_plot()
		self.m2_flux_process_plot()
		self.m2_extent_process_plot()
	
	@classmethod
	def asy_summary(cls, instance = None, galaxy_names = ('4254',)*4+('4561','4561')):
		"""6-panel figure symmarizing asy types; pass a list of 6 names."""
		gmap = {}
		fig = plt.figure(FigureClass=PaperFigure,paper='methods')
		axes = fig.subplots(2,3, subplot_kw={'polar':True})
		axes=axes.ravel()
		consume(map(ax_0N, axes))
		if instance is not None:
			galaxies = (instance for _ in range(6))
			save_name = f'{instance.filename} asy summary plots'
		else:
			galaxies = (gmap.setdefault(n, cls(n)) for n in galaxy_names)
			save_name = 'multi asy summary'
		next(galaxies).ratio_process_plotter.m1_ht_process_polar_frame(0, axes[0])
		next(galaxies).ratio_process_plotter.m1_extent_process_polar_frame(0, axes[1])
		next(galaxies).ratio_process_plotter.m1_flux_process_polar_frame(0, axes[2])
		next(galaxies).tail_map(axes[3])
		next(galaxies).ratio_process_plotter.m2_extent_process_polar_frame(0, axes[4])
		next(galaxies).ratio_process_plotter.m2_flux_process_polar_frame(0, axes[5])
		fig.save(save_name, dump=2)
		plt.close(fig)


# functions defined in other modules intended to be set on this class
for func in galaxy_plot_funcs.__all__:
	setattr(Galaxy, func, getattr(galaxy_plot_funcs, func))
for func in galaxy_fig_funcs.__all__:
	setattr(Galaxy, func, getattr(galaxy_fig_funcs, func))




__all__ = ('Galaxy',)

if __name__ == '__main__':
	import traceback
	from prop.galaxy_file import *
	import sys
	
	def run(gal):
		print('running galaxy',gal)
		g = Galaxy(gal,compute=True)
		
		absents = []
		
		for a in all_attributes:
		    try:
		    	print_update(f'{g.filename}: getting {a}')
		    	assert getattr(g,a) is not Galaxy.NULL
		    	#print(f' (type={type(_)})')
		    except AttributeError as e:
		    	if a in g.absent: absents.append(a)
		    	else: print(f'\n*\t*\t*\t{a}:\n{traceback.format_exc()}')
		    except Exception as e: print(f'\n*\t*\t*\t{a}:\n{traceback.format_exc()}')
		
		print_update('')
			
		if absents:
			print('\tattempted to access absent attributes:',absents,sep='\n\t')
	
	#gals = [incgroup[0] for gal,incgroups in vollmer_groups.items() for inc,incgroup in incgroups.items()]
	gals = sys.argv[1:]
	if not gals: gals=['4330']
	for g in gals: run(g)