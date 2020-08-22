"""The h5 module is an abstraction that allows the Galaxy class to communicate
with data stored in the filesystem. It allows for accessing two kinds of data:
scalar data (e.g. individual ratios, angles) and arrays. Internally, everything
is stored in arrays in an hdf5 file, and this module, through the class
'Galaxy_H5_Interface' and other functions, allow for accessing data indexed by:
	(1) Galaxy name
	(2) Data name

Given that h5py (perhaps hdf5 in general) can get slow when making a large
number of repeated calls to rows at different indices within a Dataset, an
algorithm is also contained to group rows contiguous to one another when
writing scalars to disk to minimze the number of distinct write events required.
"""
import os, traceback
from collections import OrderedDict, ChainMap

import numpy as np
from numpy import nan
import h5py, h5pickle
import pandas as pd

from asy_io.asy_paths import H5_PATH
from asy_io.asy_io import print_update
from prop.galaxy_file import galaxy_samples#, galaxies
from core import Queuer, consume, AttributeAbsent, groupby_whole, array_assign
from comp.array_functions import get_regions
from .Galaxy.galaxy_attribute_information import non_arrays, saveable_arrays, baseline_attributes

#_CLUSTER = True

def split_sorted_1d_coordinates_contiguous(array):
	"""
	given a list of sorted 1-d coordinates,
	split them into contiguous chunks (subarrays)
	"""
	return np.array([chunk[[0,-1]] for chunk in get_regions(array,1)])

def initialize_galaxy_data_file():#clusters=_CLUSTER
	#"""
	#
	#what we want to do with clustering:
	#	group attributes within galaxy samples
	#
	#"""
	print('initialize_galaxy_data_file called')
	h5_file = h5py.File(H5_PATH,'w')
	
	for sample in ('cluster','ATLAS3D','OTHER','rsim','vsim'):
		h5_file.create_group(sample)
	
	_intialize_galaxy_data_file_sample_groups(h5_file)
	_intialize_galaxy_data_file_scalars(h5_file)
	
	h5_file.close()

def _intialize_galaxy_data_file_sample_groups(h5_file):
	"""make h5py groups for all galaxies inside respective cluster groups"""
	consume(_make_galaxy_group(h5_file,g) for g in galaxy_samples)
	print_update('')

def _make_galaxy_group(h5_file,g):
	"""make an h5py group for a galaxy inside its respective cluster group"""
	print_update('making group for',g)
	h5_file[Galaxy_H5_Interface
		.get_h5_sample_group_from_filename(g)
	].create_group(g)

def _intialize_galaxy_data_file_scalars(h5_file):
	q = np.array(list(non_arrays - baseline_attributes),dtype='S')
	#print('_intialize_galaxy_data_file_scalars: q:',q)
	h5_file.attrs.create('scalar quantities', q)
	for key in h5_file.keys():
		group = h5_file[key]
		group.create_dataset(
			'__scalars__',
			data=np.zeros([len(group.keys()), len(q)])
		)
		included_galaxies = np.sort([
			g for g in galaxy_samples if 
				Galaxy_H5_Interface.galaxy_samples_h5_group_mapping[galaxy_samples[g]]
				== key
		]).astype('S')
		group.attrs.create('filenames',included_galaxies)

def open_galaxy_data_file(mode):
	"""Open (do not return) the hdf5 file"""
	if not os.path.exists(H5_PATH):
		initialize_galaxy_data_file()
	if mode == 'r': return h5pickle.File(H5_PATH,'r')
	elif mode in ('w','a'): return h5py.File(H5_PATH, mode)
	raise ValueError(f"mode must be one of ('r', 'w', 'a'); you provided '{mode}'")

def write_h5_file_contents():
	"""Write into a file a pretty-printed list of all groups and datasets
	contained in the hdf5 file. Datasets have size in bytes recorded in the file."""
	path = f"{H5_PATH[:H5_PATH.rfind('/')]}/h5 fields.file"
	
	from collections import deque
	from h5py import Dataset, File
	
	d = deque()
	
	# when writing to a file, record the size in bytes of each Dataset
	# object encountered; n-->'name', o-->'object'
	def add(n, o):
		if isinstance(o, Dataset): d.append((n, o[()].nbytes))
		else: d.append((n, None))
	
	# h5_file.visititems recursively walks elements contained in the hdf5 file,
	# submitting two arguments to add: (object name, object);
	# since 'add' always returns None, all elements (objects) are processed
	h5_file = File(H5_PATH,'r')
	h5_file.visititems(add)
	h5_file.close()
	
	# utility function to pretty-print a header name based on its depth within
	# the hdf5 file; ensures that headers appear nested when written
	def makeline(name,size):
		s=name.split('/')
		depth,field = len(s)-1,s[-1]
		t='\t'*depth
		if size is None: return f'{t}{field}'
		return f'{t}{field}: size = {size}'
	
	# should I explicitly chunk the writing?
	with open(path,'w') as out:
		out.write('\n'.join(map(makeline,*zip(*d))))


class Galaxy_H5_Interface:
	"""
	Note: it will become desirable to map the GalaxyCollection class to h5
	as well, meaning that this class should be generalized to an H5_interface class,
	of which Galaxy_H5_Interface and GalaxyCollection_H5_Interface can be subclasses
	"""
	_instance = None
	
	galaxy_samples_h5_group_mapping={
		'cluster':'cluster',
		'ATLAS3D':'ATLAS3D',
		'OTHER':'OTHER',
		'sim':'rsim',
		'v4501':'vsim',
		'v4522':'vsim',
		'v4388':'vsim',
		'v4330':'vsim'
	}
	
	__scalar_data__ = {}
	sample_filename_mappings = {}
	
	def __new__(cls,*a,**kw):
		if cls._instance is None:
			cls._instance = object.__new__(cls)
		cls._instance._intialized = False
		return cls._instance
	
	def __init__(self, preload_scalars=True):
# 		print('__init__ in Galaxy_H5_Interface')
		try:
			# only initialize once; if already done, this runs without error
# 			print('testing for self._initialized')
			self._initialized
			return
		except AttributeError:
			#otherwise, initialize the instance
			self._set_reader()
			
			# array of scalar quantity names in the order
			# of occurrence in the h5 scalar arrays (along axis 1)
			self.scalar_q = self.reader.attrs['scalar quantities'].astype('U')
			# would be better to define scalar_q as a @property returning keys of the following odict
			self.scalar_q_mapping = OrderedDict((q,i) for i,q in enumerate(self.scalar_q))
				# {q:i for i,q in enumerate(self.scalar_q)}
			
			#self.sample_filename_mappings = {}
			self.preload_scalars = preload_scalars
			
			for sample in self.reader:
				#print('h5 top-level group:',sample)
				d = OrderedDict((f,i) for i,f in enumerate(self.get_scalar_filename_order(sample)))
					# {f:i for i,f in enumerate(self.get_scalar_filename_order(sample))}
				self.sample_filename_mappings[sample] = d
				
				if self.preload_scalars:
					self._set_scalars_for_sample_from_disk(
						sample,
						list(d.keys()),
						h5file=self.reader
					)
					#self.__scalar_data__[sample] = pd.DataFrame(
					#	self._get_scalars_for_sample_from_disk(sample,h5file=self.reader),
					#	index=list(d.keys()),
					#	columns=self.scalar_q
					#)
			self._initialized = True
	
	def _set_reader(self):
		"""
		A reader is an h5py.File instance in 'r' mode.
		
		If no reader is defined, instantiate a new one;
		if it is, close the existing one and create a new one.
		"""
		print('called _set_reader: setting h5 reader')
		try:
			print('attempting to close reader')
			self.reader.close()
		except AttributeError:
			print('\t>> reader attribute is not defined')
			pass
		
		print('opening the reader')
		self.reader = open_galaxy_data_file('r')
		print('reader has been set')
		self.reader_set = True
	
	def _get_scalars_for_sample_from_disk(self,sample_name,*,h5file=None):
		if h5file is None:
			raise ValueError('h5file must be supplied as an h5py-compatible file')
		return h5file[sample_name]['__scalars__']
	
	def _set_scalars_for_sample_from_disk(self, sample_name, index=None, *, h5file=None):
		if index is None: index = self.sample_filename_mappings[sample].keys()
		
		self.__scalar_data__[sample_name] = data = pd.DataFrame(
			self._get_scalars_for_sample_from_disk(sample_name,h5file=h5file),
			index=index,
			columns=self.scalar_q
		)
		return data
	
	def get_scalars_for_sample(self,sample_name,*,h5file=None):
		"""
		try returning scalar array in active memory;
		if absent, retrieve from disk, save in active memory, and return
		"""
		try:
			return self.__scalar_data__[sample_name]
		except KeyError: 
			return self._set_scalars_for_sample_from_disk(sample_name, h5file=h5file)
	
	def get_scalars_for_filenames(self, galaxies, attr=None):
		sample_filename_lists = groupby_whole(galaxies, key = self.get_h5_sample_group_from_filename)
		
		scalars=[
			self.get_scalars_for_filenames_from_single_sample(g,*gal_list,attr=attr)
			for g,gal_list in sample_filename_lists.items()
		]
		
		if len(scalars)==1:
			return scalars[0]
		return pd.concat(scalars)
	
	def get_scalars_for_filenames_from_single_sample(self, sample_name, *galaxies, attr=None):
		if attr is None:
			#return self.__scalar_data__[sample_name].loc[list(galaxies)]
			return self.get_scalars_for_sample(sample_name).loc[list(galaxies)]
		if isinstance(attr,str):
			#return self.__scalar_data__[sample_name].loc[list(galaxies),[attr]]
			return self.get_scalars_for_sample(sample_name).loc[list(galaxies),[attr]]
		#return self.__scalar_data__[sample_name].loc[list(galaxies),list(attr)]
		return self.get_scalars_for_sample(sample_name).loc[list(galaxies),list(attr)]
	
	def get_scalars_for_filename(self, galaxy, attr=None):
		sample_name = self.get_h5_sample_group_from_filename(galaxy)
		return self.get_scalars_for_filenames_from_single_sample(sample_name, galaxy, attr=attr)
		"""if attr is None:
			return self.__scalar_data__[self.get_h5_sample_group_from_filename(galaxy)].loc[galaxy]
		if isinstance(attr,str):
			return self.__scalar_data__[self.get_h5_sample_group_from_filename(galaxy)].loc[galaxy,[attr]]
		return self.__scalar_data__[self.get_h5_sample_group_from_filename(galaxy)].loc[galaxy,list(attr)]"""
	
	def get_scalar_filename_order(self,sample_name,h5file=None):
		try:
			return self.__scalar_data__[sample_name].index.values
		except KeyError:
			if h5file is None: h5file = self.reader
			#print(f'get_scalar_filename_order: {sample_name}.attrs("filenames")')
			return h5file[sample_name].attrs['filenames'].astype('U')
	
	@classmethod
	def get_h5_sample_group_from_filename(cls,filename):
		return cls.galaxy_samples_h5_group_mapping[galaxy_samples[filename]]
	
	def get_h5_sample_name(self,galaxy_instance):
		if galaxy_instance.is_cluster: return 'cluster'
		if galaxy_instance.is_atlas3d: return 'ATLAS3D'
		if galaxy_instance.is_other: return 'OTHER'
		if galaxy_instance.is_rsim: return 'rsim'
		if galaxy_instance.is_vsim: return 'vsim'
		
		raise ValueError(
			f'this galaxy is not configured correctly: sample unobtainable: {galaxy_instance}'
		)
	
	@classmethod
	def group_attributes(cls,attributes):
		"""
		given a set of attributes, return a dict with two keys:
			True: the corresponding value is a list of attributes identified as arrays
			False:  the corresponding value is a list of attributes identified as non-arrays
		"""
		return groupby_whole(attributes, key = lambda v: attributes in arrays)
	
	def read_scalars(self, filenames, attributes=None):
		if self.preload_scalars:
			return self.get_scalars_for_filenames(filenames, attributes=attributes)
		else:
			attribute_groups = self.group_attributes(attributes)
			filename_mapping = {g:i for i,g in enumerate(filenames)}
			filename_groups = groupby_whole(
				enumerate(filenames),
				key = lambda p: self.get_h5_sample_group_from_filename(p[1])
			)
			
			scalar_attribute_indices = np.array([self.scalar_q_mapping[q] for q in scalar_attributes])
			
			results = [None]*len(filenames)
			for g,filename_list in filename_groups:
				# put the results into the results list
				# in the proper locations
				array_assign(
					results, # destination list
					(filename_mapping[f] for f in filename_list), # indices
					self._read_galaxy_scalars_from_sample(
						g, filename_list, scalar_attribute_indices
					) # values
				)
		
		return results
	
	def _read_galaxy_scalars_from_sample(self, sample_name, filenames, scalar_attribute_indices):
		
		sample_galaxy_ordering_map, chunk_bounds = \
			get_contiguous_filename_groups(sample_name, filenames, h5file=self.reader)
		
		scalars = np.vstack([
			self.reader[sample_name]['__scalars__'][start:end+1,scalar_attribute_indices]
			for start,end in chunk_bounds
		])
		
		return scalars
	
	def read_arrays(self, filenames, attributes):
		attribute_groups = self.group_attributes(attributes)
		filename_mapping = {g:i for i,g in enumerate(filenames)}
		filename_groups = groupby_whole(
			enumerate(filenames),
			key = lambda p: self.get_h5_sample_group_from_filename(p[1])
		)
		
		results = [None]*len(filenames)
		for g,filename_list in filename_groups:
			# put the results into the results list
			# in the proper locations
			array_assign(
				results, # destination list
				(filename_mapping[f] for f in filename_list), # indices
				self._read_galaxy_arrays_from_sample(
					g, filename_list, attributes
				) # values
			)
		
		return results
	
	def _read_galaxy_arrays_from_sample(self, sample_name, filenames, attributes):
		sample_group = self.reader[sample_name]
		return [self._read_arrays_from_galaxy(sample_group[fn],attributes) for fn in filenames]
	
	def _read_arrays_from_galaxy(self,galaxy_h5_container, attributes):
		return [galaxy_h5_container[a][()] for a in attributes]
	
	#def _read_galaxies_from_sample(self, group, filenames, array_attributes):
	#	"""
	#	Get data for any number of galaxies and any number of attributes
	#	
	#	(NOT IMPLEMENTED) Special values for 'attributes':
	#		None: return all scalar and array attributes
	#		'scalars': return all scalar attributes
	#		'arrays': return all array attributes
	#	"""
	#	
	#	return [[group[fn][a] for a in array_attributes] for fn in filenames]
	
	def read_scalar_for_instance(self, galaxy_instance, attr):
		if self.preload_scalars:
			pd_result = self.get_scalars_for_filename(galaxy_instance.filename, attr)
			return pd_result.loc[galaxy_instance.filename,attr] # getattr(pd_result,attr)
		else:
			sample_group = self.get_h5_sample_name(galaxy_instance)
			fi = self.sample_filename_mappings[sample_group][galaxy_instance.filename]
			ai = self.scalar_q_mapping[attr]
			return self.reader[sample_group]['__scalars__'][fi,ai]
	
	def read_array_for_instance(self, galaxy_instance, attr):
		"""read an array for a given galaxy from disk"""
		return self.reader[self.get_path_from_instance(galaxy_instance)][attr][()]
	
	def get_path_from_instance(self, galaxy_instance):
		"""return the path of the h5 Group corresponding to a single galaxy instance"""
		return f'{self.get_h5_sample_name(galaxy_instance)}/{galaxy_instance.filename}'
	
	def get_path_from_filename(self, filename):
		"""return the path of the h5 Group corresponding to a single galaxy name"""
		return f'{self.get_h5_sample_group_from_filename(filename)}/{filename}'
	
	
	def write_files_to_h5(self, galaxies):
		"""
		Write data to H5 file for arbitrary number of galaxies.
		
		Note: until I configure SWMR for reading for h5 files,
		the read-only h5 File objects get closed irreversibly,
		and so calling this function terminates the Python interpreter.
		
		Another option, when SWMR is configured, is to call
		the writing function in another process apart from the main one,
		perhaps by encapsulating functionality in another Python script
		and calling it via the subprocesses module
		"""
		if not galaxies:
			print('no filenames given to write to H5; returning')
			return
		
		# recompute galaxies
		#galaxies = _Galaxy.recompute(*filenames)
		# multi-thread the procedure of writing results to disk
		
		try:
			self.reader.close()
		except AttributeError:
			pass
		
		print('setting h5 writer')
		self.writer = open_galaxy_data_file('a')
		
		self._write_arrays_to_h5_multi(galaxies)
		self._write_scalars_to_h5(galaxies)
		
		print('h5 interface: closing writer')
		self.writer.close()
		#del self.writer
		print('h5 interface: setting reader')
		self._set_reader()
		print_update('')
	
	def _write_scalars_to_h5(self, galaxies):
		"""
		Write all scalars from provided galaxies to disk
		
		
		IMPORTANT: to be implemented:
			if the filename is not found:
				add to a list of not found, and then after the multithreading is done,
				then go back in and write it in
		"""
		
		print(f'_write_scalars_to_h5: galaxies (count = {len(galaxies)}):')
		galaxy_map = {g.filename:g for g in galaxies}
		for sample in self.writer.keys():
			# there is likely a better way to do the following using groupby_whole
			# outside of this loop
			galaxy_submap = {
				filename:instance
				for filename,instance in galaxy_map.items()
				if Galaxy_H5_Interface.galaxy_samples_h5_group_mapping[galaxy_samples[filename]]==sample
			}
			self._write_scalars_for_sample(sample, galaxy_submap)
	
	def get_contiguous_filename_groups(self,sample_name,filenames,*,h5file=None):
		"""
		Given the name of a particular h5 Group
		and names of particular galaxies in this group,
		collect the galaxies into groups based on
		which are contiguous to the others in memory.
		"""
		
		"""
		h5file[sample_name].attrs['filenames'] = array of filenames
		associated with data contained in the scalars array;
		
		the order of names gives the order of the data in scalars
			(traversing galaxies corresponds to moving along axis 0)
		
		so this mapping tells which filename's data (value)
			is located at the given index (key)
		"""
		
		if not isinstance(h5file,(h5py.File,h5pickle.File)):
			raise ValueError('h5file must be supplied as an h5py-compatible file')
		
		# use OrderedDict to retain order of indices added
		sample_galaxy_ordering_map = OrderedDict(
			(i,n) for i,n in enumerate(
				self.get_scalar_filename_order(sample_name,h5file=h5file)
			)
			if n in filenames
		)
		print(f'_write_scalars_for_sample: at sample {sample_name}')
		print(f'sample_galaxy_ordering_map: {sample_galaxy_ordering_map}')
		
		# we will access the galaxies at the included indices
		# we want to collect these indices into contiguous blocks
		# chunk_bounds = list of arrays of contiguous indices
		chunk_bounds = split_sorted_1d_coordinates_contiguous(
			tuple(sample_galaxy_ordering_map.keys())
		)
		
		return sample_galaxy_ordering_map, chunk_bounds
	
	def _write_scalars_for_sample(self,sample_name,galaxy_map):
		"""
		tactic:
		
		We can write an entire sample at once by being clever about the order in which galaxies
		are written
		
		h5py does not allow indexing of the variety Dataset[<array of indices>,<array of indices>],
			or any other access involving more than one index array
		It does however allow for Dataset[<array of indices>,:], Dataset[:,<array of indices>],
			or any access involving one indexing array and colons otherwise
		
		This is compatible with what can be done here:
		In particular, when a galaxy is written, ALL of its scalars are written at once
			(by design, not requirement)
		So if we can arrange galaxies into blocks that are contiguous in the respective Dataset,
			we can use fancy indexing with colons to fill in multiple galaxies simultaneously
		
		this is what is done here
		
		
		parameters:
			sample_name (str): name of the h5 sample group containing the galaxy
			galaxy_map (): mapping <filename: galaxy instance>
		"""
		if not galaxy_map:
			print(f'no galaxies given to _write_scalars_for_sample for {repr(sample_name)}; returning')
			return
		# h5py.Group instance demarcating a galaxy sample
		sample_group = self.writer[sample_name]
		# scalar array for this sample
		scalars = sample_group['__scalars__']
		
		sample_galaxy_ordering_map, chunk_bounds = \
			self.get_contiguous_filename_groups(
				sample_name,
				galaxy_map.keys(),
				h5file=self.writer
		)
		
		# if a lot of, write in parallel across threads
		if len(chunk_bounds)>10:
			print('_write_scalars_for_sample: threading via Queuer')
			q = Queuer(self.write_contiguous_galaxy_scalars_chunk_to_h5,thread_count=5)
			consume(
				q.add_call(
					scalars, sample_galaxy_ordering_map,
					chunk_i, galaxy_map
				)
				for chunk_i in chunk_bounds
			)
			q.run()
		
		# otherwise just read individually
		else:
			print('_write_scalars_for_sample: single thread via consume')
			consume(
				self.write_contiguous_galaxy_scalars_chunk_to_h5(
					scalars, sample_galaxy_ordering_map,
					chunk_i, galaxy_map
				)
				for chunk_i in chunk_bounds
			)
	
	def write_contiguous_galaxy_scalars_chunk_to_h5(self,
		scalars, sample_galaxy_ordering_map,
		start_end_pair, galaxy_map
	):
		"""
		galaxy_map:
			mapping <galaxy (file)name: galaxy instance>
		
		sample_galaxy_ordering_map:
			mapping
			<
				index where galaxy data is located in the scalars array:
				the corresponding filename
			>
		
		start_end_pair: list of arrays containing indices of data to be written
		
		
		NOTE:
			in this function we are both retrieving the attributes from galaxy instances
				and writing them to h5
			
			may want to separate these concerns and retrieve the attributes
				outside of this function
		"""
		# start and end indices
		start, end = start_end_pair
		# filenames associated with the given indices
		names = (sample_galaxy_ordering_map[i] for i in range(start,end+1))
		# the scalar attributes for each of these galaxies
		attrs = np.array([
			galaxy_map[g].tryattrs(*self.scalar_q,array=True)
			for g in names
		])
		
		"""
		we have the indices where the data should be written,
		and the data in the order it should be written there,
		so go ahead and write it
		
		start_end_pair[[0,-1]] tells the (first, last) indices to write to,
		so based on indexing rules, assign to scalars[first : last+1]
		"""
# 		print('write_contiguous_galaxy_scalars_chunk_to_h5:'
# 				f'\ngalaxies: {names}'
# 				f'\ngalaxy_map: {galaxy_map}'
# 				f'\nwriting to scalars[{start_end_pair[0]}:{start_end_pair[-1]+1}, :]'
# 				f' (shape {scalars[start_end_pair[0]:start_end_pair[-1]+1, :].shape}):'
# 				f'\ndata {attrs} with shape {attrs.shape} and dtype {attrs.dtype}'
# 		)
# 		for a,d in zip(self.scalar_q, attrs.squeeze()):
# 			print(a,d)
# 		for g in names:
# 			print(g, galaxy_map[g], galaxy_map[g].tryattrs(*self.scalar_q,array=True))
# 		print('')
		scalars[start_end_pair[0]:start_end_pair[-1]+1, :] = attrs
	
	def _write_arrays_to_h5_multi(self,galaxy_instances):
		err = None
		try: 
			if len(galaxy_instances)>1:
				q = Queuer(self._write_arrays_to_h5_single, thread_count=10)
				# each galaxy occupies a call
				consume(q.add_call(g) for g in galaxy_instances)
				q.run()
			else:
				self._write_arrays_to_h5_single(galaxy_instances[0])
		except Exception as e:
			#err = e
			traceback.print_exc()
		finally:
			pass
		
		#if err is not None:
		#	raise err
	
	def _write_arrays_to_h5_single(self,galaxy_instance):
		"""
		Lower-level function that writes galaxy data to h5 file.
		
		Do not use directly, otherwise required attributes of the class will not be set;
		instead call 'write_files_to_h5'.
		"""
		print_update(
			f'writing data for {galaxy_instance} at '
			f'{self.get_path_from_instance(galaxy_instance)}'
		)
		
		# the containing h5 Group
		gal_h5_sample_group = self.writer[self.get_path_from_instance(galaxy_instance)]
		
		for attr in saveable_arrays:
			# if the attribute is absent on this instance, skip it
			try:
				data = getattr(galaxy_instance,attr)
			except AttributeAbsent:
				continue
			try:
				# if the h5 Group exists, overwrite its data
				gal_h5_sample_group[attr][:] = data
			except KeyError:
				# otherwise create a new dataset
				# it is assumed that the shape of any given data array is static,
				# hence the 'maxshape' assignment
				gal_h5_sample_group.create_dataset(attr, data = data, maxshape=data.shape)











class H5ResultSet:
	class H5ResultAbsent(AttributeError):
		def __init__(self,value_type,values):
			self.value_type = value_type
			self.values = values
		
		def __str__(self):
			return (
				f"'{self.value_type}' is not in this H5 result set; "
				f'available ones are: {self.values}'
			)
	#__outer__ = Galaxy_H5_Interface
	
	def __init__(self,filenames,attribute_names,values,is_scalar=False):#,array_names,array_values
		"""
		quantity_names: list of quantities in scalar array
		values: 2-d array
			AXES: 0 filename (galaxy), 1 scalar quantity
			the ordering of quantities along axis 0 should match filenames
			the ordering of quantities along axis 1 should match names
		
		(not sure about format of arrays yet)
		"""
		self.filename_mapping = OrderedDict((n,i) for i,n in enumerate(filenames))
		self.attribute_mapping = OrderedDict((n,i) for i,n in enumerate(attribute_names))
		self.values = values
		self.is_scalar = is_scalar
	
	def get(self,filenames=None,attributes=None):
		if filenames is None:
			fi = range(len(self.filename_mapping))
			filenames = list(self.filename_mapping.keys())
		else:
			fi = self._get_filename_indices(filenames)
		
		if attributes is None:
			ai = range(len(self.scalar_mapping))
			attributes = list(self.attribute_mapping.keys())
		else:
			ai = self._get_attribute_indices(attributes)
		
		if self.is_scalar:
			return pd.DataFrame(
				self.values[fi[:,None],ai],
				index=filenames,
				columns = attributes
			)
		else:
			return [[self.values[f][a] for a in ai] for f in fi]
	
	def _get_filename_indices(self,filenames):
		try:
			return np.fromiter(
				(self.filename_mapping[n] for n in filenames),
				int,
				len(filenames)
			)
		except AttributeError:
			raise self.H5ResultAbsent('filename',self.values)
			
	
	def _get_attribute_indices(self,filenames):
		try:
			return np.fromiter(
				(self.attribute_mapping[n] for n in filenames),
				int,
				len(filenames)
			)
		except AttributeError:
			raise self.H5ResultAbsent('attribute',self.values)
	
	def __getattr__(self,attr):
		"""
		equivalent to getting data for all filenames
		in the result set on a single attribute
		"""
		return self.get(self.filenames,[attr])

class H5ScalarResultSet(H5ResultSet):
	"""
	Return scalar results in an array
	"""
	def __init__(self,filenames,names,values):
		"""
		scalar_names: list of quantities in scalar array
		scalars: 2-d array
			AXES: 0 filename (galaxy), 1 scalar quantity
			the ordering of quantities along axis 0 should match filenames
			the ordering of quantities along axis 1 should match scalar_names
		
		(not sure about format of arrays yet)
		"""
		super().__init__(filenames,names,values,True)

class H5GenericResultSet(H5ResultSet):
	"""
	Return array results or mixed scalar/array results
	"""
	def __init__(self,filenames,names,values):
		super().__init__(filenames,names,values,False)
