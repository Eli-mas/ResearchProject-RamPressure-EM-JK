from operator import eq
import numpy as np

from cls.classes import h5
from cls.adc import Galaxy_H5_Interface, Galaxy
from cls.classes.Galaxy.galaxy_attribute_information import *
from cls.classes.Galaxy.Galaxy import AttributeAbsent
from cls.classes.Galaxy import Galaxy_getters

from prop.galaxy_file import * # galaxies
from core import consume, Queuer
from asy_io.asy_io import print_update


def make_load_and_compute_galaxy_instances(filename):
	return Galaxy(filename,compute=False), Galaxy(filename,compute=True)

def verify_scalar_write(galaxies):
	h5i = Galaxy.h5_interface
	scalars = h5i.reader.attrs['scalar quantities']
	recomputed_galaxies = Galaxy.write_h5(galaxies)
	
	f = h5.open_galaxy_data_file('a')
	
	scalar_datasets = {g:f[g]['__scalars__'][()] for g in f.keys()}
	scalar_galaxy_orders_mapping={
		group:{name:i for i,name in enumerate(f[group].attrs['filenames'].astype('U'))}
		for group in f.keys()
	}
	
	for g in recomputed_galaxies:
		results = scalar_datasets[h5i.get_h5_group(g)][scalar_galaxy_orders_mapping[g.filename]]
		assert np.allclose(g.tryattrs(*scalars),results,equal_nan=True)

def assert_matching_quantities(gl, gc, attr, comp=np.allclose):
	"""
	given gl,gc = two Galaxy instances, one set to load,
		the other set to compute (compute=False/True),
	assert that they both yield the same results for a given attribute 'attr'
	"""
	galaxy = gl.filename
	try:
		vl,vc = getattr(gl,attr), getattr(gc,attr)
		if comp is np.allclose: c = comp(vl,vc, equal_nan=True)
		else: c = comp(vl,vc)
		assert c, \
			f"galaxy {galaxy}: the value on attribute {attr} is inconsistent"#loaded:<{vl}>, computed:<{vc}>
		print_update(f'galaxy {galaxy}: comparison on attr {attr} successful')
	except AttributeAbsent: pass
	except (KeyboardInterrupt, SystemExit):
		raise
	except Exception as e:
		print(f'\n\ngalaxy {galaxy}: exception on attribute {attr}:',repr(e))
		print('\tloaded:',vl,sep='\n\t')
		print('\tcomputed:',vc,sep='\n\t')

"""def verify_baseline_match_single(galaxy):
	print_update(galaxy)
	gl, gc = make_load_and_compute_galaxy_instances(galaxy)
	try:
		consume(verify_single_scalar_read_single_galaxy(galaxy,a) for a in baseline_attributes)
	except Exception as e:
		print('\n')
		raise e
	print_update('')

def verify_baseline_q_by_single_match(*galaxies,thread=0):
	if thread:
		# this runs much slower than it should--
		# threading with h5py seems to have hiccups
		q = Queuer(verify_baseline_match_single, iterargs=((g,) for g in galaxies), thread_count=thread)
		q.run()
	else:
		consume(verify_baseline_match_single(g) for g in galaxies)

def verify_single_scalar_read_single_galaxy(galaxy, attr, comp = np.array_equal):#eq
	gl, gc = make_load_and_compute_galaxy_instances(galaxy)
	assert_matching_quantities(gl, gc, attr, comp=comp)

def verify_all_attr_match(galaxy):
	consume(verify_single_scalar_read_single_galaxy(galaxy,attr) for attr in saveable_arrays | non_arrays)
	print_update('')

def verify_all_attr_match_all_galaxies():
	consume(verify_all_attr_match(g) for g in galaxies)
	print_update('')"""

def verify_attr_match(galaxy,attributes=None):
	gl,gc = make_load_and_compute_galaxy_instances(galaxy)
	consume(assert_matching_quantities(gl,gc,a) for a in attributes)
	print_update('')

def verify_attr_match_galaxies(*gals,attributes=None):
	if not gals: gals=sorted(galaxy_samples)
	consume(verify_attr_match(g, attributes=attributes) for g in gals)

def verify_scalars_match(galaxy):
	#consume(verify_single_scalar_read_single_galaxy(galaxy,attr) for attr in non_arrays)
	#print_update('')
	verify_attr_match(galaxy, attributes=non_arrays)

def verify_scalars_match_galaxies(*gals):
	verify_attr_match_galaxies(*gals, attributes=non_arrays)

def verify_arrays_match_galaxies(*gals):
	verify_attr_match_galaxies(*gals, attributes=saveable_arrays)

def verify_all_attrs_match(galaxy):
	verify_attr_match(galaxy, attributes = non_arrays | saveable_arrays)

def verify_all_attrs_match_galaxies(*gals):
	verify_attr_match_galaxies(*gals, attributes = non_arrays | saveable_arrays)




def verify_Galaxy_getters_script():
	import re
	pattern = re.compile(
		'^def (.+?)[(].+[)][:]((?!\sdef\s).|\n)+?return (.+?)$',
		re.MULTILINE
	)
	
	with open(Galaxy_getters.__file__, 'r') as f:
		script = f.read()
	
	matches = (m.groups() for m in pattern.finditer(script))
	matches = [(m[0],m[2]) for m in matches]
	print([m[0] for m in matches])
	for name, returned_name in matches:
		if returned_name != f'self.{name}':
			print(f'function defines attribute {name}'
				  f' but returns {returned_name}')


if __name__=='__main__':
	print('verifying all saveable attributes on all galaxies')
	verify_all_attrs_match_galaxies()
	print('done')