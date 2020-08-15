"""container for relevant classes"""

from multiprocessing import Pool, cpu_count

from .classes.Galaxy.Galaxy import Galaxy
from .classes.GalaxyCollection import *
from .classes import GalaxyCollection

from cls.classes.h5 import (initialize_galaxy_data_file, 
		Galaxy_H5_Interface, open_galaxy_data_file)


# def compute_all(f, pr=False, ret=False, **kw):
# 	err_attrs=[]
# 	exceptions=[]
# 	if pr: print_update(f)
# 	g=Galaxy(f, compute=True, reload=True, save=True, **kw)
# 	for attr in g.recognized:
# 			if pr: print('-- -- -- -- -- getting attribute: %s -- -- -- -- --'%attr)
# 			try: getattr(g, attr)
# 			except Exception:
# 				exceptions.append(sys.exc_info()[0])
# 				err_attrs.append(attr)
# 	if pr and exceptions:
# 			print()
# 			print(f+' exceptions:')
# 			print(np.column_stack((err_attrs, exceptions)))
# 	# if not pr: print_update('')
# 	if ret: return g

# def recompute_galaxies(s=None):
# 	g = tuple(galaxy_file.galaxies if s is None else s)
# 	p = Pool(min(cpu_count(),len(g)))
# 	
# 	print_update('recomputing galaxies...')
# 	p.map(cls.compute_all_values, g)
# 	
# 	print_update('closing pool...')
# 	p.close()
# 	p.join()
# 	
# 	print_update('')





"""def get_filename_sorter(s):
	d=list(filter(str.isdigit, s))
	if d: return int(d)
	return s.lower()"""


__all__ = ('Galaxy', *GalaxyCollection.__all__,
	'Galaxy_H5_Interface', 'initialize_galaxy_data_file', 'open_galaxy_data_file')
#

if __name__ == '__main__':
	import traceback, sys
	from asy_io.asy_io import print_update
	
	if len(sys.argv)>1: g = sys.argv[1]
	else: g = '4501'
	g = Galaxy(g,compute=True)
	from .classes.Galaxy import galaxy_attribute_information
	for a in galaxy_attribute_information.all_attributes:
		try:
			print_update(f'{g.filename}: getting {a}')
			_=getattr(g,a)
			#print(f' (type={type(_)})')
		except Exception as e:
			print_update(f"error on attribute '{a}':\n")
			print(traceback.format_exc())

