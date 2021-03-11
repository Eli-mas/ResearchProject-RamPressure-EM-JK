import os
from core import MultiIterator
from asy_io.asy_paths import (MAIN_PROGRAM_DIRECTORY, DATA_SOURCES_PATH, RSIM_FITS_PROCESSED_FILE,
	VSIM_FITS)
from prop.galaxy_file import OTHER, ATLAS3D, sim_fn

"""arrays = frozenset((
	'c_f_per_t', 'center_f', 'center_r', 'center_t', 'center_t_int', 'center_x', 'center_y',
	'centroids', 'data_raw', 'extentlist', 'extentlist_graph', 'extentlist_graph_corrected',
	'extentlist_graph_deproject', 'fluxscore_ar', 'fluxscore_ar_rw', 'fluxscore_ar_trig',
	'fluxscore_ar_trig_rw', 'fsort_inds', 'htscores_ar', 'htscores_ar_rw', 'htscores_ar_trig',
	'htscores_ar_trig_rw', 'iet_ar', 'interior_edge', 'longside_list_graph',
	'longside_list_graph_deproject', 'm2_ext_ratios', 'm2_weights', 'm2ext_avgs', 'm2fluxscore_ar',
	'm2interior_edge', 'm2mM', 'm2score_ar', 'nc_f_per_t', 'nc_f_per_t_rw', 'noncenter_f',
	'noncenter_r', 'noncenter_t', 'noncenter_t_int',
	'noncenter_x', 'noncenter_y', 'outer_pixel_cond', 'pl_ar',
	'pl_ar_deproject', 'plot_list_adjust', 'qfluxscore_ar', 'qfluxscore_ar_rw',
	'qfluxscore_ar_trig', 'qfluxscore_ar_trig_rw', 'qscore', 'qscore_trig',
	'rsort_inds', 'score', 'score_trig', 'shortside_list_graph',
	'shortside_list_graph_deproject', 'total_f', 'total_f_fsort', 'total_f_per_t', 'total_f_per_t_rw',
	'total_f_rsort', 'total_f_tsort', 'total_r', 'total_r_fsort', 'total_r_rsort', 'total_r_tsort',
	'total_t', 'total_t_fsort', 'total_t_int', 'total_t_int_fsort', 'total_t_int_rsort',
	'total_t_int_tsort', 'total_t_rsort', 'total_t_tsort', 'total_x', 'total_x_fsort', 'total_x_rsort',
	'total_x_tsort', 'total_y', 'total_y_fsort', 'total_y_rsort', 'total_y_tsort', 'tsort_inds',
	'wrongside_pixel_cond', 'wrongside_x', 'wrongside_y', 'zdata', 'zdata_regions'
))
"""# these tell which attributes of the Galaxy class are (numpy) arrays and which are not
from .galattr.array_attr import *
from .galattr.nonarray_attr import *

# attributes that pertain to m=2 asymmetry
m2_attributes = frozenset((
	'm2ext_angle','m2ER','m2_FluxAngle','m2FR','m2_weights',
	'extentlist_graph_deproject','shortside_list_graph_deproject',
	'longside_list_graph_deproject','extentlist_graph_corrected',
	'm2ext_avgs','m2fluxscore_ar','m2interior_edge','m2mM','m2score_ar',#'m2ext_data',
	'pl_ar_deproject','m2_ext_ratios','m2nc_f_per_t'
))

# attributes forbidden from high-inclination simulations
high_i_sim_attributes_denied = m2_attributes | observational_only_attributes

# tell a particular Galaxy instance how to behave
meta_attributes=frozenset((
	'compute','debug','save', 'absent','save'
))

# miscellaneous
other_attributes = frozenset(('ratio_process_plotter',))

dynamic_attributes = arrays | non_arrays | other_attributes
all_attributes = frozenset(
	MultiIterator(
		dynamic_attributes, baseline_attributes,
		observational_only_attributes, meta_attributes
	)
)



# these pertain to the weighted tail angle, and variants
centroid_angle_attrs=(
	'centroid_angle','fweighted_centroid_angle',
	'dweighted_centroid_angle','weighted_centroid_angle'
)
centroid_radius_attrs=(
	'centroid_radius','fweighted_centroid_radius',
	'dweighted_centroid_radius','weighted_centroid_radius'
)


sorted_array_attributes = tuple(sorted(arrays))
sorted_nonarray_attributes = tuple(sorted(non_arrays))



def get_file_info(filename):
	"""Returns the information required to access data for a given galaxy:
	'openname' (path to the fits file) and 'filetype' (tells what sample
	the galaxy belongs to). Returned as a tuple of strings."""
	if 'x' in filename or 'y' in filename:
		filetype = 's'
		"""if filename not in sim_fn:
			sim_name_file = open(MAIN_PROGRAM_DIRECTORY+'sim_names.file','a')
			sim_name_file.write(' %s'%filename)
			sim_name_file.close()
			sim_fn.append(filename)"""
		
		"""
		sim_name=DATA_SOURCES_PATH+'sim_fits/'+'sigma_gas_'+filename[0]+'_0'+filename[1:]+'.fits'#os.path.join(os.getcwd(),)
		openname=DATA_SOURCES_PATH+'sim_fits/'+'sigma_gas_'+filename[0]+'_0'+filename[1:]+'_smooth-test.fits'#os.path.join(os.getcwd(),)
		"""
		#sim_name = RSIM_FITS_SOURCE_FILE.format(rtype=filename[0], index=filename[1:])
		openname = RSIM_FITS_PROCESSED_FILE.format(filename)
	elif 'v'==filename[0]:
		filetype = 'vs'
		fcomp = filename.split('_')
		openname = VSIM_FITS+'{group}/smoothed__{gal}.fits'.format(group = '_'.join(fcomp[:-1]),gal=filename)
	else:
		filetype='r'
		if filename in OTHER:
			path1 = DATA_SOURCES_PATH+f'OTHER/{filename}.fits'
			path2 = DATA_SOURCES_PATH+f'OTHER/{filename}.FITS'
		elif filename in ATLAS3D:
			path1 = DATA_SOURCES_PATH+f'ATLAS3D/Serra2012_Atlas3D_Paper13/all_mom0/NGC{filename}_mom0.fits'
			path2 = DATA_SOURCES_PATH+f'ATLAS3D/Serra2012_Atlas3D_Paper13/all_mom0/UGC{filename}_mom0.fits'
		else:
			path1 = DATA_SOURCES_PATH+f'fits files/ngc{filename}.mom0.fits'
			path2 = DATA_SOURCES_PATH+f'fits files/{filename}.mom0.fits'
		openname = path1 if os.path.exists(path1) else path2
	return openname, filetype


if __name__ == '__main__':
	print(all_attributes)
