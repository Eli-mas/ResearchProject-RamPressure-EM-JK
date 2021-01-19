"""This module defines names that correspond to array attributes
on the Galaxy class."""

from core import MultiIterator

m_agnostic = dict(
	center = ('c_f_per_t', 'center_f', 'center_r', 'center_t', 'center_t_int', 'center_x', 'center_y'),
	extentlists = ('extentlist', 'extentlist_graph', 'qscore', 'qscore_trig'),
	extent_ratio_arrays = ('extentratio', 'score', 'score_trig'),
	flux_arrays = ('fluxscore_ar', 'qfluxscore_ar'),
	flux_arrays_alternate=(
		'fluxscore_ar_rw', 'fluxscore_ar_trig', 'fluxscore_ar_trig_rw',
		'qfluxscore_ar_rw','qfluxscore_ar_trig', 'qfluxscore_ar_trig_rw'
	),
	htscores_arrays = ('htscores_ar', 'htscores_ar_rw', 'htscores_ar_trig','htscores_ar_trig_rw'),
	sorters = ('fsort_inds', 'rsort_inds', 'tsort_inds'),
	side_lists = ('shortside_list_graph', 'longside_list_graph'),
	quantity_arrays = ('total_t', 'total_r', 'total_x', 'total_y', 'total_f'),
	quantity_arrays_qsorted = (
		
		'total_f_rsort', 'total_f_tsort', 'total_r_fsort', 'total_r_rsort', 'total_r_tsort',
		'total_t_fsort', 'total_t_int', 'total_t_int_fsort', 'total_t_int_rsort',
		'total_t_int_tsort', 'total_t_rsort', 'total_t_tsort', 'total_x_fsort', 'total_x_rsort',
		'total_f_fsort', 'total_x_tsort', 'total_y_fsort', 'total_y_rsort', 'total_y_tsort'
	),
	boundaries=('plot_list_adjust','iet_ar', 'interior_edge','pl_ar'),
	map_data = ('data_raw', 'zdata', 'zdata_regions'),
	wrongside = ('wrongside_pixel_cond', 'wrongside_x', 'wrongside_y'),
	noncenter=(
		'nc_f_per_t', 'nc_f_per_t_rw', 'noncenter_f',
		'noncenter_r', 'noncenter_t', 'noncenter_t_int',
		'noncenter_x', 'noncenter_y', 'outer_pixel_cond'
	),
	binned_total_flux = ('total_f_per_t', 'total_f_per_t_rw'),
	other = ('centroids',)
)

m2_only = dict(
	corrected_extentlists=('extentlist_graph_corrected','extentlist_graph_deproject'),
	side_lists = ('shortside_list_graph_deproject', 'longside_list_graph_deproject'),
	m2_arrays=('m2_ext_ratios', 'm2score_ar', 'm2fluxscore_ar', 'm2_weights', 'm2ext_avgs', 'm2mM'),
	m2_boundaries = ('m2interior_edge', 'pl_ar_deproject'),
	
	
)



	





arrays = frozenset(
	MultiIterator(
		*m_agnostic.values(),
		*m2_only.values()
	)
)

saveable_arrays = {
	'total_t', 'qscore', 'centroids', 'm2mM', 'qfluxscore_ar_trig',
	'extentlist_graph', 'm2fluxscore_ar', 'm2ext_avgs', 'm2score_ar',
	'htscores_ar_trig', 'score', 'm2interior_edge', 'fluxscore_ar_trig',
	'interior_edge', 'fluxscore_ar', 'qfluxscore_ar', 'total_x', 'total_r',
	'total_f_per_t', 'total_y', 'total_f', 'extentratio', 'htscores_ar',
	'plot_list_adjust', 'extentlist'
}

nonsaveable_arrays = arrays - saveable_arrays

__all__ = ('arrays','saveable_arrays', 'nonsaveable_arrays')


if __name__=='__main__':
	print('arrays',arrays,sep='\n')
	print('saveable_arrays',saveable_arrays,sep='\n')