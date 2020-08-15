from functools import reduce

centroid_angle_attrs=(
	'centroid_angle','fweighted_centroid_angle',
	'dweighted_centroid_angle','weighted_centroid_angle'
)
centroid_radius_attrs=(
	'centroid_radius','fweighted_centroid_radius',
	'dweighted_centroid_radius','weighted_centroid_radius'
)

centroid_types=('unweighted','f-weighted','r-weighted','(f,r)-weighted')

m1_attrs_non_extent=('FA', 'FR', 'qFA', 'qFR','HTA','HTR')
m1_attrs_non_extent_rw=tuple(a+'_rw' for a in m1_attrs_non_extent)
m1_attrs_non_extent_trig=tuple(a+'_trig' for a in m1_attrs_non_extent)
m1_attrs_non_extent_trig_rw=tuple(a+'_trig_rw' for a in m1_attrs_non_extent)
m1_arrays_non_extent=('htscores_ar','fluxscore_ar','qfluxscore_ar')
m1_arrays_non_extent_rw=tuple(a+'_rw' for a in m1_arrays_non_extent)
m1_arrays_non_extent_trig=tuple(a+'_trig' for a in m1_arrays_non_extent)
m1_arrays_non_extent_trig_rw=tuple(a+'_trig_rw' for a in m1_arrays_non_extent)

computable_quantities_sets={}
computable_quantities_sets['region_quantities']=(
'total_size','noncenter_size','center_size','totalflux','noncenterflux','centerflux',
)
computable_quantities_sets['m1_ratio_quantities']=(
'ER','FR','HTR','qER','qFR',
)
computable_quantities_sets['m1_angle_quantities']=(
'EA','FA','HTA','qEA', 'qFA','TA'
)
computable_quantities_sets['m2_quantities']=(
'm2ext_angle','m2ER','m2_FluxAngle','m2FR'
)
computable_quantities_sets['m1_nonext_rw_quantities']=m1_attrs_non_extent_rw
computable_quantities_sets['m1_nonext_trig_quantities']=m1_attrs_non_extent_trig
computable_quantities_sets['m1_nonext_trig_rw_quantities']=m1_attrs_non_extent_trig_rw

computable_quantities_sets['centroid_quantities']=centroid_angle_attrs+centroid_radius_attrs+('tail_A','tail_r')

"""computable_quantities_sets['noise_quantities']=(
'A_beam','channel_width','rms_ch_jy_b','rms_for_noise_runs',
)"""

computable_quantities_sets['trig_weighted_quantities']=(
'ER_trig','EA_trig','qER_trig','qEA_trig'
)

computable_quantities_sets['shortsum_quantities']=(
'shortsum','shortsum_lin','shortsum_trig'
)
computable_quantities_counts={k:len(v) for k,v in computable_quantities_sets.items()}

computable_quantities=set(reduce(tuple.__add__,iter(computable_quantities_sets.values())))
computable_quantities_loaders=dict(
	reduce(
		tuple.__add__,
		(tuple((q,qtype) for q in s) for qtype,s in computable_quantities_sets.items())
	)
)


compute_and_setattr={#data computed at every access and set as an attribute
#arrays
'outer_pixel_cond','wrongside_pixel_cond',
'noncenter_f','noncenter_t','noncenter_r','noncenter_x','noncenter_y',
'center_f','center_t','center_r','center_x','center_y','center_t_int',
'wrongside_x','wrongside_y',
'rsort_inds','fsort_inds','tsort_inds',
'total_t_int',
'total_f_per_t','total_f_per_t_rw',
'nc_f_per_t','nc_f_per_t_rw',
'c_f_per_t',
'extentlist_graph_deproject','shortside_list_graph','longside_list_graph',
'data_raw',
'zdata',#'zdata_smooth',
'm2_weights',#'m2_ext_weights',
'score_trig','qscore_trig',
#'peripheral_flux',
#'extentlist_graph_new','plot_list_adjust_new','extentlist_graph_deproject_new',
#'extentlist_graph_corrected_new'#'extents'
}

compute_and_return={#data computed at every access and NOT set as an attribute
#arrays
'total_t_int_rsort','total_t_rsort','total_r_rsort','total_f_rsort','total_x_rsort','total_y_rsort',#'total_ar_rsort',
'total_t_int_fsort','total_t_fsort','total_r_fsort','total_f_fsort','total_x_fsort','total_y_fsort',#'total_ar_fsort',
'total_t_int_tsort','total_t_tsort','total_r_tsort','total_f_tsort','total_x_tsort','total_y_tsort',#'total_ar_tsort',
'zdata_regions',
'extentlist_graph_corrected','shortside_list_graph_deproject','longside_list_graph_deproject',
'iet_ar','pl_ar','pl_ar_deproject',#'iet_ar_deproject'
'm2_ext_ratios'
}


loadable_arrays={#data that is saved to .npy files and accessed as memmaps
#arrays
#'basic_info','real_data','noise_data',
'extentlist','extentlist_graph',
'extentratio','interior_edge','m2ext_avgs',
'm2fluxscore_ar','m2interior_edge','m2mM','m2score_ar',#'m2ext_data',
'plot_list_adjust','qscore','score',
'total_f','total_f_per_t','total_r','total_t','total_x','total_y',
'centroids',
} | set(m1_arrays_non_extent+m1_arrays_non_extent_rw+
		m1_arrays_non_extent_trig+m1_arrays_non_extent_trig_rw)

load_and_setattr = (loadable_arrays | computable_quantities)

settable_attributes = (load_and_setattr | compute_and_setattr)

nonmembers={'__members__','__methods__'}

CORE={
'loadable','compute_q','compute_return','compute_setattr','settable','no_remove',
}
non_removables={
'loaded_data','PA','inclination','edge_radius','xpix','ypix','pix_scale_arcsec','d_beam_arcsec',
'xcenter','ycenter','R25','beam_PA','beam_d1','beam_d2',
#'idir', 'rdir', 'scdir', 'zdir', 'mapdir', 'directory', 'addir', 'bdir',
'filename','openname','filetype',
'is_real','is_rsim','is_vsim',
'is_other','is_atlas3d','is_cluster',
} | CORE# | {'_get__'+attr for attr in load_and_setattr}
recognized_attributes = (settable_attributes | compute_and_return | non_removables)
m2_attributes={'m2ext_angle','m2ER','m2_FluxAngle','m2FR','m2_weights',
	'extentlist_graph_deproject','shortside_list_graph_deproject',
	'longside_list_graph_deproject','extentlist_graph_corrected',
	'm2ext_avgs','m2fluxscore_ar','m2interior_edge','m2mM','m2score_ar',#'m2ext_data',
	'pl_ar_deproject','m2_ext_ratios'
}
recognized_attributes_high_i=recognized_attributes-m2_attributes

"""call_types={}
for s in (
	'compute_and_setattr','computable_quantities','compute_and_return',
	'non_removables','loadable_arrays'
):
	v=eval(s)
	call_types.update({a:s for a in v})"""

basic_info_values=('PA', 'inclination', 'xpix', 'ypix', 'pix_scale_arcsec', 'd_beam_arcsec')
real_data_values=('xcenter', 'ycenter', 'R25', 'beam_PA', 'beam_d1', 'beam_d2')
noise_data_values=('A_beam', 'channel_width', 'rms_ch_jy_b', 'rms_for_noise_runs')

basic_prop = basic_info_values + real_data_values + noise_data_values

all_attributes = recognized_attributes.union(basic_prop)