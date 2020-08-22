import numpy as np

from prop.asy_prop import *
from prop.asy_defaults import *
from asy_io.asy_io import fits_data, get_raw_data_from_fits
from comp.contiguous import zdata_ret

from comp.asymmetry_functions import (extentlist_func, get_beam_corrected_extentlist,
	get_deprojected_extentlist, digitize_outer_flux, centroid_angle_calc, score_calc,
	get_m2_inner_boundary, m2_calc)
from comp.computation_functions import reflect_reciprocate
from comp.array_functions import reindex

from plot.plotting_functions import get_edge_radius
from plot.RatioProcessPlotter import RatioProcessPlotter



def A_beam(self):
	fits_data(self)
	return self.A_beam

def EA(self):
	self.compute_m1_ext_asymmetry()
	return self.EA

def EA_trig(self):
	self.compute_m1_ext_asymmetry()
	return self.EA_trig

def ER(self):
	self.compute_m1_ext_asymmetry()
	return self.ER

def ER_trig(self):
	self.compute_m1_ext_asymmetry()
	return self.ER_trig

def FA(self):
	self.get_m1_flux_quantities()
	return self.FA

def FA_rw(self):
	self.get_m1_flux_quantities()
	return self.FA_rw

def FA_trig(self):
	self.get_m1_flux_quantities()
	return self.FA_trig

def FA_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.FA_trig_rw

def FR(self):
	self.get_m1_flux_quantities()
	return self.FR

def FR_rw(self):
	self.get_m1_flux_quantities()
	return self.FR_rw

def FR_trig(self):
	self.get_m1_flux_quantities()
	return self.FR_trig

def FR_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.FR_trig_rw

def HTA(self):
	self.get_m1_flux_quantities()
	return self.HTA

def HTA_rw(self):
	self.get_m1_flux_quantities()
	return self.HTA_rw

def HTA_trig(self):
	self.get_m1_flux_quantities()
	return self.HTA_trig

def HTA_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.HTA_trig_rw

def HTR(self):
	self.get_m1_flux_quantities()
	return self.HTR

def HTR_rw(self):
	self.get_m1_flux_quantities()
	return self.HTR_rw

def HTR_trig(self):
	self.get_m1_flux_quantities()
	return self.HTR_trig

def HTR_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.HTR_trig_rw

def PA(self):
	fits_data(self)
	return self.PA

def R25(self):
	fits_data(self)
	return self.R25

def TA(self):
	self.get_centroid_data()
	return self.TA

def beam_PA(self):
	fits_data(self)
	return self.beam_PA

def beam_d1(self):
	fits_data(self)
	return self.beam_d1

def beam_d2(self):
	fits_data(self)
	return self.beam_d2

def c_f_per_t(self):
	self.compute_inner()
	return self.c_f_per_t

def center_f(self):
	self.compute_inner()
	return self.center_f

def center_r(self):
	self.compute_inner()
	return self.center_r

def center_size(self):
	self.center_size = len(self.center_x)
	return self.center_size

def center_t(self):
	self.compute_inner()
	return self.center_t

def center_t_int(self):
	self.compute_inner()
	return self.center_t_int

def center_x(self):
	self.compute_inner()
	return self.center_x

def center_y(self):
	self.compute_inner()
	return self.center_y

def centerflux(self):
	self.centerflux = np.sum(self.zdata[self.center_y, self.center_x])
	return self.centerflux

def centroid_angle(self):
	self.get_centroid_data()
	return self.centroid_angle

def centroid_radius(self):
	self.get_centroid_data()
	return self.centroid_radius

def centroids(self):
	self.centroids = [rad_to_deg, 1] * np.column_stack(
		centroid_angle_calc(
			self, self.noncenter_x, self.noncenter_y,
			self.noncenter_f, return_radii=True
		)
	)
	return self.centroids

def channel_width(self):
	fits_data(self)
	return self.channel_width

def d_beam_arcsec(self):
	fits_data(self)
	return self.d_beam_arcsec

def data_raw(self):
	get_raw_data_from_fits(self)
	return self.data_raw

def dweighted_centroid_angle(self):
	self.get_centroid_data()
	return self.dweighted_centroid_angle

def dweighted_centroid_radius(self):
	self.get_centroid_data()
	return self.dweighted_centroid_radius

def edge_radius(self):
	self.edge_radius = get_edge_radius(self)
	return self.edge_radius

def extentlist(self):
	self.extentlist = extentlist_func(self)
	return self.extentlist

def extentlist_graph(self):
	self.get_extent_arrays()
	return self.extentlist_graph

def extentlist_graph_corrected(self):
	get_beam_corrected_extentlist(self)
	return self.extentlist_graph_corrected

def extentlist_graph_deproject(self):
	if deny_m2_high_i and self.inclination > high_i_threshold:
		for attr in m2_attributes: setattr(self, attr, None)
	else:
		get_deprojected_extentlist(self)
	return self.extentlist_graph_deproject

def extentratio(self):
	"""generate extentratio, column 0 = angles in radians, column 1 = diametric extent ratios"""
	extentratio = np.empty([a2l, 2], dtype=float)
	extentratio[:, 0], extentratio[:, 1] = (self.extentlist[:, 0] - tau / 4) % tau, reflect_reciprocate(
		self.extentlist[:, 1])
	extentratio = reindex(extentratio, ahl)
	self.extentratio = extentratio
	return self.extentratio

def fluxscore_ar(self):
	self.get_m1_flux_quantities()
	return self.fluxscore_ar

def fluxscore_ar_rw(self):
	self.get_m1_flux_quantities()
	return self.fluxscore_ar_rw

def fluxscore_ar_trig(self):
	self.get_m1_flux_quantities()
	return self.fluxscore_ar_trig

def fluxscore_ar_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.fluxscore_ar_trig_rw

def fsort_inds(self):
	self.setattr_argsort(self.total_f, 'f')
	return self.fsort_inds

def fweighted_centroid_angle(self):
	self.get_centroid_data()
	return self.fweighted_centroid_angle

def fweighted_centroid_radius(self):
	self.get_centroid_data()
	return self.fweighted_centroid_radius

def htscores_ar(self):
	self.get_m1_flux_quantities()
	return self.htscores_ar

def htscores_ar_rw(self):
	self.get_m1_flux_quantities()
	return self.htscores_ar_rw

def htscores_ar_trig(self):
	self.get_m1_flux_quantities()
	return self.htscores_ar_trig

def htscores_ar_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.htscores_ar_trig_rw

def iet_ar(self):
	self.iet_ar = self.interior_edge * [deg_to_rad, 1]
	return self.iet_ar

def inclination(self):
	fits_data(self)
	return self.inclination

def interior_edge(self):
	self.isolate_outer_galaxy()
	return self.interior_edge

def longside_list_graph(self):
	self.get_short_long_sides()
	return self.longside_list_graph

def longside_list_graph_deproject(self):
	self.get_deprojected_sides()
	return self.longside_list_graph_deproject

def m2ER(self):
	self.get_m2_quantities()
	return self.m2ER

def m2FR(self):
	self.get_m2_quantities()
	return self.m2FR

def m2_FluxAngle(self):
	self.get_m2_quantities()
	return self.m2_FluxAngle

def m2_ext_ratios(self):
	egda = self.extentlist_graph_deproject
	self.m2_ext_ratios = np.column_stack((
		egda[:, 0],
		(
				(egda[:, 1] + egda[(range_a2l + al) % a2l, 1])
				/
				(egda[(range_a2l + ahl) % a2l, 1] + egda[(range_a2l + a1hl) % a2l, 1])
		)
	))
	return self.m2_ext_ratios

def m2_weights(self):
	self.get_m2_quantities()
	return self.m2_weights

def m2ext_angle(self):
	if deny_m2_high_i and self.inclination > high_i_threshold:
		for attr in m2_attributes: setattr(self, attr, None)
	else:
		m2_calc(self)
	return self.m2ext_angle

def m2ext_avgs(self):
	self.get_m2_extent_data()
	return self.m2ext_avgs

def m2fluxscore_ar(self):
	self.get_m2_quantities()
	return self.m2fluxscore_ar

def m2interior_edge(self):
	if deny_m2_high_i and self.inclination > high_i_threshold:
		for attr in m2_attributes: setattr(self, attr, None)
	else:
		get_m2_inner_boundary(self)
	return self.m2interior_edge

def m2mM(self):
	self.get_m2_extent_data()
	return self.m2mM

def m2score_ar(self):
	self.get_m2_quantities()
	return self.m2score_ar

def nc_f_per_t(self):
	digitize_outer_flux(self)
	return self.nc_f_per_t

def nc_f_per_t_rw(self):
	digitize_outer_flux(self, rweight=True)
	return self.nc_f_per_t_rw

def noncenter_f(self):
	self.isolate_outer_galaxy()
	return self.noncenter_f

def noncenter_r(self):
	self.isolate_outer_galaxy()
	return self.noncenter_r

def noncenter_size(self):
	self.noncenter_size = len(self.noncenter_x)
	return self.noncenter_size

def noncenter_t(self):
	self.isolate_outer_galaxy()
	return self.noncenter_t

def noncenter_t_int(self):
	self.isolate_outer_galaxy()
	return self.noncenter_t_int

def noncenter_x(self):
	self.isolate_outer_galaxy()
	return self.noncenter_x

def noncenter_y(self):
	self.isolate_outer_galaxy()
	return self.noncenter_y

def noncenterflux(self):
	self.noncenterflux = np.sum(self.zdata[self.noncenter_y, self.noncenter_x])
	return self.noncenterflux

def outer_pixel_cond(self):
	self.isolate_outer_galaxy()
	return self.outer_pixel_cond

def pix_scale_arcsec(self):
	fits_data(self)
	return self.pix_scale_arcsec

def pl_ar(self):
	self.pl_ar = self.plot_list_adjust * [1, self.pix_scale_arcsec]
	return self.pl_ar

def pl_ar_deproject(self):
	self.pl_ar_deproject = self.extentlist_graph_deproject[:, :2] * [deg_to_rad, 1]
	return self.pl_ar_deproject

def plot_list_adjust(self):
	self.get_extent_arrays()
	return self.plot_list_adjust

def qEA(self):
	self.compute_m1_ext_asymmetry()
	return self.qEA

def qEA_trig(self):
	self.compute_m1_ext_asymmetry()
	return self.qEA_trig

def qER(self):
	self.compute_m1_ext_asymmetry()
	return self.qER

def qER_trig(self):
	self.compute_m1_ext_asymmetry()
	return self.qER_trig

def qFA(self):
	self.get_m1_flux_quantities()
	return self.qFA

def qFA_rw(self):
	self.get_m1_flux_quantities()
	return self.qFA_rw

def qFA_trig(self):
	self.get_m1_flux_quantities()
	return self.qFA_trig

def qFA_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.qFA_trig_rw

def qFR(self):
	self.get_m1_flux_quantities()
	return self.qFR

def qFR_rw(self):
	self.get_m1_flux_quantities()
	return self.qFR_rw

def qFR_trig(self):
	self.get_m1_flux_quantities()
	return self.qFR_trig

def qFR_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.qFR_trig_rw

def qfluxscore_ar(self):
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar

def qfluxscore_ar_rw(self):
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar_rw

def qfluxscore_ar_trig(self):
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar_trig

def qfluxscore_ar_trig_rw(self):
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar_trig_rw

def qscore(self):
	self.qscore = score_calc(self.extentratio, aql)
	return self.qscore

def qscore_trig(self):
	self.qscore_trig = score_calc(self.extentratio, aql, weight=True)
	return self.qscore_trig

def rms_ch_jy_b(self):
	fits_data(self)
	return self.rms_ch_jy_b

def rms_for_noise_runs(self):
	fits_data(self)
	return self.rms_for_noise_runs

def rsort_inds(self):
	self.setattr_argsort(self.total_r, 'r')
	return self.rsort_inds

def score(self):
	self.score = score_calc(self.extentratio, ahl)
	return self.score

def score_trig(self):
	self.score_trig = score_calc(self.extentratio, ahl, weight=True)
	return self.score_trig

def shortside_list_graph(self):
	self.get_short_long_sides()
	return self.shortside_list_graph

def shortside_list_graph_deproject(self):
	self.get_deprojected_sides()
	return self.shortside_list_graph_deproject

def shortsum(self):
	self.get_shortsums()
	return self.shortsum

def shortsum_lin(self):
	self.get_shortsums()
	return self.shortsum_lin

def shortsum_trig(self):
	self.get_shortsums()
	return self.shortsum_trig

def tail_A(self):
	self.get_centroid_data()
	return self.tail_A

def tail_r(self):
	self.get_centroid_data()
	return self.tail_r

def total_f(self):
	self.total_f = self.zdata[self.total_y, self.total_x]
	return self.total_f

def total_f_fsort(self):
	self.total_f_fsort = self.total_f[self.fsort_inds]
	return self.total_f_fsort

def total_f_per_t(self):
	self.arrange_total_f_by_t()
	return self.total_f_per_t

def total_f_per_t_rw(self):
	self.arrange_total_f_by_t()
	return self.total_f_per_t_rw

def total_f_rsort(self):
	self.total_f_rsort = self.total_f[self.rsort_inds]
	return self.total_f_rsort

def total_f_tsort(self):
	self.total_f_tsort = self.total_f[self.tsort_inds]
	return self.total_f_tsort

def total_r(self):
	self.compute_total_theta_radii()
	return self.total_r

def total_r_fsort(self):
	self.total_r_fsort = self.total_r[self.fsort_inds]
	return self.total_r_fsort

def total_r_rsort(self):
	self.total_r_rsort = self.total_r[self.rsort_inds]
	return self.total_r_rsort

def total_r_tsort(self):
	self.total_r_tsort = self.total_r[self.tsort_inds]
	return self.total_r_tsort

def total_size(self):
	self.total_size = len(self.total_x)
	return self.total_size

def total_t(self):
	self.compute_total_theta_radii()
	return self.total_t

def total_t_fsort(self):
	self.total_t_fsort = self.total_t[self.fsort_inds]
	return self.total_t_fsort

def total_t_int(self):
	self.total_t_int = (np.round(self.total_t * a2l / tau).astype(int)) % a2l
	return self.total_t_int

def total_t_int_fsort(self):
	self.total_t_int_fsort = self.total_t_int[self.fsort_inds]
	return self.total_t_int_fsort

def total_t_int_rsort(self):
	self.total_t_int_rsort = self.total_t_int[self.rsort_inds]
	return self.total_t_int_rsort

def total_t_int_tsort(self):
	self.total_t_int_tsort = self.total_t_int[self.tsort_inds]
	return self.total_t_int_tsort

def total_t_rsort(self):
	self.total_t_rsort = self.total_t[self.rsort_inds]
	return self.total_t_rsort

def total_t_tsort(self):
	self.total_t_tsort = self.total_t[self.tsort_inds]
	return self.total_t_tsort

def total_x(self):
	self.get_total_xy()
	return self.total_x

def total_x_fsort(self):
	self.total_x_fsort = self.total_x[self.fsort_inds]
	return self.total_x_fsort

def total_x_rsort(self):
	self.total_x_rsort = self.total_x[self.rsort_inds]
	return self.total_x_rsort

def total_x_tsort(self):
	self.total_x_tsort = self.total_x[self.tsort_inds]
	return self.total_x_tsort

def total_y(self):
	self.get_total_xy()
	return self.total_y

def total_y_fsort(self):
	self.total_y_fsort = self.total_y[self.fsort_inds]
	return self.total_y_fsort

def total_y_rsort(self):
	self.total_y_rsort = self.total_y[self.rsort_inds]
	return self.total_y_rsort

def total_y_tsort(self):
	self.total_y_tsort = self.total_y[self.tsort_inds]
	return self.total_y_tsort

def totalflux(self):
	self.totalflux = np.sum(self.total_f)
	return self.totalflux

def tsort_inds(self):
	self.setattr_argsort(self.total_t, 't')
	return self.tsort_inds

def weighted_centroid_angle(self):
	self.get_centroid_data()
	return self.weighted_centroid_angle

def weighted_centroid_radius(self):
	self.get_centroid_data()
	return self.weighted_centroid_radius

def wrongside_pixel_cond(self):
	self.isolate_outer_galaxy()
	return self.wrongside_pixel_cond

def wrongside_x(self):
	self.isolate_outer_galaxy()
	return self.wrongside_x

def wrongside_y(self):
	self.isolate_outer_galaxy()
	return self.wrongside_y

def xcenter(self):
	fits_data(self)
	return self.xcenter

def xpix(self):
	fits_data(self)
	return self.xpix

def ycenter(self):
	fits_data(self)
	return self.ycenter

def ypix(self):
	fits_data(self)
	return self.ypix

def zdata(self):
	self.zdata = zdata_ret(self)
	return self.zdata

def zdata_regions(self):
	zdata_regions = np.array(self.zdata, dtype=complex)
	zdata_regions[self.noncenter_y, self.noncenter_x] *= 1j
	self.zdata_regions = zdata_regions
	return self.zdata_regions


def ratio_process_plotter(self):
	self.ratio_process_plotter = RatioProcessPlotter(self)
	return self.ratio_process_plotter


"""
for input_array_type in sorted(('total_x', 'total_y', 'total_f', 'total_r', 'total_t_int', 'total_t')):
	for sorter_type in sorted(('t', 'r', 'f')):
		print(f'self.{input_array_type}_{sorter_type}sort = self.{input_array_type}[self.{sorter_type}sort_inds]')
		print(f'return self.{input_array_type}_{sorter_type}sort')
"""