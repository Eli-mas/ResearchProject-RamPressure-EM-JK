"""Every function defined in this module obeys four properties:
	(1) it is the name of a data attribute defined on the Galaxy class
	(2) it takes only one argument, which is the Galaxy instance
	(3) it does any work needed to set the data attribute on the instance passed
	(4) it returns the value of that attribute for that instance

The functions in this module are used in combination with the logic
of the `Galaxy` class to allow for dynamically generating data attributes.

Unless otherwise indicated,
	* Angles are expressed in degrees relative to North.
	* Asymmetry angles indicate the angle that minimize
	  (rather than maximize) a quantity of interest.
"""

import numpy as np

from prop.asy_prop import *
from prop.asy_defaults import *
from asy_io.asy_io import fits_data, get_raw_data_from_fits
from comp.contiguous import zdata_ret

from comp.asymmetry_functions import (extentlist_func, get_beam_corrected_extentlist,
	get_deprojected_extentlist, digitize_outer_flux, centroid_angle_calc, score_calc,
	get_m2_inner_boundary, m2_calc, get_m2_noncenter_data)
from comp.computation_functions import reflect_reciprocate
from comp.array_functions import reindex

from plot.plotting_functions import get_edge_radius
from plot.RatioProcessPlotter import RatioProcessPlotter

from common.decorators import add_doc_constants, add_doc

from .galaxy_attribute_information import m2_attributes

def A_beam(self):
	"""beam area"""
	fits_data(self)
	return self.A_beam

def EA(self):
	"""extent angle: minimizes value of 180-degree smoothed
	ratio of extents on opposite sides of the galaxy."""
	self.compute_m1_ext_asymmetry()
	return self.EA

def EA_trig(self):
	"""sinusoidally weighted extent angle: minimizes value of ratio of extents
	on opposite sides of the galaxy convolved with the 180-degree window of
	a cosine signal cenetered on 0."""
	self.compute_m1_ext_asymmetry()
	return self.EA_trig

@add_doc(EA)
def ER(self):
	"""extent ratio: maximal ratio that identifies EA."""
	self.compute_m1_ext_asymmetry()
	return self.ER

@add_doc(EA_trig)
def ER_trig(self):
	"""sinusoidally weighted extent ratio: maximal ratio identifying EA_trig."""
	self.compute_m1_ext_asymmetry()
	return self.ER_trig

def FA(self):
	"""outer flux angle: identifies direction of the maximally negative value
	of the ratio of outer flux on opposite sides of the galaxy, where negative
	flux is taken from flux on the opposite side, and positive flux is taken
	from the given angle."""
	self.get_m1_flux_quantities()
	return self.FA

@add_doc(FA)
def FA_rw(self):
	"""outer flux-weighted-by-radius angle: analogous to FA."""
	self.get_m1_flux_quantities()
	return self.FA_rw

@add_doc(FA, EA_trig)
def FA_trig(self):
	"""sinusoidally weighted outer flux angle: analogous to FA,
	with the same weighting scheme as in EA_trig."""
	self.get_m1_flux_quantities()
	return self.FA_trig

@add_doc(FA_trig, FA)
def FA_trig_rw(self):
	"""sinusoidally weighted outer flux-weighted-by-radius angle,
	analogous to FA_trig and FA."""
	self.get_m1_flux_quantities()
	return self.FA_trig_rw

@add_doc(FA)
def FR(self):
	"""outer flux ratio: maximally positive value of the ratio
	identified by FA."""
	self.get_m1_flux_quantities()
	return self.FR

@add_doc(FA_rw)
def FR_rw(self):
	"""outer flux-weighted-by-radius ratio: maximally positive value
	of the ratio identified by FA_rw."""
	self.get_m1_flux_quantities()
	return self.FR_rw

@add_doc(FA_trig)
def FR_trig(self):
	"""sinusoidally weighted outer flux ratio: maximally positive value
	of the ratio identified by FA_trig."""
	self.get_m1_flux_quantities()
	return self.FR_trig

@add_doc(FA_trig_rw)
def FR_trig_rw(self):
	"""sinusoidally weighted outer flux-weighted-by-radius ratio:
	maximally positive value of the ratio identified by FA_trig_rw."""
	self.get_m1_flux_quantities()
	return self.FR_trig_rw

def HTA(self):
	"""global head-tail flux angle: minimizes value of ratio of flux
	contained on opposite sides of the galaxy."""
	self.get_m1_flux_quantities()
	return self.HTA

@add_doc(HTA)
def HTA_rw(self):
	"""global head-tail flux-weighted-by-radius angle: analogous to HTA"""
	self.get_m1_flux_quantities()
	return self.HTA_rw

@add_doc(HTA, EA_trig)
def HTA_trig(self):
	"""sinusoidally weighted global head-tail flux angle: analgous to HTA,
	with same weighting scheme as in EA_trig."""
	self.get_m1_flux_quantities()
	return self.HTA_trig

@add_doc(HTA_trig, HTA)
def HTA_trig_rw(self):
	"""sinusoidally weighted global head-tail flux-weighted-by-radius angle:
	analgous to HTA_trig and HTA."""
	self.get_m1_flux_quantities()
	return self.HTA_trig_rw

@add_doc(HTA)
def HTR(self):
	"""global head-tail flux ratio: maximal value of the ratio identified
	by HTA"""
	self.get_m1_flux_quantities()
	return self.HTR

@add_doc(HTA_rw)
def HTR_rw(self):
	"""global head-tail flux-weighted-by-radius ratio: maximal value of
	the ratio identified by HTA_rw"""
	self.get_m1_flux_quantities()
	return self.HTR_rw

@add_doc(HTA_trig)
def HTR_trig(self):
	"""sinusoidally weighted global head-tail flux ratio: maximal value of
	the ratio identified by HTA_trig"""
	self.get_m1_flux_quantities()
	return self.HTR_trig

@add_doc(HTA_trig_rw)
def HTR_trig_rw(self):
	"""sinusoidally weighted global head-tail flux-weighted-by-radius ratio: maximal value of
	the ratio identified by HTA_trig_rw"""
	self.get_m1_flux_quantities()
	return self.HTR_trig_rw

def PA(self):
	"""position angle of the galaxy's major axis"""
	fits_data(self)
	return self.PA

def R25(self):
	"""radius measure corresponding to galaxy's stellar disk"""
	fits_data(self)
	return self.R25

def TA(self):
	"""outer weighted tail angle"""
	self.get_centroid_data()
	return self.TA

def beam_PA(self):
	"""position angle of the beam used to take image of a galaxy;
	only meaningful for real galaxies."""
	fits_data(self)
	return self.beam_PA

def beam_d1(self):
	"""first diameter of beam used to take image of a galaxy"""
	fits_data(self)
	return self.beam_d1

def beam_d2(self):
	"""second diameter of beam used to take image of a galaxy"""
	fits_data(self)
	return self.beam_d2

@add_doc_constants(globals(), 'a2l')
def c_f_per_t(self):
	"""array telling how much flux lies along each angle
	in the galaxy's inner region; size = a2l."""
	self.compute_inner()
	return self.c_f_per_t

def center_f(self):
	"""flux for each pixel in the galaxy's inner region"""
	self.compute_inner()
	return self.center_f

def center_r(self):
	"""radius (distance from center) for each pixel in the galaxy's inner region"""
	self.compute_inner()
	return self.center_r

def center_size(self):
	"""number of pixels in galaxy's inner region"""
	self.center_size = len(self.center_x)
	return self.center_size

def center_t(self):
	"""angle (radians) for each pixel in the galaxy's inner region"""
	self.compute_inner()
	return self.center_t

@add_doc_constants(globals(), 'a2l')
def center_t_int(self):
	"""angle index for each pixel in the galaxy's inner region;
	the index is between 0 and a2l."""
	self.compute_inner()
	return self.center_t_int

def center_x(self):
	"""x pixel coordinate for each pixel in the galaxy's inner region"""
	self.compute_inner()
	return self.center_x

def center_y(self):
	"""y pixel coordinate for each pixel in the galaxy's inner region"""
	self.compute_inner()
	return self.center_y

def centerflux(self):
	"""total amount of flux in galaxy inner region"""
	self.centerflux = np.sum(self.zdata[self.center_y, self.center_x])
	return self.centerflux

def centroid_angle(self):
	"""angle of centroid of outer pixels"""
	self.get_centroid_data()
	return self.centroid_angle

def centroid_radius(self):
	"""radius of centroid of outer pixels"""
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
	"""width of channels that made the moment-0 map"""
	fits_data(self)
	return self.channel_width

def d_beam_arcsec(self):
	"""diameter in arcsec of beam used to take image"""
	fits_data(self)
	return self.d_beam_arcsec

def data_raw(self):
	"""unprocessed moment-0 map as 2-d numpy array"""
	get_raw_data_from_fits(self)
	return self.data_raw

def dweighted_centroid_angle(self):
	"""angle of centroid of outer pixels weighted by distance"""
	self.get_centroid_data()
	return self.dweighted_centroid_angle

def dweighted_centroid_radius(self):
	"""radius of centroid of outer pixels weighted by distance"""
	self.get_centroid_data()
	return self.dweighted_centroid_radius

def edge_radius(self):
	"""used for plotting purposes in polar graphs"""
	self.edge_radius = get_edge_radius(self)
	return self.edge_radius

def extentlist(self):
	"""Extent along angles in pixels; rather than relative to North,
	these are expressed relative to the direction pointing right.
	Column 0 = angle (radians), column 1 = extent (pixels)."""
	self.extentlist = extentlist_func(self)
	return self.extentlist

def extentlist_graph(self):
	"""Extents (column 1, arcsec) along angles (column 0, degrees).
	Starts from angle=0 (North)."""
	self.get_extent_arrays()
	return self.extentlist_graph

@add_doc(extentlist_graph)
def extentlist_graph_corrected(self):
	"""Same as extentlist_graph, but with extents corrected for beam smearing"""
	get_beam_corrected_extentlist(self)
	return self.extentlist_graph_corrected

def extentlist_graph_deproject(self):
	"""Similar to extentlist_graph, but second column now contains
	deprojected extents, and a third column is added containing the
	deprojection ratios at each angle."""
	if deny_m2_high_i and self.inclination > high_i_threshold:
		for attr in m2_attributes: setattr(self, attr, None)
	else:
		get_deprojected_extentlist(self)
	return self.extentlist_graph_deproject

def extentratio(self):
	"""column 0 = angles in radians, column 1 = diametric extent ratios"""
	extentratio = np.empty([a2l, 2], dtype=float)
	extentratio[:, 0], extentratio[:, 1] = (self.extentlist[:, 0] - tau / 4) % tau, reflect_reciprocate(
		self.extentlist[:, 1])
	extentratio = reindex(extentratio, ahl)
	self.extentratio = extentratio
	return self.extentratio

def fluxscore_ar(self):
	"""At each angle, outer flux contained in the 180-degree
	sector of the outer galaxy centered at that angle.
	Angles in column 0, flux in column 1. Computes FR, FA."""
	self.get_m1_flux_quantities()
	return self.fluxscore_ar

def fluxscore_ar_rw(self):
	"""At each angle, outer flux weighted by radius contained
	in the 180-degree sector of the outer galaxy centered at that angle.
	Angles in column 0, flux in column 1. Computes FR_rw, FA_rw."""
	self.get_m1_flux_quantities()
	return self.fluxscore_ar_rw

def fluxscore_ar_trig(self):
	"""At each angle, sinusoidally weighted outer flux contained
	in the 180-degree sector of the outer galaxy centered at that angle.
	Angles in column 0, flux in column 1. Computes FR_trig, FA_trig."""
	self.get_m1_flux_quantities()
	return self.fluxscore_ar_trig

def fluxscore_ar_trig_rw(self):
	"""At each angle, sinusoidally weighted outer flux weighted by radius
	contained in the 180-degree sector of the outer galaxy centered at
	that angle. Angles in column 0, flux in column 1. Computes FR_trig_rw,
	FA_trig_rw."""
	self.get_m1_flux_quantities()
	return self.fluxscore_ar_trig_rw

def fsort_inds(self):
	"""Indices which, when applied to the default ordering of all pixels
	in the galaxy, sort them by flux."""
	self.setattr_argsort(self.total_f, 'f')
	return self.fsort_inds

def fweighted_centroid_angle(self):
	"""angle of centroid of outer pixels weighted by flux"""
	self.get_centroid_data()
	return self.fweighted_centroid_angle

def fweighted_centroid_radius(self):
	"""angle of centroid of outer pixels weighted by radius"""
	self.get_centroid_data()
	return self.fweighted_centroid_radius

def htscores_ar(self):
	"""At each angle, ratio of flux contained in the 180-degree
	sector of the galaxy centered at that angle to flux in the
	opposing 180 degree sector. Angles in column 0, flux in column 1.
	Computes HTR, HTA."""
	self.get_m1_flux_quantities()
	return self.htscores_ar

def htscores_ar_rw(self):
	"""At each angle, ratio of flux contained in the 180-degree
	sector of the galaxy centered at that angle to flux in the
	opposing 180 degree sector, where fluxes are weighted by radius
	before taking the ratio. Angles in column 0, flux in column 1.
	Computes HTR_rw, HTA_rw."""
	self.get_m1_flux_quantities()
	return self.htscores_ar_rw

def htscores_ar_trig(self):
	"""At each angle, ratio of sinusoidally weighted flux contained in the
	180-degree sector of the galaxy centered at that angle to flux in the
	opposing 180 degree sector. Angles in column 0, flux in column 1.
	Computes HTR_trig, HTA_trig."""
	self.get_m1_flux_quantities()
	return self.htscores_ar_trig

def htscores_ar_trig_rw(self):
	"""At each angle, ratio of sinusoidally weighted flux contained in the
	180-degree sector of the galaxy centered at that angle to flux in the
	opposing 180 degree sector where fluxes are weighted by radius before
	taking the ratio. Angles in column 0, flux in column 1.
	Computes HTR_trig_rw, HTA_trig_rw."""
	self.get_m1_flux_quantities()
	return self.htscores_ar_trig_rw

def iet_ar(self):
	"""Inner boundary expressed in a 2-d array with each row being a
	[theta (radians), radius (arcsec)]; aligns with interior_edge."""
	self.iet_ar = self.interior_edge * [deg_to_rad, 1]
	return self.iet_ar

def inclination(self):
	"""Viewing angle (inclination) of the galaxy, in the range [0, 90]."""
	fits_data(self)
	return self.inclination

def interior_edge(self):
	"""Inner boundary expressed in a 2-d array with each row being a
	[angle (degrees), radius (arcsec)]. Computed by finding EA
	and reflecting the side of the galaxy's outer boundary centered
	on EA about the galaxy's origin. Distinguishes inner and outer galaxy
	for m=1 analysis."""
	self.isolate_outer_galaxy()
	return self.interior_edge

@add_doc(EA, extentlist_graph)
def longside_list_graph(self):
	"""Returns the portion of extentlist_graph that corresponds
	to the 180-degree region of the galaxy centered on EA+180."""
	self.get_short_long_sides()
	return self.longside_list_graph

@add_doc(longside_list_graph)
def longside_list_graph_deproject(self):
	"""Returns the portion of extentlist_graph_deproject that corresponds
	to the 180-degree region of the galaxy centered on EA+180."""
	self.get_deprojected_sides()
	return self.longside_list_graph_deproject

def m2ER(self):
	"""Maximal value of weighted m=2 extent ratio, analgous to m=1 extent ratio
	but where asymmetry is taken between pairs of opposing quadrants
	rather than between opposing sides."""
	self.get_m2_quantities()
	return self.m2ER

def m2FR(self):
	"""Maximal value of weighted m=2 flux ratio, analgous to m=1 flux ratio
	but where asymmetry is taken between pairs of opposing quadrants
	rather than between opposing sides."""
	self.get_m2_quantities()
	return self.m2FR

@add_doc(m2FR)
def m2_FluxAngle(self):
	"""Direction of maximal location of m2FR; not used."""
	self.get_m2_quantities()
	return self.m2_FluxAngle

def m2_ext_ratios(self):
	"""Computes unweighted, unsmoothed extent ratios at each angle
	as the ratio of average extent in opposing quadrants centered on
	a given angle against that of opposing quadarants 90 degrees away.
	Column 0 = angle (degrees), column 1 = ratio."""
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

def m2nc_f_per_t(self):
	get_m2_noncenter_data(self)
	return self.m2nc_f_per_t

def m2_weights(self):
	"""The weights applied to smoothed m=2 asymmetry values
	at each angle (1d array)."""
	self.get_m2_quantities()
	return self.m2_weights

@add_doc(m2ER)
def m2ext_angle(self):
	"""Direction of maximal location of m2ER; not used."""
	if deny_m2_high_i and self.inclination > high_i_threshold:
		for attr in m2_attributes: setattr(self, attr, None)
	else:
		m2_calc(self)
	return self.m2ext_angle

@add_doc(m2_ext_ratios)
def m2ext_avgs(self):
	"""180-degree smoothed version of the unweighted, unsmoothed
	m=2 extent ratios as calculated by m2_ext_ratios."""
	self.get_m2_extent_data()
	return self.m2ext_avgs

@add_doc(m2_ext_ratios)
def m2fluxscore_ar(self):
	"""Holds weighted m=2 flux values; same regions as in m=2 extent ratio."""
	self.get_m2_quantities()
	return self.m2fluxscore_ar

def m2interior_edge(self):
	"""m=2 interior edge, distinguishing inner and outer galaxy for
	m=2 analysis. Unlike m=1 analysis, this is created as an ellipse."""
	if deny_m2_high_i and self.inclination > high_i_threshold:
		for attr in m2_attributes: setattr(self, attr, None)
	else:
		get_m2_inner_boundary(self)
	return self.m2interior_edge

def m2mM(self):
	"""min/max m=2 extent-ratio values over all angles"""
	self.get_m2_extent_data()
	return self.m2mM

def m2score_ar(self):
	"""Weighted, smoothed m=2 extent ratios."""
	self.get_m2_quantities()
	return self.m2score_ar

def nc_f_per_t(self):
	"""Outer flux contained along each angle (1d array)."""
	digitize_outer_flux(self)
	return self.nc_f_per_t

def nc_f_per_t_rw(self):
	"""Outer flux weighted by radius contained along each angle (1d array)."""
	digitize_outer_flux(self, rweight=True)
	return self.nc_f_per_t_rw

def noncenter_f(self):
	"""Flux for every pixel in the outer galaxy"""
	self.isolate_outer_galaxy()
	return self.noncenter_f

def noncenter_r(self):
	"""Radius (pixels) for every pixel in the outer galaxy"""
	self.isolate_outer_galaxy()
	return self.noncenter_r

def noncenter_size(self):
	"""number of pixels in galaxy's outer region"""
	self.noncenter_size = len(self.noncenter_x)
	return self.noncenter_size

def noncenter_t(self):
	"""Theta (radians) for every pixel in the outer galaxy"""
	self.isolate_outer_galaxy()
	return self.noncenter_t

@add_doc_constants(globals(), 'a2l')
def noncenter_t_int(self):
	"""angle index for each pixel in the galaxy's outer region;
	the index is between 0 and a2l."""
	self.isolate_outer_galaxy()
	return self.noncenter_t_int

def noncenter_x(self):
	"""x pixel coordinate for each pixel in the galaxy's outer region"""
	self.isolate_outer_galaxy()
	return self.noncenter_x

def noncenter_y(self):
	"""y pixel coordinate for each pixel in the galaxy's outer region"""
	self.isolate_outer_galaxy()
	return self.noncenter_y

def noncenterflux(self):
	"""sum of all flux in the outer galaxy"""
	self.noncenterflux = np.sum(self.zdata[self.noncenter_y, self.noncenter_x])
	return self.noncenterflux

def outer_pixel_cond(self):
	"""Boolean array which, when applied to list of all pixels in the galaxy,
	is True for the ones that lie in the outer galaxy."""
	self.isolate_outer_galaxy()
	return self.outer_pixel_cond

def pix_scale_arcsec(self):
	"""Side length of a pixel in arcseconds (arcsec)."""
	fits_data(self)
	return self.pix_scale_arcsec

def pl_ar(self):
	"""Outer boundary in array of [theta (radians), radius (arcsec)] coordinates"""
	self.pl_ar = self.plot_list_adjust * [1, self.pix_scale_arcsec]
	return self.pl_ar

def pl_ar_deproject(self):
	"""Deprojected outer boundary in array of [theta (radians), radius (arcsec)] coordinates"""
	self.pl_ar_deproject = self.extentlist_graph_deproject[:, :2] * [deg_to_rad, 1]
	return self.pl_ar_deproject

def plot_list_adjust(self):
	"""Outer boundary in array of [theta (radians), radius (pixels)] coordinates"""
	self.get_extent_arrays()
	return self.plot_list_adjust

def qEA(self):
	"""quadrant extent angle: minimizes value of 180-degree smoothed
	ratio of extents on opposite sides of the galaxy."""
	self.compute_m1_ext_asymmetry()
	return self.qEA

def qEA_trig(self):
	"""sinusoidally weighted quadrant extent angle: minimizes value of
	ratio of extents on opposite sides of the galaxy convolved with the
	90-degree window of a cosine signal cenetered on 0."""
	self.compute_m1_ext_asymmetry()
	return self.qEA_trig

@add_doc(qEA)
def qER(self):
	"""quadrant extent ratio: maximal ratio that identifies EA."""
	self.compute_m1_ext_asymmetry()
	return self.qER

@add_doc(qEA_trig)
def qER_trig(self):
	"""sinusoidally weighted quadrant extent ratio: maximal ratio
	identifying qEA_trig."""
	self.compute_m1_ext_asymmetry()
	return self.qER_trig

def qFA(self):
	"""outer flux angle: identifies direction of the maximally negative
	value of the ratio of outer flux on opposite quadrants of the galaxy,
	where negative flux is taken from flux on the opposite side, and
	positive flux is taken from the given angle."""
	self.get_m1_flux_quantities()
	return self.qFA

@add_doc(qFA)
def qFA_rw(self):
	"""outer quadrant flux-weighted-by-radius angle: analogous to qFA."""
	self.get_m1_flux_quantities()
	return self.qFA_rw

@add_doc(qFA, qEA_trig)
def qFA_trig(self):
	"""sinusoidally weighted quadrant outer flux angle: analogous to qFA,
	with the same weighting scheme as in qEA_trig."""
	self.get_m1_flux_quantities()
	return self.qFA_trig

@add_doc(qFA_trig, qFA)
def qFA_trig_rw(self):
	"""sinusoidally weighted quadrant outer flux-weighted-by-radius angle,
	analogous to qFA_trig and qFA."""
	self.get_m1_flux_quantities()
	return self.qFA_trig_rw

@add_doc(qFA)
def qFR(self):
	"""quadrant outer flux ratio: maximally positive value of the ratio
	identified by qFA."""
	self.get_m1_flux_quantities()
	return self.qFR

@add_doc(qFA_rw)
def qFR_rw(self):
	"""quadrant outer flux-weighted-by-radius ratio: maximally positive value
	of the ratio identified by qFA_rw."""
	self.get_m1_flux_quantities()
	return self.qFR_rw

@add_doc(qFA_trig)
def qFR_trig(self):
	"""sinusoidally weighted quadrant outer flux ratio: maximally positive value
	of the ratio identified by qFA_trig."""
	self.get_m1_flux_quantities()
	return self.qFR_trig

@add_doc(qFA_trig_rw)
def qFR_trig_rw(self):
	"""sinusoidally weighted quadrant outer flux-weighted-by-radius ratio:
	maximally positive value of the ratio identified by qFA_trig_rw."""
	self.get_m1_flux_quantities()
	return self.qFR_trig_rw

def qfluxscore_ar(self):
	"""At each angle, outer flux contained in the 90-degree
	sector of the outer galaxy centered at that angle.
	Angles in column 0, flux in column 1. Computes FR, FA."""
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar

def qfluxscore_ar_rw(self):
	"""At each angle, outer flux weighted by radius contained
	in the 90-degree sector of the outer galaxy centered at that angle.
	Angles in column 0, flux in column 1. Computes qFR_rw, qFA_rw."""
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar_rw

def qfluxscore_ar_trig(self):
	"""At each angle, sinusoidally weighted outer flux contained
	in the 90-degree sector of the outer galaxy centered at that angle.
	Angles in column 0, flux in column 1. Computes qFR_trig, qFA_trig."""
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar_trig

def qfluxscore_ar_trig_rw(self):
	"""At each angle, sinusoidally weighted outer flux weighted by radius
	contained in the 90-degree sector of the outer galaxy centered at
	that angle. Angles in column 0, flux in column 1. Computes qFR_trig_rw,
	qFA_trig_rw."""
	self.get_m1_flux_quantities()
	return self.qfluxscore_ar_trig_rw

@add_doc(extentratio)
def qscore(self):
	"""90-degree smoothing over `extentratio`. First column
	has angles in radians, second has smoothed extents.
	Computes qER, qEA."""
	self.qscore = score_calc(self.extentratio, aql)
	return self.qscore

@add_doc(extentratio)
def qscore_trig(self):
	"""90-degree smoothing over `extentratio` with a sinusoidal kernel.
	First column has angles in radians, second has smoothed extents.
	Computes qER_trig, qEA_trig."""
	self.qscore_trig = score_calc(self.extentratio, aql, weight=True)
	return self.qscore_trig

def rms_ch_jy_b(self):
	fits_data(self)
	return self.rms_ch_jy_b

def rms_for_noise_runs(self):
	fits_data(self)
	return self.rms_for_noise_runs

def rsort_inds(self):
	"""Indices which, when applied to the default ordering of all pixels
	in the galaxy, sort them by radius (distance from center)."""
	self.setattr_argsort(self.total_r, 'r')
	return self.rsort_inds

@add_doc(extentratio)
def score(self):
	"""180-degree smoothing over `extentratio`. First column
	has angles in radians, second has smoothed extents.
	Computes ER, EA."""
	self.score = score_calc(self.extentratio, ahl)
	return self.score

@add_doc(extentratio)
def score_trig(self):
	"""180-degree smoothing over `extentratio` with a sinusoidal kernel.
	First column has angles in radians, second has smoothed extents.
	Computes ER_trig, EA_trig."""
	self.score_trig = score_calc(self.extentratio, ahl, weight=True)
	return self.score_trig

@add_doc(extentlist_graph, EA)
def shortside_list_graph(self):
	"""The 180-degree window of `extentlist_graph` centered on `EA`."""
	self.get_short_long_sides()
	return self.shortside_list_graph

@add_doc(extentlist_graph_deproject, EA)
def shortside_list_graph_deproject(self):
	"""The 180-degree window of `extentlist_graph_deproject` centered on `EA`."""
	self.get_deprojected_sides()
	return self.shortside_list_graph_deproject

def shortsum(self):
	"""Fraction of shortside angles where radius is shorter
	than on opposite side."""
	self.get_shortsums()
	return self.shortsum

def shortsum_lin(self):
	"""Weighted fraction of shortside angles where radius is shorter
	than on opposite side. Weights: linear kernel with maximum at center."""
	self.get_shortsums()
	return self.shortsum_lin

def shortsum_trig(self):
	"""Weighted fraction of shortside angles where radius is shorter
	than on opposite side. Weights: sinusoidal kernel with maximum at center."""
	self.get_shortsums()
	return self.shortsum_trig

@add_doc(TA)
def tail_A(self):
	"""(`TA`+180)%360"""
	self.get_centroid_data()
	return self.tail_A

@add_doc(TA, tail_A)
def tail_r(self):
	"""Radius of the centroid determining the tail angle.
	Corresponds to `tail_A`."""
	self.get_centroid_data()
	return self.tail_r

def total_f(self):
	"""Flux data for all pixels."""
	self.total_f = self.zdata[self.total_y, self.total_x]
	return self.total_f

def total_f_fsort(self):
	"""Flux data for all pixels sorted by flux."""
	self.total_f_fsort = self.total_f[self.fsort_inds]
	return self.total_f_fsort

@add_doc_constants(globals(), 'a2l')
def total_f_per_t(self):
	"""Flux binned per angle (over `a2l` bins)."""
	self.arrange_total_f_by_t()
	return self.total_f_per_t

@add_doc_constants(globals(), 'a2l')
def total_f_per_t_rw(self):
	"""Radius-weighted flux binned per angle (over `a2l` bins)."""
	self.arrange_total_f_by_t()
	return self.total_f_per_t_rw

def total_f_rsort(self):
	"""Flux data for all pixels sorted by radius."""
	self.total_f_rsort = self.total_f[self.rsort_inds]
	return self.total_f_rsort

def total_f_tsort(self):
	"""Flux data for all pixels sorted by angle."""
	self.total_f_tsort = self.total_f[self.tsort_inds]
	return self.total_f_tsort

def total_r(self):
	"""Radius (unit:pixels) for all pixels."""
	self.compute_total_theta_radii()
	return self.total_r

def total_r_fsort(self):
	"""Radius (unit:pixels) for all pixels, sorted by flux."""
	self.total_r_fsort = self.total_r[self.fsort_inds]
	return self.total_r_fsort

def total_r_rsort(self):
	"""Radius (unit:pixels) for all pixels, sorted by radius."""
	self.total_r_rsort = self.total_r[self.rsort_inds]
	return self.total_r_rsort

def total_r_tsort(self):
	"""Radius (unit:pixels) for all pixels, sorted by angle."""
	self.total_r_tsort = self.total_r[self.tsort_inds]
	return self.total_r_tsort

def total_size(self):
	"""Number of pixels in the galaxy"""
	self.total_size = len(self.total_x)
	return self.total_size

def total_t(self):
	"""Theta (radians) for all pixels."""
	self.compute_total_theta_radii()
	return self.total_t

def total_t_fsort(self):
	"""Theta (radians) for all pixels, sorted by flux."""
	self.total_t_fsort = self.total_t[self.fsort_inds]
	return self.total_t_fsort

@add_doc_constants(globals(), 'a2l')
def total_t_int(self):
	"""Theta index (0 <= i < a2l) for all pixels."""
	self.total_t_int = (np.round(self.total_t * a2l / tau).astype(int)) % a2l
	return self.total_t_int

def total_t_int_fsort(self):
	"""Theta index (0 <= i < a2l) for all pixels, sorted by flux."""
	self.total_t_int_fsort = self.total_t_int[self.fsort_inds]
	return self.total_t_int_fsort

def total_t_int_rsort(self):
	"""Theta index (0 <= i < a2l) for all pixels, sorted by radius."""
	self.total_t_int_rsort = self.total_t_int[self.rsort_inds]
	return self.total_t_int_rsort

def total_t_int_tsort(self):
	"""Theta index (0 <= i < a2l) for all pixels, sorted by theta."""
	self.total_t_int_tsort = self.total_t_int[self.tsort_inds]
	return self.total_t_int_tsort

def total_t_rsort(self):
	"""Theta (radians) for all pixels, sorted by radius."""
	self.total_t_rsort = self.total_t[self.rsort_inds]
	return self.total_t_rsort

def total_t_tsort(self):
	"""Theta (radians) for all pixels, sorted by theta."""
	self.total_t_tsort = self.total_t[self.tsort_inds]
	return self.total_t_tsort

def total_x(self):
	"""x-coordinates for all pixels"""
	self.get_total_xy()
	return self.total_x

def total_x_fsort(self):
	"""x-coordinates for all pixels, sorted by flux."""
	self.total_x_fsort = self.total_x[self.fsort_inds]
	return self.total_x_fsort

def total_x_rsort(self):
	"""x-coordinates for all pixels, sorted by radius."""
	self.total_x_rsort = self.total_x[self.rsort_inds]
	return self.total_x_rsort

def total_x_tsort(self):
	"""x-coordinates for all pixels, sorted by theta."""
	self.total_x_tsort = self.total_x[self.tsort_inds]
	return self.total_x_tsort

def total_y(self):
	"""y-coordinates for all pixels"""
	self.get_total_xy()
	return self.total_y

def total_y_fsort(self):
	"""y-coordinates for all pixels, sorted by flux."""
	self.total_y_fsort = self.total_y[self.fsort_inds]
	return self.total_y_fsort

def total_y_rsort(self):
	"""y-coordinates for all pixels, sorted by radius."""
	self.total_y_rsort = self.total_y[self.rsort_inds]
	return self.total_y_rsort

def total_y_tsort(self):
	"""y-coordinates for all pixels, sorted by theta."""
	self.total_y_tsort = self.total_y[self.tsort_inds]
	return self.total_y_tsort

def totalflux(self):
	"""Sum of all flux in the galaxy."""
	self.totalflux = np.sum(self.total_f)
	return self.totalflux

def tsort_inds(self):
	"""Indices which, when applied to the default ordering of all pixels
	in the galaxy, sort them by theta."""
	self.setattr_argsort(self.total_t, 't')
	return self.tsort_inds

def weighted_centroid_angle(self):
	"""Angle from center of the centroid that defines the tail angle."""
	self.get_centroid_data()
	return self.weighted_centroid_angle

def weighted_centroid_radius(self):
	"""Radius of the centroid that defines the tail angle."""
	self.get_centroid_data()
	return self.weighted_centroid_radius

def wrongside_pixel_cond(self):
	"""Boolean array applied to default pixel ordering that identifies
	which pixels are considered wrongside (True) and not (False)
	for the m=1 analysis."""
	self.isolate_outer_galaxy()
	return self.wrongside_pixel_cond

def wrongside_x(self):
	"""x-coordinates for m=1 wrongside pixels"""
	self.isolate_outer_galaxy()
	return self.wrongside_x

def wrongside_y(self):
	"""y-coordinates for m=1 wrongside pixels"""
	self.isolate_outer_galaxy()
	return self.wrongside_y

def xcenter(self):
	"""RA (right ascension) of the galaxy (in degrees)"""
	fits_data(self)
	return self.xcenter

def xpix(self):
	"""x-coordinate of the galaxy center (unit: pixels)"""
	fits_data(self)
	return self.xpix

def ycenter(self):
	"""DEC (declination) of the galaxy (in degrees)"""
	fits_data(self)
	return self.ycenter

def ypix(self):
	"""y-coordinate of the galaxy center (unit: pixels)"""
	fits_data(self)
	return self.ypix

def zdata(self):
	"""Moment-0 map with background set to 0."""
	self.zdata = zdata_ret(self)
	return self.zdata

def zdata_regions(self):
	"""Moment-0 map with inner and outer regions stored in an array
	with dtype 'complex'. The inner galaxy is stored in the real
	component of the data, the outer galaxy in the imaginary component.
	Where either component is 0, the 0-pixels are not part of the associated
	region."""
	zdata_regions = np.array(self.zdata, dtype=complex)
	zdata_regions[self.noncenter_y, self.noncenter_x] *= 1j
	self.zdata_regions = zdata_regions
	return self.zdata_regions


@add_doc(RatioProcessPlotter)
def ratio_process_plotter(self):
	"""Instance of the `RatioProcessPlotter` class set on
	this Galaxy instance."""
	self.ratio_process_plotter = RatioProcessPlotter(self)
	return self.ratio_process_plotter


"""
for input_array_type in sorted(('total_x', 'total_y', 'total_f', 'total_r', 'total_t_int', 'total_t')):
	for sorter_type in sorted(('t', 'r', 'f')):
		print(f'self.{input_array_type}_{sorter_type}sort = self.{input_array_type}[self.{sorter_type}sort_inds]')
		print(f'return self.{input_array_type}_{sorter_type}sort')
"""