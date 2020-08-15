"""
Notes:
current alrogithm for beam-related corrections is `correct_for_beam_smearing_new`
	former algorithms may be found in asymmetry_functions_adc_non_decommented_02_19_15.py

02-19-19 removals:
get_outer_border_coors,original_extent_at_index,original_extentlist,
<verbose, non-functionalized version of centroid_angle_calc>,
commented lines in {digitize_outer_flux,get_galaxy_inner_regions,ratio_angle_calc,get_m2_inner_boundary,m2_calc}

02-22-19:
	ratio_angle_calc revised and functionalized;
		fluxscore_ar,htscores_ar now start at angle 0 degrees rather than -90 degrees
	
	!	!	! a note about the `interior_edge_by_shortside` function !	!	!
		at some point, the function had a statement at its beginning:
			>>> interior_edge=np.array(extentlist_graph)
		
		I removed this line, and then had this problem:
			the function assigned the name interior_edge to the array referenced by extentlist_graph,
				and then modified the array in place
			
			thus the array referenced by 'extentlist_graph' became corrupted by this function
			
			the fix: bring back in a statement like the line removed, but use np.copy:
				>>> interior_edge=np.array(extentlist_graph)
		
		IMPORTANT
		Going forward, if I want to make copies of data, do not use np.array
		Use np.copy, it is more explicit and tells me that I am interested in making a copy,
			not converting a non-array to an array

02-23-19:
	extentlist_graph,extentlist_graph_deproject,interior_edge,extentlist_graph_corrected,plot_list_adjust
		rewritten to start at 0 degrees
	
	changes had to be propogated throughout other programs, including:
		adc
		plotting_functions
	
	computation of 'm2ext_data' is removed
"""


import numpy as np; from numpy import searchsorted
from scipy.stats import gmean
from prop.galaxy_file import *
from prop.asy_prop import *
from prop.asy_prop_plot import *
from asy_io.asy_paths import *
from comp.polar_functions import polar_reduction
from comp.computation_functions import (ellipse, c_to_p, reindex, p_to_c,
	deproject_graph, quadrature_sub, quadrature_sum,
	rolling_gmean_wrap, rolling_sum_wrap, sort_arrays_by_array,# rolling_mean_wrap,
	round_to_nearest, znan)#
from common.arrays.roll_wrap_funcs import rolling_mean_wrap
from comp.array_functions import in_sorted, get_region_inds, complex_reindex_2d
from plot.plotting_functions import hline, vline, ax_0N, test_fig_save, full_legend, create_hidden_axis#, fig_size_save
from matplotlib import pyplot as plt
from matplotlib.path import Path as mPath
from prop.asy_defaults import *


def max_ratio_locate(f_set,m,type,pr=False):
	f_set = np.atleast_1d(f_set)
	r_max=np.max(f_set[:,1])
	max_angles=f_set[np.isclose(f_set[:,1],r_max),0]
	if pr: print('-- -- --','max_ratio_locate','r_max',r_max,'max_angles',max_angles, sep='\n')
	return r_max, max_angles

def ratio_angle_index_determine(angle_list,m,mod=a2l,rt=int):
	if m==2: angle_list=angle_list[:len(angle_list)//2]
	if len(angle_list)==1: return angle_list[0]
	return rt(polar_reduction(np.median,angle_list,mod=mod))

def interior_edge_by_pairwise_minima(extentlist_graph,unused):
	extentlist_graph=np.copy(extentlist_graph)
	extents=extentlist_graph[:,1]
	extents_opposite=np.append(extents[al:],extents[:al])
	interior_edge=np.column_stack((extentlist_graph[:,0],np.minimum(extents,extents_opposite)))
	inedge_ar=np.array(interior_edge)
	return interior_edge

def interior_edge_by_shortside(extentlist_graph,EA):
	interior_edge=np.copy(extentlist_graph)
	loc=np.where(np.isclose(interior_edge[:,0]%360,EA%360))[0][0]
	(interior_edge[:,1])[(loc+al+birange_ahl)%a2l]=(interior_edge[:,1])[(loc+birange_ahl)%a2l]
	return interior_edge

#ie_function=interior_edge_by_shortside
ie_function=interior_edge_by_pairwise_minima

if ie_function is interior_edge_by_pairwise_minima: WRONGSIDE_M1=False
else: WRONGSIDE_M1=True

def in2d_lexsort(xc,yc,total_y,total_x_indices,x0,xlast):
	"""
	finds whether (xc,yc) is contained in the total xy coordinate list
	
	this requires that total_x,total_y are lexsorted by x, then y
	i.e. total_xy=xy[lexsort(y,x)]
	"""
	if (xc>xlast) or (xc<x0): return False
	xpos=xc-x0
	
	yslice=total_y[total_x_indices[xpos]:total_x_indices[xpos+1]]
	return in_sorted(yslice,yc)
	

def extent_at_index_optimized(i, xpix, ypix, total_x, total_y, total_x_indices, points):
	"""
	Get the extent of the galaxy at a particular index (angle) by finding
	the outermost point at that angle which is contained in the total xy set
	"""
	xp = np.round(xpix+points*np.cos(i*tau/a2l)).astype('int')
	yp = np.round(ypix+points*np.sin(i*tau/a2l)).astype('int')
	x_unique, y_unique = np.unique(np.array([xp, yp]), axis = 1)
	x0, xlast = total_x[0], total_x[-1]
	contained = np.fromiter(
		(in2d_lexsort(xc, yc, total_y, total_x_indices, x0, xlast)
			for xc,yc in zip(x_unique, y_unique)),
		dtype = bool, count = x_unique.shape[0]
	)
	return np.max(quadrature_sum(x_unique[contained]-xpix,y_unique[contained]-ypix))

def extentlist_optimized(Ginstance):
	"""Optimized version of generating extentlist, i.e.,
	the extent of the galaxy at all angles"""
	xpix,ypix,total_x,total_y=Ginstance.getattrs('xpix','ypix','total_x','total_y')
	points=np.arange(0,np.ceil(np.max(Ginstance.total_r))+1,1)
	total_x_indices=np.append(searchsorted(total_x,np.arange(total_x[0],total_x[-1]+1)),len(total_x))
	res=np.empty([a2l,2],dtype=float)
	res[:,1]=np.fromiter(
		(extent_at_index_optimized(i,xpix,ypix,total_x,total_y,total_x_indices,points)
			for i in range_a2l),
		dtype=float,count=a2l
	)
	res[:,0]=range_a2l*index_to_rad
	return res

def get_grid_xy(xmin=None,xmax=None,ymin=None,ymax=None,x=None,y=None):
	if xmax is None:
		xmin,xmax=int(x.min()),int(np.ceil(x.max()))
		ymin,ymax=int(y.min()),int(np.ceil(y.max()))
	X,Y=np.meshgrid(list(range(xmin,xmax+1)),list(range(ymin,ymax+1)))
	return np.array((X.ravel(),Y.ravel())).T

def points_in_container(c=None,cx=None,cy=None,p=None,px=None,py=None):
	"""
	c=container (the boundary for membership in which to test)
		N x 2 array, columns=(x,y)
		if not provided, cx and cy must be given
	p=points (points to be checked for membership in c)
		N x 2 array, columns=(x,y)
		if not provided, px and py must be given
	
	returns the boolean array telling which points are in the container
	"""
	#print '-- -- -- points_in_container -- -- --'
	if c is None: c=np.column_stack((cx,cy))
	if p is None: p=np.column_stack((px,py))
	#print 'c:'
	#print c
	#print 'p:'
	#print p
	result=mPath(c).contains_points(p)
	#print 'result:'
	#print result
	#print '-- -- -- --'
	return p[result]#

def EA_calc(ar):
	'''Calculate the half-disk extent angle given the appropriate array.'''
	ER=np.max(ar[:,1])
	score_reduced=180+ar[~((ar[:,1]-ER).astype(bool)),0]*rad_to_deg
	EA=round_to_nearest(polar_reduction(np.median,score_reduced,mod=360),1.8)
	return EA,ER

def score_calc(extentratio, aspan, weight=False):
	"""Compute the 'score' array, i.e. the array of the averaged
	extent ratios of the galaxy along all angles. The raw ratios
	along each angle are provided by the 2-d array `extentratio`.
	`aspan` tells the averaging half-window; it is measured in
	the unit of the array index. To convert to degrees, multiply
	it by index_to_deg. `weight` can be used to apply sinusoidal weighting."""
	ar=np.empty([a2l,2],dtype=float)
	ar[:,0]=extentratio[:,0]
	
	if weight: weights=trig_weights[ahl-aspan:ahl+aspan+1]
	else: weights=None
	
	ar[:,1]=rolling_gmean_wrap(extentratio[:,1],aspan,weights=weights)
	
	return ar

extentlist_func=extentlist_optimized

def get_centroid_t_r(xdata, ydata, xpix, ypix, weights=None):
	"""previously named `get_centroid_tr`"""
	return c_to_p(
		np.average(xdata,weights=weights),
		np.average(ydata,weights=weights),
		xpix,ypix
	)

def centroid_angle_calc(Ginstance,xdata,ydata,flux,return_radii=False):
	"""
	calculate the centroid(s) of {xdata,ydata} and convert to polar
	{xdata,ydata} are 1-d arrays;
		they may cover any part of the galaxy (inner, outer, some radial bin, etc.)
	
	sorting by f,r,etc. is not required,
		though the xy coordinates and 'flux' should describe parallel observations
	
	NOTE: consider using 'scipy.spatial.center_of_mass' for optimization
	"""
	filename, xpix, ypix = Ginstance.filename, Ginstance.xpix, Ginstance.ypix
	
	if len(xdata)==0:
		if return_radii: return np.full([2,4],nan)
		return np.full(4,nan)
	
	centroid_angle,centroid_radius=get_centroid_t_r(xdata,ydata,xpix,ypix)
	
	fweighted_centroid_angle,fweighted_centroid_radius=get_centroid_t_r(xdata,ydata,xpix,ypix,weights=flux)
	
	dist=np.sqrt(np.power(xdata-xpix,2) + np.power(ydata-ypix,2))
	dweighted_centroid_angle,dweighted_centroid_radius=get_centroid_t_r(xdata,ydata,xpix,ypix,weights=dist)
	
	weighted_centroid_angle,weighted_centroid_radius=get_centroid_t_r(xdata,ydata,xpix,ypix,weights=flux*dist)
		
	centroid_angles=(centroid_angle, fweighted_centroid_angle, dweighted_centroid_angle, weighted_centroid_angle)
	if not return_radii: return centroid_angles
	radii=(centroid_radius, fweighted_centroid_radius, dweighted_centroid_radius, weighted_centroid_radius)
	return centroid_angles,radii

def galaxy_outer_configure(Ginstance):
	"""Derive and set to attributes of Ginstance the following:
		** 'outer_pixel_cond': which pixels are in the outer galaxy (outer pixels)
			boolean array corresponding to total_r_tsort
		** 'noncenter_x': array of x-values of outer pixels
		** 'noncenter_y': array of y-values of outer pixels
		** 'wrongside_x': array of x-values of wrongside outer pixels
		** 'wrongside_y': array of y-values of wrongside outer pixels
	"""
	filename=Ginstance.filename
	(
	total_t_int_tsort,
	total_r_tsort, total_f_tsort, total_t_tsort, total_f_per_t,
	total_x_tsort, total_y_tsort
	)=Ginstance.getattrs(
		'total_t_int_tsort',
		'total_r_tsort', 'total_f_tsort', 'total_t_tsort', 'total_f_per_t',
		'total_x_tsort', 'total_y_tsort'
	)
	Ginstance.interior_edge=ie_function(Ginstance.extentlist_graph,Ginstance.EA)
	inedge_ar=Ginstance.interior_edge[:,1]/Ginstance.pix_scale_arcsec
	pl_compare=np.copy(Ginstance.plot_list_adjust[:,1])
	
	
	inedge_ar_per_pixel_psa=inedge_ar[total_t_int_tsort]
	pl_compare_per_pixel=pl_compare[total_t_int_tsort]
	
	if WRONGSIDE_M1:
		# wrongside pixels can only be located in regions where
		# outer boundary radii (pl_compare) is recorded as being smaller
		# than inner boundary radii (inedge_ar)
		ie_pl_dif=np.round(pl_compare-inedge_ar)
		wrongside_pixels_possible=np.sum(ie_pl_dif<0)
		if wrongside_pixels_possible:
			#print filename+' pl_compare-inedge_ar: (max neg=%g)'%np.min(ie_pl_dif[ie_pl_dif<0])
			#print pl_compare-inedge_ar
			#print Ginstance.space+filename+': wrongside_pixels_possible'
			
			# get the radii along the interior edge shifted by 180 degrees
			# get the radius along every angle the number of times
			# a pixel occurs at that angle
			inedge_ar_reindex_per_pixel_psa = reindex(inedge_ar, al)[total_t_int_tsort]
			# get the radii along the outer boundary shifted by 180 degrees
			# get the radius along every angle the number of times
			# a pixel occurs at that angle
			pl_compare_reindex_per_pixel = reindex(pl_compare, al)[total_t_int_tsort]
			
			# identify all pixels that are in reversed regions (where
			# radius of outer boundary is less than that of inner boundary)
			reversed_bounds_cond = (pl_compare_reindex_per_pixel < inedge_ar_reindex_per_pixel_psa)
			
			# identify all pixels within the inner boundary
			within_inner_cond = (total_r_tsort <= inedge_ar_reindex_per_pixel_psa)
			# identify all pixels outside the outer boundary
			outside_outer_cond = (total_r_tsort > pl_compare_reindex_per_pixel)
			
			# wrongside pixels are each of the above:
			# inside the inner boundary, outside the outer boundary,
			# and between the two boundaries in reversed order
			wrongside_pixel_cond = reversed_bounds_cond & outside_outer_cond & within_inner_cond
			
			# set to None to clear reference to arrays
			inedge_ar_reindex_per_pixel_psa=pl_compare_reindex_per_pixel=None
		else: wrongside_pixel_cond=np.array([],dtype=int)
	else: wrongside_pixel_cond=np.array([],dtype=int)
	
	# set to None to clear reference to arrays
	pl_compare=inedge_ar=None
	
	#ie_pl_pp_dif=np.round(inedge_ar_per_pixel_psa)-np.round(pl_compare_per_pixel)
	
	# outer pixels are those where the radius of a pixel
	# is greater than the radius of the outer boundary at
	# the pixel's angle
	outer_pixel_cond = total_r_tsort > inedge_ar_per_pixel_psa
	# pixels that sit on the border are deemed inner pixels
	outer_pixel_cond[
		outer_pixel_cond & 
		((np.round(inedge_ar_per_pixel_psa)-np.round(pl_compare_per_pixel))==0)
	]=False
	
	# set to None to clear reference to arrays
	inedge_ar_per_pixel_psa=None
	pl_compare_per_pixel=None
	
	# set Galaxy attributes
	Ginstance.noncenter_x=total_x_tsort[outer_pixel_cond]
	Ginstance.noncenter_y=total_y_tsort[outer_pixel_cond]
	Ginstance.wrongside_x=total_x_tsort[wrongside_pixel_cond]
	Ginstance.wrongside_y=total_y_tsort[wrongside_pixel_cond]
	
	# index into other arrays to identify values describing outer pixels
	noncenter_t_int,noncenter_f,noncenter_t,noncenter_r=[
		ar[outer_pixel_cond] for ar in
		(total_t_int_tsort,total_f_tsort,total_t_tsort,total_r_tsort)
	]
	for suffix,ar in zip(('t_int','t','r','f'),(noncenter_t_int,noncenter_t,noncenter_r,noncenter_f)):
		setattr(Ginstance,'noncenter_'+suffix,ar)
	
	Ginstance.outer_pixel_cond=outer_pixel_cond
	Ginstance.wrongside_pixel_cond=wrongside_pixel_cond

# def get_peripheral_flux(Ginstance,p1=None,p2=None,pr=False):
# 	theta_int,flux,radii=Ginstance.getattrs('total_t_int','total_f','total_r')
# 	theta_int_vals=np.unique(theta_int)
# 	quartiles_per_t=np.empty(a2l,dtype=float)
# 	for i in range(a2l):
# 		cond=np.where(theta_int==i)[0]
# 		fset=flux[cond]
# 		rset=radii[cond]
# 		fset,rset=sort_arrays_by_array(rset,(fset,rset))
# 		try:
# 			if p1 is None: rpos1=0
# 			else: rpos1=np.where(rset>=np.percentile(rset,p1))[0][0]
# 			if p2 is None: rpos2=len(rset)
# 			else: rpos2=np.where(rset>=np.percentile(rset,p2))[0][0]
# 			quartiles_per_t[i]=np.sum(fset[rpos1:rpos2])
# 			if pr:
# 				#print('rset len, rpos, percentile (%s):'%(p),len(rset),rpos,np.percentile(rset,p))
# 				print(rset)
# 		except IndexError: quartiles_per_t[i]=nan
# 	return quartiles_per_t

def compute_region_f_per_t(theta_int,flux):
	"""
	
	
	theta_int, flux have to correspond,
		but are not required to be sorted in any particular way
	
	Rewriting this in numba would be *much* more efficient
	"""
	theta_int_vals=np.unique(theta_int)
	f_per_t=np.zeros(a2l,dtype=float)
	f_per_t[theta_int_vals]=tuple(np.sum(flux[np.where(theta_int==i)[0]]) for i in theta_int_vals)
	return f_per_t

def digitize_outer_flux(Ginstance,return_data=False,test=False,rweight=False):
	"""
	idea for having distance-weight flux ratios (outer half, outer quadrant, global head-tail):
	simply multiply flux value of every pixel by the distance of the pixel
	and compute all quantities the same
	"""
	filename = Ginstance.filename
	outer_pixel_cond = Ginstance.outer_pixel_cond
	wrongside_pixel_cond = Ginstance.wrongside_pixel_cond
	wrongside_pixels_possible = bool(Ginstance.wrongside_x.shape[0])
	"""noncenter_t_int_vals=np.unique(noncenter_t_int)
	nc_f_per_t=np.zeros(a2l)
	nc_f_per_t[noncenter_t_int_vals]=tuple(np.sum(noncenter_f[np.where(noncenter_t_int==i)[0]]) for i in noncenter_t_int_vals)
	noncenter_t_int_vals=None
	assert np.allclose(compute_region_f_per_t(noncenter_t_int,noncenter_f),nc_f_per_t)
	print(filename+' nc_f_per_t assertion passed')
	"""
	
	if rweight:
		flux = Ginstance.noncenter_f * Ginstance.noncenter_r
	else:
		flux = Ginstance.noncenter_f
	# outer flux contained along every angle in range(a2l)
	nc_f_per_t = compute_region_f_per_t(Ginstance.noncenter_t_int, flux)
	
	if wrongside_pixels_possible:
		if test and (not rweight):
			totalflux = Ginstance.totalflux
			ax=plt.subplot()
			ax.plot(nc_f_per_t,label=r'f($\theta$)')
			ax2=ax.twinx()
			ax2.plot(rolling_sum_wrap(nc_f_per_t,ahl)/totalflux,c='cyan',label=r'fsum($\theta$)')
			ax2.plot(return_flux_ratio_score_array(nc_f_per_t)/totalflux,c='b',label=r'r_f($\theta$)')
		
		# 
# 		total_t_int_tsort, total_f_tsort = , 
		wrongside_t_int = Ginstance.total_t_int_tsort[wrongside_pixel_cond]
		wrongside_f = Ginstance.total_f_tsort[wrongside_pixel_cond]
		wrongside_t_int_vals = np.unique(wrongside_t_int)
		nc_f_per_t[(wrongside_t_int_vals+al)%a2l] -= tuple(np.sum(wrongside_f[np.where(wrongside_t_int==i)[0]]) for i in wrongside_t_int_vals)
		
		if test and (not rweight):
			ax.plot(nc_f_per_t,ls='--',label=r'f($\theta$) --w')
			ax2.plot(rolling_sum_wrap(nc_f_per_t,ahl)/totalflux,c='fuchsia',ls='--',label=r'fsum($\theta$) --w')
			ax2.plot(return_flux_ratio_score_array(nc_f_per_t)/totalflux,c='r',ls='--',label=r'r_f($\theta$) --w')
			ax.set_xticklabels(ax.get_xticks()*index_to_deg)
			full_legend(ax,ax2,ncol=2)
			plt.suptitle(filename+' wrongside flux process')
			test_fig_save(filename+' wrongside flux.pdf','wrongside process (flux ratios)',Ginstance.directory)
			plt.clf()
	
	if rweight:
		Ginstance.nc_f_per_t_rw = nc_f_per_t
	else:
		Ginstance.nc_f_per_t = nc_f_per_t
	#if return_data: return nc_f_per_t

def get_galaxy_inner_regions(Ginstance):
	#print 'TEST: inside get_galaxy_inner_regions'
	total_t_int_tsort,total_r_tsort, total_f_tsort, total_t_tsort, total_f_per_t=\
	Ginstance.getattrs(
		'total_t_int_tsort','total_r_tsort',
		'total_f_tsort', 'total_t_tsort', 'total_f_per_t')
	inner_pixel_cond=~Ginstance.outer_pixel_cond
	for attr,ar in zip(('x','y'),(Ginstance.total_x_tsort,Ginstance.total_y_tsort)):
		setattr(Ginstance,'center_'+attr,ar[inner_pixel_cond].astype('int'))
	#Ginstance.center_x=center_x; Ginstance.center_y=center_y
	center_f,center_t,center_r,center_t_int=(
		ar[inner_pixel_cond] for ar in
		(total_f_tsort,total_t_tsort,total_r_tsort,total_t_int_tsort)
	)
	c_f_per_t=compute_region_f_per_t(center_t_int,center_f)
	Ginstance._setattrs_unzip(
		('center_f','center_t','center_r','center_t_int','c_f_per_t'),
		(center_f,center_t,center_r,center_t_int,c_f_per_t)
	)
	"""
	,center_r,center_f=[ar[inner_pixel_cond] for ar in (total_r_tsort,total_f_tsort)]
	center_t_int_vals=np.unique()
	c_f_per_t=np.zeros(a2l)
	c_f_per_t[center_t_int_vals]=tuple(np.sum(center_f[np.where(==i)[0]]) for i in center_t_int_vals)
	"""

def correct_for_beam_smearing_new(r, d_beam_arcsec, mult=1, expand=False, plot=False):#1.1#1.4
	"""as it is used, r=extentlist_graph[:,1]"""
	r_beam_arcsec=d_beam_arcsec/2
	
	"""asymmetry_functions_adc_non_decommented_02_19_15.py contains commented-out former formula"""
	if plot:
		plt.plot(r)
		plt.plot(mult * r_beam_arcsec)
		plt.show()
	return quadrature_sub(r, mult * r_beam_arcsec)

def get_beam_corrected_extentlist(Ginstance,plot=False):
	if Ginstance.is_real:
		beam=ellipse(a=Ginstance.beam_d1,b=Ginstance.beam_d2,PA=Ginstance.beam_PA,inclination=0,d_beam_arcsec=None)
		beam=reindex(beam,ahl,axis=1)
		beam_diameter=beam[1]
	else: beam_diameter=Ginstance.d_beam_arcsec
	
	extentlist_graph=Ginstance.extentlist_graph
	extentlist_graph_corrected=np.column_stack((
		extentlist_graph[:,0],
		correct_for_beam_smearing_new(extentlist_graph[:,1], beam_diameter, plot=plot)
	))
	Ginstance.extentlist_graph_corrected = extentlist_graph_corrected
	
	if plot:
		b=np.array(beam).T
		ax=plt.subplot(polar=True); ax.set_theta_zero_location('N')
		ax.plot(*(b*[deg_to_rad,1.1/2]).T)
		ax.plot(*(Ginstance.extentlist_graph*[deg_to_rad,1]).T)
		plt.title(Ginstance.filename)
		ax.plot(np.linspace(0,tau,100),[Ginstance.d_beam_arcsec*1.1/2.]*100,c='cyan',ls='--')
		plt.show()
	#return np.array(beam)

def return_flux_ratio_score_array(region_f_per_t, sub=True, half=ahl, trig=False):
	"""
	Given an array `region_f_per_t` resulting from 'compute_region_f_per_t',
	compute the result of summing flux over the spcified window at every angle.
	`half` is expressed as an index. `trig` tells whether to apply
	sinusoidal weighting. `sub` controls the operation performed between flux
	on a given side of the galaxy and flux on the opposite side; if true,
	subtraction is performed, otherwise division is performed.
	"""
	if trig: weights=trig_weights[ahl-half:ahl+half+1]
	else: weights=None
	res=rolling_sum_wrap(region_f_per_t,half,weights=weights)
	res_opp=res[(range_a2l+al)%a2l]
	if sub: return res-res_opp
	else: return res/res_opp

def return_asy_score_ar(region_f_per_t,sub=True,half=ahl,trig=False):
	"""Wrapper around `return_flux_ratio_score_array` that embeds
	the result of that function in a 2-d array formatted to the expectations
	of other asymmetry functions."""
	ratio_array=np.empty([a2l,2],dtype=float)
	ratio_array[:,0]=range_a2l
	ratio_array[:,1]=return_flux_ratio_score_array(region_f_per_t,sub=sub,half=half,trig=trig)
	asy_ar=ratio_array
	asy_ar[:,0]%=a2l; asy_ar[:,0]*=index_to_deg
	return asy_ar

def get_ratio_angle_m1(ar):
	"""Given an m=1 asymmetry array with columns <angle>, <quantity>,
	get the maximal ratio and the polar median of the angles matching
	that ratio."""
	R = np.max(ar[:,1])
	A = polar_reduction(np.median, ar[ar[:,1]==R,0]-180, 360)
	return R,A

def ratio_angle_calc(Ginstance, rweight=False, trig=False):
	"""Calculate the half-disk/quadrant outer flux ratios
	and half-disk global-head tail flux ratios, and set as
	attributes of the passed Galaxy instance. `rweight`/`trig`
	tell whether distance-based/sinusoidal weighting is applied
	in calculating the resultant arrays."""
	#print Ginstance.space+'ratio_angle_calc is computing'
	filename,pix_scale_arcsec=Ginstance.getattrs('filename','pix_scale_arcsec')
	
	if rweight:
		totalflux = np.dot(Ginstance.total_f, Ginstance.total_r)
		nc_f_per_t = Ginstance.nc_f_per_t_rw
	else:
		totalflux=Ginstance.totalflux
		nc_f_per_t=Ginstance.nc_f_per_t
	
	fluxscore_ar=return_asy_score_ar(nc_f_per_t,trig=trig)*[1,1/totalflux]
	qfluxscore_ar=return_asy_score_ar(nc_f_per_t,half=aql,trig=trig)*[1,1/totalflux]
	htscores_ar=return_asy_score_ar(
		(Ginstance.total_f_per_t_rw if rweight else Ginstance.total_f_per_t),
		sub=False,trig=trig
	)
	
	FR,FA = get_ratio_angle_m1(fluxscore_ar)
	qFR,qFA = get_ratio_angle_m1(qfluxscore_ar)
	HTR,HTA = get_ratio_angle_m1(htscores_ar)
	attrs=('FA', 'FR', 'qFA', 'qFR', 'HTA', 'HTR')
	if trig: attrs = tuple(a+'_trig' for a in attrs)
	if rweight: attrs = tuple(a+'_rw' for a in attrs)
	values=(FA, FR, qFA, qFR, HTA, HTR)
	Ginstance._setattrs_unzip(attrs, values)
	
	arrays = ('htscores_ar','fluxscore_ar','qfluxscore_ar')
	if trig: arrays = tuple(a+'_trig' for a in arrays)
	if rweight: arrays = tuple(a+'_rw' for a in arrays)
	Ginstance._setattrs_unzip(arrays, (htscores_ar, fluxscore_ar, qfluxscore_ar))

def get_deprojected_extentlist(Ginstance):
	"""Get a deprojected version of the extentlist array."""
	d_beam_arcsec,extentlist_graph,inclination,PA,extentlist_graph_corrected=\
	Ginstance.getattrs('d_beam_arcsec','extentlist_graph','inclination','PA','extentlist_graph_corrected')
	
	egda=np.column_stack(
		(extentlist_graph[:,0],)+
		deproject_graph(extentlist_graph[:,0],extentlist_graph_corrected[:,1],inclination,PA)
	)
	#egda[:,0]=egda[:,0]%a2l
	Ginstance.extentlist_graph_deproject=egda

def get_m2_ext_quantities(Ginstance):#,newcheck=True
	"""Get the m=2 averaged deprojected extents at all angles,
	as well as the min/max of these averaged extents, and
	set as instance attributes."""
	if deny_m2_high_i and Ginstance.inclination>high_i_threshold:
		"""Ginstance.m2score_ar=nan
		Ginstance.m2ER=nan
		Ginstance.m2_ext_weights=nan"""
		#Ginstance.m2ext_data=nan
		return
	#get_deprojected_extentlist(Ginstance)
	egda=Ginstance.extentlist_graph_deproject
	m2ext_avgs=np.array([np.average(egda[np.concatenate([i+np.arange(-aql,aql),i+al+np.arange(-aql,aql)])%a2l,1]) for i in range_a2l])#_new
	m2avg_min=np.min(m2ext_avgs)
	m2avg_max=np.max(m2ext_avgs)
	Ginstance.m2mM=np.array([m2avg_min,m2avg_max])
	Ginstance.m2ext_avgs=m2ext_avgs

def get_m2_inner_boundary(Ginstance):
	"""Get the m=2 inne boundary, expressed in projected radii."""
	if deny_m2_high_i and Ginstance.inclination>high_i_threshold:
		Ginstance.m2interior_edge=nan
		return
	
	#get_m2_ext_quantities(Ginstance)
	inclination,PA,d_beam_arcsec,pix_scale_arcsec,total_t_int_tsort,total_r_tsort=\
	Ginstance.getattrs(
		'inclination','PA','d_beam_arcsec','pix_scale_arcsec','total_t_int_tsort','total_r_tsort'
	)
	m2avg_min=Ginstance.m2mM[0]
	m2ie_angles,m2ie_radii = ellipse(m2avg_min,d_beam_arcsec,inclination,PA,m2=True)
	# if beam_correction: m2ie_radii=correct_for_beam_smearing_new(m2ie_radii,d_beam_arcsec,expand=True)
	# NO: we don't recorrect the ellipse
	m2interior_edge = np.column_stack((m2ie_angles,m2ie_radii))
	m2interior_edge[:,0] *= tau/360
	m2interior_edge = m2interior_edge[np.arange(a2l+1)%a2l]
	Ginstance.m2interior_edge = reindex(m2interior_edge,ahl)
	
	m2inedge_ar = reindex(m2ie_radii/pix_scale_arcsec,ahl)
	m2inedge_ar_per_pixel = m2inedge_ar[total_t_int_tsort]
	m2inner_pixel_cond = total_r_tsort <= m2inedge_ar_per_pixel
	
	return m2inner_pixel_cond, m2inedge_ar, m2inedge_ar_per_pixel

# def get_m2_center_data(Ginstance):
# 	
# 	m2inner_pixel_cond,_,_=get_m2_inner_boundary(Ginstance)
# 	m2center_t_int,m2center_t,m2center_r,m2center_f=[ar[m2inner_pixel_cond] for ar in
# 		Ginstance.getattrs('total_t_int_tsort','total_t_tsort','total_r_tsort','total_f_tsort')
# 	]
# 	"""
# 	m2center_ar=total_ar_tsort[m2inner_pixel_cond].astype('int')
# 	m2center_t_int_vals=np.unique(m2center_t_int)
# 	m2c_f_per_t=np.zeros(a2l)
# 	for i in m2center_t_int_vals: m2c_f_per_t[i]=np.sum(m2center_f[np.where(m2center_t_int==i)[0]])
# 	"""

def get_m2_noncenter_data(Ginstance,check_wrongside=False,test=False):#,newcheck=False
	filename=Ginstance.filename
	total_t_int_tsort,total_f_tsort,total_r_tsort=Ginstance.getattrs('total_t_int_tsort','total_f_tsort','total_r_tsort')
	m2inner_pixel_cond,m2inedge_ar,m2inedge_ar_per_pixel=get_m2_inner_boundary(Ginstance)
	m2outer_pixel_cond=~m2inner_pixel_cond
	#m2noncenter_t_int,m2noncenter_t,m2noncenter_r,m2noncenter_f=[ar[~m2inner_pixel_cond] for ar in (total_t_int_tsort,total_t_tsort,total_r_tsort,total_f_tsort)]
	m2noncenter_t_int,m2noncenter_f=[ar[m2outer_pixel_cond] for ar in (total_t_int_tsort,total_f_tsort)]
	#m2noncenter_t,m2noncenter_r=[ar[~m2inner_pixel_cond] for ar in (total_t_tsort,total_r_tsort)]
	#m2noncenter_ar=total_ar_tsort[~m2inner_pixel_cond].astype('int')
	m2nc_f_per_t=np.zeros(a2l)
	m2noncenter_t_int_vals=np.unique(m2noncenter_t_int).astype(int)
	m2nc_f_per_t[m2noncenter_t_int_vals]=[np.sum(m2noncenter_f[np.where(m2noncenter_t_int==i)[0]]) for i in m2noncenter_t_int_vals]
	#if not check_wrongside:
	#	plt.plot(m2nc_f_per_t)
	#	plt.show()
	
	# now the fun part
	if check_wrongside:
		pl_compare=Ginstance.plot_list_adjust[:,1]
		#if newcheck: 
		#else: pl_compare=reindex(Ginstance.plot_list_adjust[:,1],ahl)
		#print 'm2 wrongside pl check:',np.allclose(pl_compare,pl_compare_new)
		m2ie_pl_dif=pl_compare-m2inedge_ar
		wrongside_pixels_possible=np.sum(m2ie_pl_dif<0)
		if wrongside_pixels_possible:
			"""
			new way of getting the pixels:
			get theta,radii where inner boundary is beyond outer boundary
				DONE: (ar[wrongside_base] for ar in (m2inedge_ar,pl_compare))
			get the x,y coordinates contained within this region
				first, get x,y coordinates along ellipse
				might be helpful to reprogram the `ellipse` function to store the
					major,minor axes (`a`,`b`) as attributes of the Galaxy instance
				then use this information to compute, analytically, the locations
					of xy pixels along the ellipse
				wind up with a series of x and y values used to generate the full pixel array:
					get the inner x, outer x at some particular y
					fill in the space between:
						s=np.sign(x_outer-x_inner)
						np.arange(x_inner,x_outer+s,s)
					do the above for all unique y values in the outer boundary
			"""
			
			wrongside_base=(m2ie_pl_dif<=0)
			
			wrongside_region_inds=get_region_inds(wrongside_base)
			"""
			a list of arrays
				each array in the list contains indices of pixels
			
			each array is azimuthally disconnected with all others (if any),
				while the pixels in any given array are considered part of the same region
				as all pixels in the array
			"""
			
			#wrongside_buondary_x,wrongside_boundary_y
			
			total_t_tsort=Ginstance.total_t_tsort
			PA,inclination,pix_scale_arcsec=Ginstance.getattrs('PA','inclination','pix_scale_arcsec')
			total_rd_tsort=deproject_graph(total_t_tsort*rad_to_deg,total_r_tsort,inclination,PA)[0]
			
			wrongside_cond=wrongside_base[total_t_int_tsort]
			wrongside_theta_int=total_t_int_tsort[wrongside_cond]
			wrongside_radii=total_r_tsort[wrongside_cond]
			wrongside_radii_deproject=total_rd_tsort[wrongside_cond]
			wrongside_theta=total_t_tsort[wrongside_cond]
			
			m2_ie=np.column_stack((np.linspace(0,tau,a2l,endpoint=False),m2inedge_ar))
			pl_c=np.column_stack((np.linspace(0,tau,a2l,endpoint=False),pl_compare))
			xpix,ypix=Ginstance.xpix,Ginstance.ypix
			wrongside_boundaries=(
				np.row_stack((m2_ie[inds],pl_c[inds][::-1]))
				for inds in wrongside_region_inds
				if inds.shape[0]>=1
			)
			"""
			`wrongside_boundaries`: iterable of arrays
				each array consists of (theta,radius) pairs
				these define the boundary of the wrongside pixels in this region
			"""
			wrongside_boundaries=tuple(
				ar[np.arange(ar.shape[0]+1)%ar.shape[0]]
				for ar in wrongside_boundaries
			)
			"""
			extend `wrongside_boundaries` so that the last pair is equal to the first pair
			"""
			wrongside_boundaries_xy=tuple(
				np.column_stack(p_to_c(*ar.T,xc=xpix,yc=ypix))
				for ar in wrongside_boundaries
			)
			"""
			wrongside_boundaries_xy: convert (t,r) in `wrongside_boundaries` to (x,y)
			"""
			wrongside_pixel_groups=tuple(
				points_in_container(c=xy,p=get_grid_xy(x=xy[:,0],y=xy[:,1]))
				for xy in wrongside_boundaries_xy
			)
			"""
			wrongside_pixel_groups: tuple of arrays
				each array contains the points (as [x,y] pairs) within the polygon defined by
				the corresponding array in `wrongside_boundaries_xy`
				uses matplotlib's Path functionality
			"""
			#wrongside_pixel
			
			zdata=Ginstance.zdata
			total_rd_rdsort,total_f_rdsort=sort_arrays_by_array(total_rd_tsort,(total_rd_tsort,total_f_tsort))
			for xy in wrongside_pixel_groups:
				#t,r=c_to_p(xy[:,0],xy[:,1],xpix,ypix)
				#xy=xy/pix_scale_arcsec
				#ax.scatter(tz,r,c=zdata[xy[:,1],xy[:,0]])
				x,y=xy[:,0],xy[:,1]
				#fx,fy=(xpix+xpix-xy[:,0]).round().astype(int),(ypix+ypix-xy[:,1]).round().astype(int)
				#fsub=zdata[fy,fx]
				theta,radii=c_to_p(x,y,xpix,ypix)
				"""t and r of all pixels in the given wrongside group"""
				radii_deproject=deproject_graph(theta,radii,inclination,PA)[0]
				"""
				`radii_deproject`: deprojected radii of all pixels in the given wrongside group
				"""
				
				"""print 'x'
				print x
				print 'y'
				print y
				ax=ax_0N()
				ax.plot(theta,radii,'b+')
				ax.plot(theta,radii_deproject,'r*')
				plt.show()"""
				
				
				wrongside_rd_low_high_inds=np.array([
					np.searchsorted( total_rd_rdsort , radii+r_ , side)
					for r_,side in zip([-.5,.5],('left','right'))
				]).T
				"""
				`wrongside_rd_low_high_inds`: array with two columns of indices
					indices refer to positions apropos `total_rd_rdsort`
					first column contains the lowest indices where pixels can be found
						with radii (.5 pixel distance) less than corresponding radii
						in `radii_deproject`
					second column contains the highest indices where pixels can be found
						with radii (.5 pixel distance) greater than corresponding radii
						in `radii_deproject`
				"""
				wrongside_radii_avg_flux=np.array(tuple(np.average(total_f_rdsort[low:high]) for low,high in wrongside_rd_low_high_inds))
				"""
				`wrongside_radii_avg_flux`: 1-d array
					for every radius in radii_deproject,
					the average flux at all pixels within +-.5 pixel distance of this radius
				"""
				
				theta_int=np.round(theta*rad_to_index).astype(int)
				"""rounded theta values of all pixels in the given wrongside group"""
				t_int_vals=np.unique(theta_int)
				""""print 't_int_vals'
				print t_int_vals
				print 't_int_vals results'
				print tuple(
					np.average(wrongside_radii_avg_flux[theta_int==i])
					for i in t_int_vals
				)
				print 'addend'
				print m2nc_f_per_t[t_int_vals]"""
				
				
				
				m2nc_f_per_t[t_int_vals]-=tuple(
					np.average(wrongside_radii_avg_flux[theta_int==i])
					for i in t_int_vals
				)
			if test:
				"""
				for xy in wrongside_pixel_groups:
					try: plt.scatter(*xy.T)
					except ZeroDivisionError: pass
				test_fig_save(filename+' m2 wrong pixels xy','m2 wrongside process',clf=True)
				"""
				#plt.show()
				#return
				#noregion=0
				plt.plot(*m2_ie.T,c='grey',label='inner')
				plt.plot(*pl_c.T,c='k',label='outer')
				for b in wrongside_boundaries:
					plt.plot(*b.T)#,ls='none',marker='+',markersize=10
					#print b
				#plt.title(noregion)
				test_fig_save(filename+'/'+filename+' m2 wrong pixel regions cartesian','m2 wrongside process',clf=True)
				ax=ax_0N()
				ax.plot(*m2_ie.T,c='k',label='inner')
				ax.plot(*pl_c.T,c=outer_HI_color,label='outer')
				for inds in wrongside_region_inds:
					ax.fill_between(m2_ie[inds,0],m2inedge_ar[inds],pl_compare[inds],color=pos_shading)
				test_fig_save(filename+'/'+filename+' m2 wrong pixel regions','m2 wrongside process',clf=True)
				#plt.show()
				""""""
				#return
				zmax=np.max(zdata)
				print('zmax',zmax)
				#THE FOLLOWING IS WRONG
				"""fig,axes=plt.subplots(1,2,subplot_kw=(dict(polar=True)))
				axes=np.array(axes.flat)
				for i in (0,1):
					ax=axes[i]; ax.set_theta_zero_location('N')
					axi=create_hidden_axis(axes[i],fig=fig)
					#axi=fig.add_axes(ax.get_position())
					#axi.imshow(Ginstance.zdata)
					#ax.plot(wrongside_theta,wrongside_radii,'bo',markersize=5,zorder=-1)
					for xy in wrongside_pixel_groups:
						#t,r=c_to_p(xy[:,0],xy[:,1],xpix,ypix)
						#xy=xy/pix_scale_arcsec
						#ax.scatter(tz,r,c=zdata[xy[:,1],xy[:,0]])
						x,y=xy[:,0],xy[:,1]
						fx,fy=(xpix+xpix-xy[:,0]).round().astype(int),(ypix+ypix-xy[:,1]).round().astype(int)
						if i: axi.scatter(fx,fy,c=zdata[fy,fx],vmin=0,vmax=zmax)#
						else: axi.scatter(x,y,c=zdata[fy,fx],vmin=0,vmax=zmax)#
					axi.axis(Ginstance.get_imshow_lims()); axi.set_aspect('equal')
					#print m2inedge_ar
					ax.plot(*m2_ie.T,zorder=1)
					ax.plot(*pl_c.T,zorder=1)
					#ax.plot(*(tuple(m2_ie[wrongside_base].T)+('r+',)),zorder=2)
					#ax.plot(*(tuple(pl_c[wrongside_base].T)+('g*',)),zorder=2)
					ax.set_ylim(0,Ginstance.edge_radius/pix_scale_arcsec)"""
				#USE THIS INSTEAD
				ax=ax_0N()
				ax.figure.set_size_inches(13,8)
				axi=create_hidden_axis(ax)
				for xy in wrongside_pixel_groups:
					#t,r=c_to_p(xy[:,0],xy[:,1],xpix,ypix)
					#xy=xy/pix_scale_arcsec
					#ax.scatter(tz,r,c=zdata[xy[:,1],xy[:,0]])
					x,y=xy[:,0],xy[:,1]
					fx,fy=(xpix+xpix-xy[:,0]).round().astype(int),(ypix+ypix-xy[:,1]).round().astype(int)
					axi.scatter(x,y,c=zdata[fy,fx],vmin=0,vmax=zmax)#
				axi.axis(Ginstance.get_imshow_lims()); axi.set_aspect('equal')
				axi.plot(*p_to_c(*m2_ie.T,xc=xpix,yc=ypix))
				axi.plot(*p_to_c(*pl_c.T,xc=xpix,yc=ypix))
				#ax.plot(*m2_ie.T,zorder=1)
				#ax.plot(*pl_c.T,zorder=1)
				test_fig_save(filename+'/'+filename+' m2 wrong pixels','m2 wrongside process',clf=True)
				#plt.show()
				
				"""plt.scatter(*p_to_c(m2_ie[:,0][wrongside_base],m2_ie[:,1][wrongside_base],xpix,ypix),marker='+',c='r')
				plt.scatter(*p_to_c(pl_c[:,0][wrongside_base],pl_c[:,1][wrongside_base],xpix,ypix),marker='*',c='g')
				plt.show()
				#print np.column_stack(wrongside_base)
				plt.plot(*m2_ie.T,c='r',label='inner')
				plt.plot(*pl_c.T,c='g',label='outer')
				plt.plot(*m2_ie[wrongside_base].T,c='r',marker='+',ls='none')
				plt.plot(*pl_c[wrongside_base].T,c='g',marker='*',ls='none')
				plt.legend()
				plt.show()"""
	
	quadrant_outer_flux_set=rolling_sum_wrap(m2nc_f_per_t,aql)
	
	"""assertion tests in asymmetry_functions_adc_non_decommented_02_19_15.py"""
	
	return quadrant_outer_flux_set,m2nc_f_per_t

def get_m2_flux_arrays(Ginstance,check_wrongside=True,test=False,mode=1,return_full=False):
	totalflux=Ginstance.totalflux
	qofs,m2nc_f_per_t=get_m2_noncenter_data(Ginstance,check_wrongside=check_wrongside,test=test)
	
	if mode==0:
		qdata=qofs
		abcd_rows=complex_reindex_2d(np.broadcast_to(qdata,[4,a2l]),(0,al,ahl,a1hl),axis=1)
		#a_all,b_all,c_all,d_all=qdata[(range_a2l.reshape(a2l,1)+).T%a2l]
		#ABCD_all=a_all+b_all+c_all+d_all
		a_b,c_d=abcd_rows[:2],abcd_rows[2:]
		
		a_b_sum,c_d_sum=np.sum(a_b,axis=0),np.sum(c_d,axis=0)
		abcd_sum=np.sum(abcd_rows,axis=0)#ABCD_all
		#m2_weights=((a_b_sum*a_b_min)+(c_d_sum*c_d_min))/abcd_sum
		#a_b_min=np.minimum(np.divide(*a_b),np.divide(*a_b[::-1]))
		#c_d_min=np.minimum(np.divide(*c_d),np.divide(*c_d[::-1]))
		a_b_dif=np.abs(np.diff(a_b,axis=0)).squeeze()
		c_d_dif=np.abs(np.diff(c_d,axis=0)).squeeze()
		w_ab=1-(a_b_dif/abcd_sum)
		w_cd=1-(c_d_dif/abcd_sum)
		w_ab[w_ab<0]=0
		w_cd[w_cd<0]=0
		m2_weights=w_ab*w_cd
		m2_fluxscore_unweighted=(a_b_sum-c_d_sum)/totalflux
		m2flux_weighted_all=m2_weights*m2_fluxscore_unweighted
		
		lambda_AB=znan(w_ab,remove_nan=True,inplace=True)
		lambda_CD=znan(w_cd,remove_nan=True,inplace=True)
		abcd_stack=abcd_rows.T
	
	elif mode==1:
		qdata=rolling_mean_wrap(Ginstance.extentlist_graph_deproject[:,1],aql)
		a_all,b_all,c_all,d_all=qdata[(range_a2l.reshape(a2l,1)+(0,al,ahl,a1hl)).T%a2l]
		A_all,B_all=np.minimum(a_all,b_all),np.maximum(a_all,b_all)
		C_all,D_all=np.minimum(c_all,d_all),np.maximum(c_all,d_all)
		rsw_all,wsw_all=.5*A_all/B_all,.5*C_all/D_all
		m2_weights=rsw_all+wsw_all
		
		qflux=qofs[(range_a2l.reshape(1,a2l)+np.vstack((0,al,ahl,a1hl)))%a2l]
		m2_fluxscore_unweighted=(qflux[:2].sum(0)-(qflux[2:].sum(0)))/totalflux
		m2flux_weighted_all=m2_weights*m2_fluxscore_unweighted
		abcd_stack=np.column_stack((a_all,b_all,c_all,d_all))
		
		lambda_AB=lambda_CD=None
	
	if test or return_full: return (
		m2flux_weighted_all,m2_weights,
		m2_fluxscore_unweighted,m2nc_f_per_t,qdata,
		abcd_stack,
		np.column_stack((lambda_AB,lambda_CD))
	)
	
	return m2flux_weighted_all,m2_weights,m2_fluxscore_unweighted

def m2_calc(Ginstance,check_wrongside=True,test=False,mode=1,pr=False):
	if deny_m2_high_i and Ginstance.inclination>high_i_threshold:
		Ginstance.m2score_ar=nan
		Ginstance.m2ER=nan
		
		Ginstance.m2fluxscore_ar=nan
		Ginstance.m2FR=nan
		Ginstance.m2_FluxAngle=nan
		
		return
	"""
	IMPORTANT:
	
	if any issues arise in the m2 calculation, checks and commented-out implementation by the prior,
		non-vectorized, non-refined version can be found in
		'asymmetry_functions_non_decommented_11_12_18.py'
	
	
	NOTES:
		extentlist_graph and egda correspond to the actual graph in astronomical coordinates
		the angles are not measured from E; they are measured from N
		thus the extents along the angles specified in these arrays correspond to the asymmetry plots
	"""
	filename,PA,pix_scale_arcsec=Ginstance.getattrs('filename','PA','pix_scale_arcsec')
	#if test or check_wrongside: print('<><> m2 check here <><>')
	if test:
		m2flux_weighted_all_new,m2_weights_new,m2_fluxscore_unweighted_new,m2nc_f_per_t_new,qofs_new,abcd_new,gammas_new=get_m2_flux_arrays(Ginstance,check_wrongside=True,test=True,mode=mode)
		m2flux_weighted_all,m2_weights,m2_fluxscore_unweighted,m2nc_f_per_t,qofs,abcd,gammas=get_m2_flux_arrays(Ginstance,check_wrongside=False,mode=mode,return_full=True)
		i=1
		for ar in ('m2nc_f_per_t','qofs','abcd','gammas','m2_weights','m2_fluxscore_unweighted','m2flux_weighted_all'):
			if mode==1 and ar in ('m2nc_f_per_t','qofs','gammas'): continue
			lines=plt.plot(np.linspace(0,360,a2l),eval(ar))
			lines.extend(plt.plot(np.linspace(0,360,a2l),eval(ar+'_new'),ls='--'))
			if ar=='abcd' or ar=='gammas':
				ldiv=len(lines)//2
				for l0,l1,char in zip(lines[:ldiv],lines[ldiv:],ar):
					l1.set_color(l0.get_color())
					if ar[0]=='a': l0.set_label(char)
			if ar[0]=='a':
				plt.legend()
				if mode==0: hline(0,ls=':')
			plt.xticks(np.arange(0,405,45))
			plt.title(filename+' '+ar)
			plt.grid(True,axis='x')
			test_fig_save(f'{filename}/{i} - {filename} plot__{ar}','m2 weighting process',f'mode={mode}',clf=True)
			i+=1
		
		if mode==0 and False:
			ax=plt.subplot(); ax2=ax.twinx()
			ax.plot(np.linspace(0,360,a2l),gammas)
			ax2.plot(np.linspace(0,360,a2l),gammas_new,ls='--')
			ax.set_xticks(np.arange(0,405,45))
			ax.grid(True,axis='x')
			test_fig_save(filename+' plot__gammas dual axis','m2 weighting process','mode=%s'%mode,clf=True)
		
	elif check_wrongside:
		m2flux_weighted_all,m2_weights,m2_fluxscore_unweighted = get_m2_flux_arrays(Ginstance,check_wrongside=True)
	else:
		m2flux_weighted_all,m2_weights,m2_fluxscore_unweighted = get_m2_flux_arrays(Ginstance,check_wrongside=True)
	
	m2fluxscore = np.column_stack((range_a2l,m2flux_weighted_all))
	m2ExtentFluxScore, m2flux_angles = max_ratio_locate(m2fluxscore,2,'flux',pr=pr)
	if pr: print('m2fluxscore', m2fluxscore, 'm2flux_angles', m2flux_angles, sep='\n')
	m2fluxscore[:,0] *= index_to_deg
	m2_FluxAngle = m2fluxscore[np.argmax(m2fluxscore[:,1]),0]
	
	m2_ext_ratios=Ginstance.m2_ext_ratios
	"""np.column_stack((
		egda[:,0],
		(
			(egda[:,1]+egda[(range_a2l+al)%a2l,1])
							/
			(egda[(range_a2l+ahl)%a2l,1]+egda[(range_a2l+a1hl)%a2l,1])
		)
	))"""
	
	m2_score = np.column_stack((m2_ext_ratios[:,0],rolling_gmean_wrap(m2_ext_ratios[:,1],aql)))
	
	m2_score_weight = np.column_stack((
		m2_score[(range_a2l+ahl)%a2l,0],
		m2_score[:,1]**m2_weights
	))[(a1hl+range_a2l)%a2l]
	ExtentScore_deproject_weighted = np.max(m2_score_weight[:,1])
	""""""
	
	"""
	#! ! ! would like to get the following to work, but it is not working yet ! ! !
	ExtentScore_deproject_weighted,m2ext_angles=max_ratio_locate(m2_score_weight,2,'ext')
	m2ext_angle=(90+((ratio_angle_index_determine(m2ext_angles,2)-ahl)%a2l)*index_to_deg)%360
	Ginstance.m2ext_angle=m2ext_angle
	"""
	
	
	Ginstance.m2score_ar = m2_score_weight
	Ginstance.m2ER = ExtentScore_deproject_weighted
	Ginstance.m2ext_angle = m2_score_weight[np.argmax(m2_score_weight[:,1]),0]
	""""""
	Ginstance.m2fluxscore_ar = m2fluxscore
	Ginstance.m2FR = m2ExtentFluxScore
	Ginstance.m2_FluxAngle = m2_FluxAngle
	Ginstance.m2_weights = m2_weights
	
	"""Ginstance.m2_a_r_data=Ginstance.getattrs(
		'm2ext_angle','m2_FluxAngle','m2ER','m2FR'
	)"""


__all__ = ('extentlist_func','EA_calc','score_calc','get_centroid_t_r',
'centroid_angle_calc','galaxy_outer_configure',#'get_peripheral_flux',
'compute_region_f_per_t','digitize_outer_flux','get_galaxy_inner_regions',
'correct_for_beam_smearing_new','get_beam_corrected_extentlist',
'return_flux_ratio_score_array','return_asy_score_ar','get_ratio_angle_m1',
'ratio_angle_calc','get_deprojected_extentlist','get_m2_ext_quantities',
'get_m2_inner_boundary','get_m2_noncenter_data',#'get_m2_center_data',
'get_m2_flux_arrays','m2_calc',)

