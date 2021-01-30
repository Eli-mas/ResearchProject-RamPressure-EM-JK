"""
Plotting functions for this project.
Note, plotting functionality is incorporated across several scripts.

A number of functions initially defined here were moved to mpl_wrap
because they have general utility.

An important decorator imported from mpl_wrap is makeax--
given a function with a keyword corresponding to a matplotlib
Axes instance (usually 'ax') defaulted to None, makeax
automatically checks to see if the keyword is assigned an axis,
replacing the default value with the result of plt.gca()
at the time the function is called if not.
"""

import os
from shutil import copyfile
#from itertools import cycle
from copy import deepcopy
#import inspect
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np;

from mpl_wrap.plotting_functions import *
from core.core import getattrs

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from prop.ref_rel_data import *
from prop.asy_prop_plot import *
from prop.asy_prop import *

from asy_io.asy_paths import *
from asy_io.asy_io import makepath, get_shortside_data, touch_directory, merge_dicts

from comp.polar_functions import *
from comp.computation_functions import sigfig, find_containing_angle, p_to_c, \
	sig_asy_mask, sig_angle_mask, sig_asy_full_mask

from comp.array_functions import minmax, get_region_inds, take_by_condition, get_regions_from_other


#import matplotlib.animation as manimation
#FFMpegWriter = manimation.writers['ffmpeg']

# def savefigure(figure_path,record=False,fig=None,close=False,clf=False,fmt='pdf',additional_path=None,**kw):
# 	"""Automates actions related to creating a figure:
# 	generating a matplotlib Figure, making required directories,
# 	and closing the plot/figure."""
# 	fmt=fmt.replace('.','')
# 	makepath('/'.join(figure_path.split('/')[:-1]))
# 	#print('figure being saved')
# 	if additional_path is not None:
# 		if isinstance(additional_path,str):
# 			NotImplemented
# 		else:
# 			NotImplemented
# 			#for path in additional_path:
# 			#	...
# 	if fig is None:
# 		fig=plt.gcf()
# 	if figure_path[-len(fmt):]!=fmt:
# 		figure_path=figure_path+'.'+fmt
# 	fig.savefig(figure_path,**kw)
# 	if close:
# 		plt.close(fig)
# 	elif clf:
# 		fig.clf()
# 	if record:
# 		print('function `savefigure`: handling of `record` argument is presently not configured')
# 	return figure_path
# 
# def fig_size_save(path,s=(13,8),resize=True,record=False,fig=None,**kw):
# 	"""Resize a figure before saving it via 'savefigure'."""
# 	if fig is None:
# 		fig=plt.gcf()
# 	if resize:
# 		if not isinstance(resize,bool):
# 			fig.set_size_inches(np.array(fig.get_size_inches())*resize)
# 		else:
# 			fig.set_size_inches(s)
# 	return savefigure(path,record=record,fig=fig,**kw)

def test_fig_save(name,*folders,**k):
	"""Save a figure in the test figure location via 'fig_size_save'."""
	folder_path = os.path.join(*folders)
	path = TEST_FIGURE_LOCATION+os.path.join(folder_path,name)
	kw = merge_dicts({'bbox_inches':'tight'},k)
	return fig_size_save(path, **kw)

def paper_fig_save(name,paper,*folders,**k):
	"""Save a figure in a paper figure location via 'fig_size_save'."""
	name=name.replace(' ','_')
	if isinstance(paper,int):
		#kt=merge_dicts(deepcopy(k),close=False,clf=False)
		return fig_size_save(PAPER_TEX_FIGURE_SAVE.format(paper)+name,**k)#kt
		paper=('paper %i'%paper)
	return fig_size_save(PAPER_FIGURE_LOCATION+os.path.join(os.path.join(paper,*folders),name),**k)

def topical_fig_save(name,*folders,**k):
	"""Save a figure in the topical figure location via 'fig_size_save'."""
	folder_path=os.path.join(*folders)
	path=TOPICAL_FIGURE_LOCATION+os.path.join(folder_path,name)
	kw=merge_dicts({'bbox_inches':'tight','clf':True},k)
	return fig_size_save(path,**kw)

def active_fig_save(name,*folders,**k):
	"""Save a figure in the active figure location via 'fig_size_save'."""
	if folders:
		folder_path=os.path.join(*folders)
		path=ACTIVE_FIGURE_LOCATION+os.path.join(folder_path,name)
	else: path=ACTIVE_FIGURE_LOCATION+name
	kw=merge_dicts({'bbox_inches':'tight','clf':True},k)
	return fig_size_save(path,**kw)

def paper_tex_save(name, paper, *, link_folder=None, touch=0, dump=None, **k):
	"""Save a figure in a paper figure location via 'fig_size_save'.
	Simultaneously create or replace a symlink to the resultant image file
	in a directory reserved for such symlinks, and update the mtime
	file attribute of the directories containing the symlink
	by way of 'touch_directory', up to the top-level directory
	delegated for these symlinks."""
	name=name.replace(' ','_')
	kw=merge_dicts({'bbox_inches':'tight','clf':True}, k)
	paper_tex_path, paper_path = (PAPER_TEX_FIGURE_SAVE, PAPER_PATH) \
		if isinstance(paper,int) or paper.isdigit() else \
		(PAPER_TEX_FIGURE_SAVE_UNNUMBERED, PAPER_PATH_UNNUMBERED)
	
	path=fig_size_save(paper_tex_path.format(paper)+name.split('/')[-1],**kw)
	if dump is not None:
		if isinstance(dump,int) or dump.isdigit():
			dump_loc = PAPER_DUMP_FIGURE_SAVE
		else:
			dump_loc = PAPER_DUMP_FIGURE_SAVE_UNNUMBERED
		
		copyfile(path, f'{dump_loc.format(dump)}{path.split("/")[-1]}')
	
	if link_folder is not None:
		if not isinstance(link_folder,(str,os.PathLike)):
			link_folder=os.path.join(*map(str,link_folder))
		if '/' in name:
			nameparts = name.split('/')
			link_folder = os.path.join(link_folder,'/'.join(nameparts[:-1]))
			name = nameparts[-1]
		folder = makepath(os.path.join(
			paper_path.format(paper),
			'figure links',
			link_folder
		))
		dest=os.path.join(folder,path.split('/')[-1])
		
		try:
			os.symlink(path,dest)
		except FileExistsError:
			os.unlink(dest)
			os.symlink(path,dest)

		relative_path_from_figure_links = os.path.join(link_folder,path.split('/')[-1])
		touch = relative_path_from_figure_links.count('/') - 1
		#print(f'paper_tex_save: touch = {touch}')

		folder = Path(folder)
		for directory in islice(folder.parents,touch):
			#print(f'paper_tex_save: directory will be touched: {directory}')
			touch_directory(directory)
	
	
	return path


misc_fig_save = partial(paper_tex_save,link_folder='--misc',close=True)

def saveplot(func,savefunc=fig_size_save,show=False):
	"""
	Baseline decorator for automatically saving the
	plot generated by a given function to a location
	affiliated with a specific paper.
	"""
	def wrapper(*a,path=None,save_args=None,save_kw={},**k):
		#print('calling `%s`'%func.__name__)
		#print('*a:',a)
		#print('**k',k)
		#print('clf=%s'%clf)
		results=func(*a,**k)
		#for key in ('clf','close'): k.setdefault(key,True)
		#save_kw.update(dict(clf=clf,close=close))
		if path is not None:
			savefunc(path,**save_kw)
		elif save_args:
			savefunc(*save_args,**save_kw)
		elif show: plt.show()
		#print('end of wrapper func\n')
		return results
	#print('returning wrapper')
	return wrapper
paperplot=partial(saveplot,savefunc=paper_tex_save)
paperplot1=partial(saveplot,savefunc=partial(paper_tex_save,paper=1))
paperplot2=partial(saveplot,savefunc=partial(paper_tex_save,paper=2))

def plot_shortside(Ginstance, ax=None):
	"""Plot for inspecting the extents along the shortside of a galaxy."""
	if ax is None: ax=plt.gca(polar=False)
	
	shortside_show,shortside_median=get_shortside_data(Ginstance, deny_m2_high_i=True)
	
	PA_plot=-90+(Ginstance.PA-shortside_show[0,0])%360
	if PA_plot>90: PA_plot-=180
	ax.plot(360/a2l*np.arange(0,len(shortside_show))-90,shortside_show[:,1],c='green')
	ax.set_xlim(-95,95)
	if Ginstance.inclination>high_i_threshold:
		ax.set_xlabel(extent_plot_xlabel)
		ax.set_ylabel(extent_plot_ylabel_highi)
		low,high=int(np.min(shortside_show[:,1])),int(np.max(shortside_show[:,1]))
		ax.set_yticks([low,high])
		ax.set_yticklabels([low,high])
	else:
		ax.set_xlabel(extent_plot_xlabel)
		ax.set_ylabel(extent_plot_ylabel_lowi)
		ax.set_ylim(0.3,2.2)
		ax.set_yticks([0.3,1.25,2.2])
		ax.set_yticklabels([0.3,1.25,2.2])
	ax.bar(PA_plot, keepax(ax=ax)[-1], color=[0,1,0], edgecolor='none')

'''what follows is for `radial_distribution`'''
#include 4921
#ic3392 included as 3392
symmetric=['4192', '4216', '4222', '4298', '4380', '4394', '4405', '4450', '4548', '4561', '4580', '4606', '4607', '4689', '4713', '4772', '3392']
asymmetric_low=['4064', '4189', '4351', '4321', '4424', '4457', '4501', '4532', '4535', '4536', '4569', '4579', '4568', '4567', '4254', '4694', '4651', '4654', '4808', '4698', '4383']#, '4921'
asymmetric_high=['4302', '4330', '4388', '4396', '4402', '4522', '4533']
high_i_cutoff_list={
'4302':180,
'4330':100,
'4388':60,
'4396':160,#very different from r_trun in viva-hi-distributions
'4402':80,
'4522':53,#significantly different from r_trun in viva-hi-distributions
'4533':75
}
def eff_rad_calc(region, f_eff=.9):
	"""Calculate the effective gas radii for gas in a given region."""
	radii,density=region.T
	HI_flux=np.trapz(density,x=radii)
	eff_ind=np.searchsorted(
		[np.trapz(density[:i+1],x=radii[:i+1]) >= f_eff*HI_flux
		for i in range(len(radii))]
	, True)
	pre_r_eff,d_pre_r_eff=radii[[eff_ind-1,eff_ind]],density[[eff_ind-1,eff_ind]]
	r_eff=np.interp(f_eff,d_pre_r_eff,pre_r_eff)
	d_eff=np.interp(r_eff,pre_r_eff,d_pre_r_eff)
	return r_eff,d_eff

def radial_distribution(Ginstance, ax=None, ylabel=True,
						text_right=1, text_top=.99, text_hspace=.2):
	"""
	Get the global radial distribution of flux
	"""
	filename,R25,=Ginstance.filename,Ginstance.R25
	_,shortside_median=get_shortside_data(Ginstance, deny_m2_high_i=True)
	
	if ax is None: ax=plt.gca(polar=False)
	if '3392' in filename: galfile, galaxy = DATA_SOURCES_PATH+'radial profiles/ic03392.both', '3392'
	else: galfile, galaxy = DATA_SOURCES_PATH+'radial profiles/ngc'+filename+'.both', filename
	
	with open(galfile) as f: filelines=f.readlines()
	filelines=np.array(tuple(fline.strip().split() for fline in filelines[2:]),dtype=float)
	radius,density=filelines.T
	
	if galaxy in symmetric:
		ax.plot(radius,density,color=[0,0,1],lw=rd_width)
		inner=filelines
	else:
		if galaxy in asymmetric_low: cutoff=(shortside_median/60)/R25
		elif galaxy in asymmetric_high: cutoff=(float(high_i_cutoff_list[galaxy])/60)/R25
		
		div_index=np.searchsorted(radius>=cutoff,True)
		inner=filelines[:div_index]
		
		ax.plot(radius[:(div_index+1)],density[:(div_index+1)],'-',color=[0,0,1],lw=rd_width)
		ax.plot(radius[(div_index-1):],density[(div_index-1):],'--',color=[0,0,1],lw=rd_width)
	
	if ylabel:
		ax.set_ylabel('HI surface density $[M_\odot/pc^2]$',
					  fontsize=20, labelpad = ax3_labelpad, rotation=270,
					  va='top')
		ax.yaxis.set_label_position("right")
# 	ax.tick_params(axis='y', which='major', pad=ax3_tickpad)
	ax.set_xlabel(r'$r/R_{25}$',fontsize=20)
	
	ax.axis([0,5,0.1,27])
	ax.set_yscale('log')
	
	text_kw=dict(fontsize=17,ha='right',va='top',transform=ax.transAxes)
	if '3392' in filename: ax.text(text_right,text_top,'ic3392',**text_kw)
	else: ax.text(text_right,text_top,'NGC '+filename,**text_kw)
	
	r_eff_inner, d_eff_inner=eff_rad_calc(inner)
	ax.text(text_right,text_top-text_hspace,'$\mathregular{R_{eff(in)}}$: %g'%(round(r_eff_inner,2)),**text_kw)
	ax.plot([r_eff_inner,r_eff_inner],[.1,d_eff_inner],color='k')
	
	if galaxy not in symmetric:
		r_eff_total, d_eff_total=eff_rad_calc(filelines)
		ax.text(text_right,text_top-2*text_hspace,'$\mathregular{R_{eff(tot)}}$: %g'%(round(r_eff_total,2)),**text_kw)
		ax.plot([r_eff_total,r_eff_total],[.1,d_eff_total],color='k',ls='--')
	
	
	"""ax.annotate('$\mathregular{F_{HI}}$: %g'%(HI_flux),(3,20))
	ax.annotate('pre $\mathregular{R_{eff}}$: %g,%g'%(pre_r_eff,partial_flux),(3,20/2))
	ax.annotate('ratio (p/t): %g'%(partial_flux/HI_flux),(3,20/4))"""

def format_ratio_axis(ax, m=1, labels=True):
	"""Aesthetics function for plotting norm-1 ratios."""
	ax1=ax
	ax1.tick_params(axis='y',which='major',pad=ax1_tickpad)
	
	if m==1:
		ylabel, ylim = asymmetry_plot_m1_r1_axis_label, ax1_ylim
	else:
		ylabel, ylim = asymmetry_plot_m2_r1_axis_label, m2ax1_ylim
		ylabel, ylim = asymmetry_plot_m2_r1_axis_label, m2ax1_ylim
	
	if labels:
		ax1.set_ylabel(ylabel, fontsize = ax1_fs, labelpad = ax1_labelpad)
		ax1.set_xlabel('Angle from North (degrees)', fontsize = ax1_fs)
	
	ax1.set_xticks([0,90,180,270,360])
	ax1.set_xticklabels([0,90,180,270,0])
	
	ax1.set_xlim(0, xmax=360)
	ax1.set_yscale('log')
	ymin, ymax = sorted((1/ylim, ylim))
	ax1.set_ylim(ymin = ymin, ymax = ymax)
	
	ax1.grid(b=True, which='major', axis='x')
	ax1.axhline(y=1, color='black', ls=':')

def format_flux_ratio_axis(Ginstance, ax2, labels=True):
	"""Aesthetics function for plotting norm-0 ratios."""
	ax2.set_ylim(ymin = -ax2_ylim, ymax = ax2_ylim)
	
	if labels:
		ax2.set_ylabel(asymmetry_plot_m1_r0_axis_label,
					   rotation=-90, labelpad=20, fontsize=ax1_fs)
		ax2.set_xlabel('Angle from North', fontsize=ax1_fs)

def get_edge_radius(Ginstance):
	"""Get a radius that will allow for comfortably viewing
	the entirety of a galaxy in a polar matplotlib plot."""
	return 1.05*(np.max(Ginstance.extentlist_graph[:,1]) +
				 np.sqrt(2)*Ginstance.pix_scale_arcsec)

def angles_figplot(polar,edge_radius,angle,c,err=0,errheight=0,ref_key=None,asy_key=None,errplot_kwargs={},ref_value=False,sig2=False,**kwargs):
	"""Plot an asymmetry angle on a polar axis.
	The axis is given by the `polar` argument."""
	polar.plot([angle,angle],[0,edge_radius],color=c,**kwargs)
	if ref_key: errheight=ref_angle_error_heights[ref_key]
	if asy_key:
		if ref_value: errheight=asy_angle_error_heights_refplot[asy_key]
		else: errheight=asy_angle_error_heights[asy_key]
	if err and errheight:
		if sig2: polar.plot(np.linspace(angle-2*err,angle+2*err,400),np.full(400,errheight*edge_radius),color=c,ls=':',**errplot_kwargs)#,**kwargs
		polar.plot(np.linspace(angle-err,angle+err,200),np.full(200,errheight*edge_radius),color=c,**errplot_kwargs)#,**kwargs

default_sig_region_kwargs=dict(facecolor='None',edgecolor='purple',alpha=.35,hatch='\\')
@makeax
def sig_region_plot(fill_condition,ax=None,X_array=None,fill_defaults=default_sig_region_kwargs):
	"""Given a boolean `fill_condition` array, fill an axis from
	top to bottom in the horizontal regions where this array is true.
	X_array gives the x-values if provided, otherwise
	range(len(fill_condition)) is taken as the x-values."""
	regions=get_region_inds(fill_condition)
	ax2=create_hidden_axis(ax); ax2.set_position(ax.get_position())
	if X_array is None: X_array=np.arange(len(fill_condition))
	for r in regions:
		if len(r)==0: continue
		#print(r)
		ax2.fill(X_array[r[[0,-1,-1,0]]],[1,1,0,0],**fill_defaults)
	ax2.plot(X_array[[0,-1]],[0,0],alpha=0)
	ax2.set_ylim(0,1)
	return ax2

default_sig_asy_kwargs=dict(facecolor='None',edgecolor='g',alpha=.4,hatch='\\')
def sig_asy_plot(ER,FR,*args,**kwds):
	"""Calculate the region where m=1 asymmetry is significant
	as deemed by extent and flux ratios, and fill the region in
	a timeplot."""
	kwds.update({'fill_defaults':default_sig_asy_kwargs})
	fill_condition = sig_asy_mask(ER,FR)
	res=sig_region_plot(fill_condition,*args,**kwds)
	return res

default_sig_angle_kwargs=dict(facecolor='None',edgecolor='y',alpha=.4,hatch='/')
def sig_angle_plot(EA,FA,*args,**kwds):
	"""Calculate the region where asymmetry angles are
	close to a reference angle, and fill the region in
	a timeplot. EA, FA should be provided relative to
	a reference angle, i.e. they should be 0 where there
	is no offset."""
	kwds.update({'fill_defaults':default_sig_angle_kwargs})#,'pr':True
	fill_condition=sig_angle_mask(EA, FA)
	res=sig_region_plot(fill_condition,*args,**kwds)
	return res
"""prior sig_angle_plot definition in simulation_definitions_former.py"""
	
default_sig_m1_m2_kwargs=dict(facecolor='None',edgecolor=color_darken('lightskyblue',.8),alpha=.4,hatch='o')
def sig_m1_m2_plot(ER,FR,m2ER,m2FR,*args,**kwds):
	"""Calculate the region where asymmetry is significant
	as deemed by m=1 extent and flux ratios and where m=1
	asymmetry dominates over m=2 asymmetry, and fill the region in
	a timeplot."""
	kwds.update({'fill_defaults':default_sig_m1_m2_kwargs})
	fill_condition=sig_asy_full_mask(ER,FR,m2ER,m2FR)
	res=sig_region_plot(fill_condition,*args,**kwds)
	return res

@makeax
def polar_on_cart_plot(t,r,xc=None,yc=None,ax=None,arrow=False,line=True,**kw):
	if xc is None:
		xc=np.average(ax.get_xlim())
		yc=np.average(ax.get_ylim())
	t,r = np.atleast_1d(t), np.atleast_1d(r)
	x,y = p_to_c(t, r, xc, yc)
	#print('t,r:',t,r)
	#print('x,y:',x,y)
	if arrow:
		arrow_plot(x[-2], y[-2], x[-1], y[-1], ax=ax, **kw)
		# shrink radius because the link will extend beyond the
		# 	arrow otherwise.
		# Another option is to shift the position of the arrow;
		# 	this may be more correct.
		x,y = p_to_c(t, r*.9, xc, yc)
	if line: ax.plot(x, y, **kw)#[0]

@makeax
def arrow_plot(x0,y0,x1,y1,ax=None,**kw):
	kw={**dict(arrowstyle="->",color=[0,.9,0],lw=2,shrinkA=0,shrinkB=0),**kw}
	if 'c' in kw:
		kw['color']=kw['c']
		kw.pop('c')
	#print('arrow:',(x0,y0,x1,y1))
	xl,xr,yb,yt=ax.axis()
	xd=xr-xl
	yd=yt-yb
	slope=(y1-y0)/(x1-x0)
	#print('slope:',slope)
	delta=.025*xd
	dist=np.sqrt((y1-y0)**2+(x1-x0)**2)
	x2,y2=x1+delta*(x1-x0)/dist,y1+delta*(y1-y0)/dist
	#print('x0,y0,x2,y2:',x0,y0,x2,y2)
	ax.annotate("",xytext=[x0,y0],xy=[x2,y2],arrowprops=kw)
	#ax.annotate("",xytext=[0,0],xy=[100,100],arrowprops=kw)


@makeax
def plot_2d_label(labels,*a,ax=None,**kw):
	for line,label in zip(ax.plot(*a,**kw),labels):
		line.set_label(label)

@makeax
def dual_fill(x,y0,y1,fc=None,ec=None,ax=None,**kw):
	#print('dual_fill kw:',kw)
	f=ax.fill_between(x,y0,y1,color=fc,edgecolor='none')
	f2=ax.fill_between(x,y0,y1,facecolor='none',edgecolor=ec,
						zorder=f.get_zorder()+1,**kw)
	return f,f2

def polar_fill(theta,radii1,radii2=None,facecolor='none',edgecolor=None,ax=None,**kw):
	ax=ax_0N(ax)
	if radii2 is None:
		ax.fill(
			*polar_reindex_multi(theta,tau,radii1),
			facecolor=facecolor,edgecolor=edgecolor,**kw)
		return
	
	theta,radii1,radii2=take_by_condition(radii1<radii2,theta,radii1,radii2)
	theta,radii1,radii2=polar_reindex_multi(theta,tau,radii1,radii2)
	theta,radii1,radii2=get_regions_from_other(theta,index_to_rad*1.1,radii1,radii2)
	
	for t,r1,r2 in zip(theta,radii1,radii2):
		tplot=np.concatenate((t,t[::-1],t[:1]))
		rplot=np.concatenate((r1,r2[::-1],r1[:1]))
		ax.fill(tplot,rplot,facecolor=facecolor,edgecolor=edgecolor,**kw)

def iter_axis_ticks(*axes, x=None, y=None, xkw=None, ykw=None, **kw):
	if x is not None:
		for ax in axes: ax.set_xticks(x,**({} if not xkw else xkw), **kw)
	if y is not None:
		for ax in axes: ax.set_yticks(y,**({} if not ykw else ykw), **kw)

def iter_axis_ticklabels(*axes, x=None, y=None, xkw=None, ykw=None, **kw):
	if x is not None:
		for ax in axes: ax.set_xticklabels(x,**({} if not xkw else xkw), **kw)
	if y is not None:
		for ax in axes: ax.set_yticklabels(y,**({} if not ykw else ykw), **kw)

def plot_asy_contours(ax_image,ax_colorbar,zdata,contour_plot,xpix_i,ypix_i,edge_radius,pix_scale_arcsec):
	#for l in ax_polar.lines[:]: l.set_zorder(0)
	cmap=contour_colormap
	cplot=ax_image.contour(zdata,contour_plot,cmap=cmap)#,zorder=-4
	ax_image.set_zorder(-4)
	cl_levels=cplot.levels
	cl_norm=Normalize(cl_levels[0],cl_levels[-1])
	cl_map=cl_norm(cl_levels)
	for cl in cl_map:
		ax_colorbar.plot([0,1],[cl,cl],c=cmap(cl),lw=2)
	ax_colorbar.axis([0,1,-.05,1.05])
	ax_colorbar.set_xticklabels([])
	ax_colorbar.set_yticks(cl_map)
	ax_colorbar.set_yticklabels(cl_levels.astype(object))
	ax_colorbar.yaxis.tick_right()
	ax_image.scatter(xpix_i,ypix_i,marker='+',s=200,c='orange')
	ax_image.set_xlim(xpix_i-edge_radius/pix_scale_arcsec,xpix_i+edge_radius/pix_scale_arcsec)
	ax_image.set_ylim(ypix_i-edge_radius/pix_scale_arcsec,ypix_i+edge_radius/pix_scale_arcsec)

def stabilize_angles(ar, mod, pr=False):
	"""Modulate angles about `mod` such that consecutive angles are
	moduldated so as to be separated by the minimal polar distance
	between them.
	
	E.g., say the angles are (10, 350, 355). One could imagine traveling from
	10 --> 355 --> 350, moving only in the forwards direction; however, a
	shorter (shortest) overall path is formed by traveling 10 --> -5 --> -10,
	moving in both +/- directions, which is equivalent (mod 360) to the first
	path. Thus the result returned is [10. -5. -10.].
	"""
	ar = np.atleast_1d(ar)
	if ar.ndim>2:
		raise ValueError('`stabilize_angles` is not configured for arrays with ndim>2')
	
	elif ar.ndim==2:
		"""
		assumes that the different angle sets are stacked as separate columns,
		as they would be passed to plt.plot
		"""
		return np.column_stack([stabilize_angles(ar[:,i], mod, pr=pr)
								for i in range(ar.shape[1])])
		
	else:
		original=ar
		nans=False
		nancond=np.isnan(ar)
		if np.any(nancond):
			nans=True
			nonnan=np.where(~nancond)
			ar=ar[nonnan]
		polar_dif=np.append(0.,polar_offset(ar[1:],ar[:-1],mod))
		stabilized=np.add.accumulate(polar_dif)+ar[0]
		if nans:
			new=np.full_like(original,nan)
			new[nonnan]=stabilized
			stabilized=new
		if pr:
			dif=np.round(original-stabilized,10)
			print('<original-stabilized>:',dif[~np.isnan(dif)]%mod,sep='\n')
		return stabilized

@makeax
def stabilize_plot(ar, mod=360, center=0, pr=False, ax=None,
	X_array=None, set_ylim=True, scatter=False, test=False, return_plots=False,
	center_on_0=False, **kwargs):
	"""Plot angles processed by `stabilize_angles`"""
	if X_array is None: X_array=np.arange(len(ar))
	stabilized=stabilize_angles(ar,mod,pr=pr)

	if center_on_0:
		stabilized -= center
		center -= center

	bottom=center-mod/2
	top=center+mod/2
	add_to_max=1+(bottom-np.nanmax(stabilized))//mod
	add_to_min=(top-np.nanmin(stabilized))//mod

	plots=[]
	for count,v in enumerate(np.arange(add_to_max,add_to_min+1)):
		if count==1: kwargs['label']=None
		if scatter:
			plots.append(ax.scatter(X_array,stabilized+v*mod,**kwargs))
# 			try: 
# 			except AttributeError as e: print('stabilize_plot error:',repr(e))#plots.extend(ax.plot(X_array,stabilized+v*mod,**kwargs))
		else:
			plots.extend(ax.plot(X_array,stabilized+v*mod,**kwargs))
# 			try: 
# 			except AttributeError as e: print('stabilize_plot error:',repr(e))#plots.append(ax.scatter(X_array,stabilized+v*mod,**kwargs))
		
		if stabilized.ndim==2:
			if scatter:
				for l,lo in zip(plots[-stabilized.shape[1]:],plots[:stabilized.shape[1]]):
					l.set_color(lo.get_color())
			else:
				for l,lo in zip(plots[-stabilized.shape[1]:],plots[:stabilized.shape[1]]):
					l.set_facecolor(lo.get_facecolor())
	if stabilized.ndim==1:
		if scatter:
			for l in plots[1:]: l.set_facecolor(plots[0].get_facecolor())
		else:
			for l in plots[1:]: l.set_color(plots[0].get_color())

	if set_ylim:
		print('setting ylim:',(bottom,top))
		ax.set_ylim(bottom,top)
	else:
		plots = sorted(plots,key = lambda p: np.abs(p.get_data()[1]-center).mean())
		for p in plots[1:]: p.remove()
		del plots[1:]
		autoscale(ax,1)

	if test: ax.set_title('count=%i'%(count+1))
	if return_plots: return plots
	return stabilized


from mpl_wrap.plotting_functions import __all__
__all__ = ('default_sig_angle_kwargs', 'polar_on_cart_plot',
'create_hidden_axis', 'paperplot', 'sig_region_plot', 'eff_rad_calc', 'fig_size_save',
'paper_fig_save', 'asymmetric_low', 'asymmetric_high', 'saveplot',
'paperplot1', 'dual_fill', 'test_fig_save', 'format_ratio_axis', 'get_edge_radius',
'format_flux_ratio_axis', 'sig_m1_m2_plot', 'sig_angle_plot', 'plot_2d_label',
'active_fig_save', 'angles_figplot', 'arrow_plot', 'symmetric', 'paperplot2',
'plot_shortside', 'default_sig_region_kwargs', 'high_i_cutoff_list', 'misc_fig_save',
'default_sig_m1_m2_kwargs', 'create_dual_axis', 'polar_fill', 'iter_axis_ticklabels',
'sig_asy_plot', 'radial_distribution', 'default_sig_asy_kwargs',
'iter_axis_ticks', 'savefigure', 'paper_tex_save', 'topical_fig_save','stabilize_angles','stabilize_plot'
, *__all__)
# 'ax_select', 



# ('makeax','axlabels','ax_0N','neatbins','keepax','savefigure','paper_tex_save',
# 'test_fig_save','paper_fig_save','topical_fig_save','active_fig_save','fig_size_save',
# 'misc_fig_save','saveplot','paperplot','paperplot1','paperplot2','axis_add',
# 'create_overlapping_axis','create_hidden_axis','create_dual_axis','vline','hline',
# 'dline','stabilize_angles','autoscale','limscale','axlim','pushax','stabilize_plot',
# 'plot_shortside','symmetric','asymmetric_low','asymmetric_high','high_i_cutoff_list',
# 'eff_rad_calc','radial_distribution','format_ratio_axis','format_flux_ratio_axis',
# 'full_legend','plot_asy_contours','square_axis_lims','default_sig_asy_kwargs',
# 'angles_figplot','default_sig_region_kwargs','sig_region_plot','get_edge_radius',
# 'sig_asy_plot','default_sig_angle_kwargs','sig_angle_plot','default_sig_m1_m2_kwargs',
# 'sig_m1_m2_plot','make_common_axlims','ax_select','polar_on_cart_plot','arrow_plot',
# 'plot_2d_label','dual_fill','polar_fill','multi_axis_labels','iter_axis_ticks',
# 'iter_axis_ticklabels','extremum_plot_prop','max_plot_prop','min_plot_prop',
# 'sorted_legend', 'ticklabels', 'lineplot')