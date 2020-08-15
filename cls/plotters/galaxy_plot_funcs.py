import numpy as np
from matplotlib import gridspec, pyplot as plt

from plot.plotting_functions import *
from prop.asy_prop_plot import *

from comp.computation_functions import *
from comp.array_functions import *
from comp.asymmetry_functions import EA_calc, score_calc
from comp.polar_functions import polar_reduce

from asy_io.asy_io import listify2

from plot.asy_figplot_base import polar_plotter

from cls.adc_prep import centroid_types
from prop.asy_prop import *
from prop.asy_prop_plot import *

@makeax
def zshow(self,ax=None,data=None,WA=None, border=True,
		  inout=True, outer=False, center=False,
		  vmin=None,vmax=None,
		  axis=True, cmap = None,
		  center_lines=True):

	if inout and not (center or outer):
		p=(
			ax.imshow(znan(self.zdata_regions.real),cmap=HI_inner_colormap),
			ax.imshow(znan(self.zdata_regions.imag),cmap=HI_outer_colormap)
		)
	else:
		cmap = cmap if cmap else HI_inner_colormap
		if outer:
			p=ax.imshow(znan(self.zdata_regions.imag),cmap=cmap)
		elif center:
			p=ax.imshow(znan(self.zdata_regions.real),cmap=cmap)
		else:
			if data is None:
				data = self.zdata

			if vmin is None:
				p=ax.imshow(znan(data), cmap = cmap)
			else:
				p=ax.imshow(znan(data), vmin=vmin, vmax=vmax, cmap = cmap)

		p=(p,)

	if border:
		if not isinstance(border,dict): border = dict(c='gold',ls=(0,(1,.5)),lw=3)
		ax.plot(
			*p_to_c(*(self.iet_ar/[1,self.pix_scale_arcsec]).T,
			xc=self.xpix,yc=self.ypix), **border)
	#ax.scatter(self.xpix,self.ypix,marker=(6,2),c=[[1,.5,0]],s=300)
	if axis:
		ax.axis(self.get_imshow_lims())
	else:
		ax.set_ylim(np.sort(ax.get_ylim()))

	ax.set_xticks([]); ax.set_yticks([])

	if center_lines:
		vline(self.xpix,ax=ax,alpha=.65)
		hline(self.ypix,ax=ax,alpha=.65)

	return p,ax

def polar_imshow(self,polar_ax=None, inout=False, border=False, center_lines=False, **kw):
	if polar_ax is None: polar_ax=ax_0N(plt.gcf().add_axes([.1,.1,.8,.8],polar=True))
	ax=create_hidden_axis(polar_ax)
	self.zshow(ax=ax, inout=inout,border=border,center_lines=False, **kw)
	#polar_ax.set_xticklabels(polar_ax.set_yticklabels([]))
	polar_ax.set_yticks([])



def plot_ratio_arrays(self, index=None, ax=None, labels=True, legend_kw={}, which=set(range(5))):
	"""`which` argument should be a set"""
	#filename=self.filename
	arrays=self.getattrs(
		self.score * [rad_to_deg, 1], self.qscore * [rad_to_deg, 1],
		'htscores_ar','fluxscore_ar','qfluxscore_ar'
	)
	arrays=[(ar if i in which else None) for i,ar in enumerate(arrays)]
	if index is not None: arrays=[ar[index] if (ar is not None) else ar for ar in arrays]
	score_ar,qscore_ar,htscores_ar,fluxscore_ar,qfluxscore_ar=arrays
	lkw=dict(loc=8)#,size=lsize
	if ax is None: ax=plt.gca(polar=False)
	for ar,c in zip((score_ar,qscore_ar,htscores_ar),(ext_color,qext_color,ht_color)):
		if ar is not None: ax.plot(ar[:,0],ar[:,1], lw=2, color=c)
	
	format_ratio_axis(ax,labels=labels)
	
	if which.difference({0,1,2}):
		ax2 = ax.twinx()
		for ar,c in zip ((fluxscore_ar,qfluxscore_ar),(flux_color,qflux_color)):
			if ar is not None: ax2.plot(ar[:,0],ar[:,1],color=c,lw=2)
		
		legend=isinstance(legend_kw,dict)
		if legend and legend_kw: lkw.update(legend_kw)
		format_flux_ratio_axis(self, ax2, which, labels=labels, legend=legend, **lkw)
		
		return ax2

def iter_plot_ratio_arrays(self,show=False,labels=False,which=set(range(4)),gs_kw={},fig_kw={},**kw):
	if not isinstance(which,set): which=set(listify2(which))
	fig=plt.figure(**merge_dicts({'figsize':(12,16)},fig_kw))
	gs1=gridspec.GridSpec(4, 4, **merge_dicts(dict(hspace=.1, wspace=0),gs_kw))
	r_axes=np.array([plt.subplot(gs1[i,2:]) for i in range(4)])
	for i,ax,index in zip(list(range(r_axes.shape[0])),r_axes,(ahl,al,a1hl,a2l)):
		self.plot_ratio_arrays(ax=ax,index=np.s_[:index],which=which,labels=labels,legend_kw=False,**kw)
	#for ax in r_axes[1:]: ax.tick_params(axis='y',which='both',labelsize=0)
	for ax in r_axes[:-1]: ax.tick_params(axis='x',which='both',labelsize=0)
	p_axes=np.array([plt.subplot(gs1[i,:2],polar=True) for i in range(4)])
	pl_ar=self.pl_ar
	for i,ax in enumerate(p_axes):
		iet_plot=polar_plotter(self,ax,plot_results=False)
		if (which & {0,1}):
			iet_plot.remove()
			t1,r1,t2,r2=pl_ar[(np.array([0,al])+(i+1)*ahl)%a2l].flat
			for t,r,c in ((t1,r1,[0,1,0]),(t2,r2,[0,0,1])):
				ax.plot([t]*2,[0,r],c=c)
				ax.scatter(t,r,c=c,s=100,zorder=1)
		elif (which & {3,4}):
			index=(i*ahl+np.arange(al+1))%a2l
			iet_ar=np.array(iet_plot.get_data()).T
			ax.fill_between(iet_ar[index,0],iet_ar[index,1],pl_ar[index,1],color=pos_shading)
			index+=al; index%=a2l
			ax.fill_between(iet_ar[index,0],iet_ar[index,1],pl_ar[index,1],color=neg_shading)
			ax.plot(iet_ar[index[[0,al]],0],iet_ar[index[[0,al]],1],c='b')
		#ax.plot(*(pl_ar[:al].T))
		ax.scatter(0,0,marker='o',c='orange',s=40,zorder=2)
		ax.set_xticks([]); ax.set_yticks([])
	paper_fig_save(
		'+'.join(p1_ratio_plot_order[np.array(sorted(which))])+' process.pdf',
		1,'process figs',
		resize=False,bbox_inches='tight'
	)
	
	
	if show: plt.show()
	else: plt.clf()

def polar_boundary_plot(self,polar_ax=None,projected=True,deprojected=False,PA=False):
	if polar_ax is None: polar_ax=ax_0N()
	lines=[]
	if projected: lines.extend(polar_ax.plot(*self.pl_ar.T))
	if deprojected: lines.extend(polar_ax.plot(*self.pl_ar_deproject.T))
	if PA:
		r=self.pl_ar[int(self.PA*deg_to_index),1]#keepax(ax=polar_ax)[-1]
		angles_figplot(polar_ax, r, self.PA*deg_to_rad, PA_color, lw=3, ls='--')
	return lines

@makeax
def noncenter_show(self,ax=None):
	plt.imshow(setnan(znan(self.zdata),np.s_[self.center_y,self.center_x]))
	ax.invert_yaxis()
	xpix,ypix=self.xpix,self.ypix
	for c,ct in zip((self.centroids*[deg_to_rad,1]),centroid_types):
		plt.scatter(*p_to_c(*c,xc=xpix,yc=ypix),s=200,marker=(6,2),zorder=1,label=ct)
		plt.legend()
	hline(ypix,zorder=-1)
	vline(xpix,zorder=-1)
	topical_fig_save(self.filename+' outer map','outer maps',self.directory)

@makeax
def plot_angular_property(self,array,center_angle,span_degrees,plot_angles=True,ax=None,adjust=0,**kw):
	"""
	required format of array:
		the data is distributed over azimuthal space along axis 0
		size along axis 0 = a2l
	
	array can be 1-d or 2-d; if 1-d, assumed to start at 0 and proceed to 360-delta_t
	"""
	
	center_index=int(center_angle*deg_to_index)
	span=int(span_degrees*deg_to_index)
	half=int(np.round(span/2))
	
	if isinstance(array,str): array=getattr(self,array)
	if array.ndim==1: array=np.column_stack((range_a2l*index_to_deg,array))
	plot=reindex(array,center_index-half,axis=0)
	
	if plot_angles:
		angles=plot[:span,0]
		if not np.allclose(angles.round(2),angles): angles=angles*rad_to_deg
		if adjust: angles=(angles+adjust)%360
		ax.plot(stabilize_angles(angles,360),plot[:span,1],**kw)
		#print('min,max of plot:',plot[:span,1].min(),plot[:span,1].max())
	else: ax.plot(plot[:span,1],**kw)
	
	return ax

def plot_outer_boundary_sector(self,*a,**kw):
	return self.plot_angular_property(self.extentlist_graph,*a,**kw)

def plot_total_flux_sector(self,*a,**kw):
	return self.plot_angular_property(self.total_f_per_t,*a,**kw)

def plot_outer_flux_sector(self,*a,**kw):
	return self.plot_angular_property(self.nc_f_per_t,*a,**kw)






__all__ = ('zshow','polar_imshow','zshow','plot_ratio_arrays','polar_boundary_plot',
'iter_plot_ratio_arrays','shortlong_sidesum_plot','score_spectrum_plot',
'plot_fr_weighted_galaxy','noncenter_show','plot_angular_property',
'plot_outer_boundary_sector','plot_total_flux_sector','plot_outer_flux_sector',)
