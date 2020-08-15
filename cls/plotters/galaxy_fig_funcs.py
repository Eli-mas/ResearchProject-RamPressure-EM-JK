import numpy as np, pandas as pd
from matplotlib import gridspec, pyplot as plt

from plot.plotting_functions import *

from prop.asy_prop import *
from prop.asy_prop_plot import *
from prop.asy_defaults import *

from common.arrays.roll_wrap_funcs import *
from comp.computation_functions import *
from comp.array_functions import get_region_inds, reindex, minmax
import comp.asymmetry_functions as asy_funcs

REGION_HATCHING = False

# INITIAL=


def m1_m2_regions(self, m1_kw=None, m2_kw = None):
	_, axes=plt.subplots(2, 2, gridspec_kw={'hspace':0, 'wspace':-.4}, subplot_kw=dict(polar=True))
	ax1, ax2, ax3, ax4=axes.flat
	show_m1_regions(self, ax1)
	if m1_kw: show_m1_regions(self, ax2, active=True, **m1_kw)
	else: show_m1_regions(self, ax1, active=True)
	show_m2_regions(self, ax3)
	if m2_kw: show_m2_regions(self, ax4, active=True, **m2_kw)
	else: show_m2_regions(self, ax4, active=True)
	paper_tex_save(
		self.filename+' asy regions and active panels', 2, 
		link_folder='m1 m2 regions')#, resize=False

def polar_center_scatter(polar_ax):
	polar_ax.scatter(0,0,marker=(6,2,30),s=200,linewidths=2,c='g')

def m1_extent_process_polar_frame(self, pos, polar_ax, arrow_kw={}, boundary = True):
	if boundary: self.polar_boundary_plot(polar_ax)[0].set_zorder(-1)
	t, r=self.extentlist_graph[pos%a2l]; t*=deg_to_rad
	polar_ax.annotate(
		"", xytext=[t, 0], xy=[t, r], zorder=-1,
		arrowprops={**dict(arrowstyle="->", color='r', lw=2, shrinkA=0, shrinkB=0, zorder=-1), **arrow_kw})
	polar_ax.plot(
		[0, t+tau/2], [0, self.extentlist_graph[(pos+al)%a2l, 1]],
		c=arrow_kw.get('color','r'), ls=':', lw=2,zorder=-1)
	polar_center_scatter(polar_ax)
	polar_ax.set_xticks([])
	"""for j, ind in enumerate(range(0, a2l, ahl)):
		i = (pos+ind)%a2l
		c = [0, 1, 0] if j%2==0 else 'b'
		t, r=self.extentlist_graph[i, :2]; t*=deg_to_rad
		polar_ax.plot([t, t], [0, r], c=c, lw=2)
		polar_ax.scatter(t, r, c=c, s=80)"""

"""def m1_flux_process_polar_frame(self, pos, polar_ax):
	for j, ind in enumerate(range(0, a2l, ahl)):
		i=(pos+ind-aql)%a2l
		#i1, i2 = i, (i+aql)%a2l
		c = 'lightblue' if j%2==0 else 'violet'
		inds=np.arange(i, i+ahl+1)%a2l
		t=self.extentlist_graph_deproject[inds, 0]*deg_to_rad
		r=self.extentlist_graph_deproject[inds, 1]
		r0=self.m1interior_edge[inds, 1]
		polar_ax.fill_between(t, 0, r0, color='lightgreen')#
		polar_ax.fill_between(t, r0, r, color=c)#, lw=2
		
		c='b' if j%2==0 else 'purple'
		wrongside_inds=get_region_inds(r0>r)
		for inds in wrongside_inds:
			polar_ax.fill_between(t[inds], r0[inds], r[inds], color=c)"""

def m1_flux_process_polar_frame(self, pos, polar_ax, outer=True, arrow_kw = {}):
	"""

	:param self: Galaxy instance
	:param pos: varies from 0 - a2l
	:param polar_ax: polar axis
	:param outer: show outer flux vs. global flux
	:param arrow_kw: gets passes to arrow plotting in show_m1_regions
	:return:
	"""
	show_m1_regions(self, polar_ax=polar_ax, active=True, angle=pos*index_to_deg, outer=outer, arrow_kw = arrow_kw)
	polar_ax.set_yticks([])

def m1_extent_process_ratio_frame(self, pos, ax, unsmoothed=True):
	"""

	:param self: Galaxy instance
	:param pos: varies from 0 - a2l
	:param ax: standard cartesian axis
	:return:
	"""
	data=self.extentratio[:, 1]
	if unsmoothed:
		ax.plot(np.arange(pos)*index_to_deg, data[:pos],
				ls='--', zorder=-1, label='extent ratio (unsmoothed)')
	if pos==a2l:
		ax.plot(range_a2l*index_to_deg, rolling_gmean_wrap(data, ahl),
				c=ext_color, zorder=0, label='extent ratio')
		vline(self.EA+180, mod=360, ls=':', ax=ax, lw=2)
		ax.scatter((self.EA+180)%360, self.ER, c='r', s=100, marker='x', zorder=1)
		ax.set_title(f'm=1: max={self.ER:.2}', size=16)
	pushax(ax,y = minmax(data))
	ax.set_xlim(0, 360)

def m1_flux_process_ratio_frame(self, pos, ax, outer=True):
	"""

	:param self: Galaxy instance
	:param pos: varies from 0 - a2l
	:param ax: standard cartesian axis
	:param outer: outer flux?
	:return:
	"""
	if outer:
		smooth, ratio, angle=self.fluxscore_ar[:, 1], self.FR, self.FA
		color=flux_color
	else:
		smooth, ratio, angle=self.htscores_ar[:, 1], self.HTR, self.HTA
		color=ht_color
	"""data=rolling_sum_wrap(source, ahl)/self.totalflux
	ax.plot(data, c=[1, 1, 1, 0])
	ax.plot(np.arange(pos)*index_to_deg, data[:pos], ls='--')
	if pos==a2l:
		smooth_ax=create_dual_axis(ax)
		smooth_ax.plot(range_a2l*index_to_deg, smooth, c=color)
		smooth_ax.set_xlim(0, 360)
		smooth_ax.set_xticks([])"""
	ax.plot(range_a2l, smooth, c=[1, 1, 1, 0], zorder=-1)
	ax.plot(np.arange(pos)*index_to_deg, smooth[:pos], c=color, zorder=-1)
	ax.set_xlim(0, 360)
	if pos==a2l:
		angle=(angle+180)%360
		vline(angle, ax=ax)
		ax.scatter(angle, ratio, c='b', s=100, marker='x', zorder=1)

def m1_extent_process_plot_frame(self, p, polar_ax=None, ax=None):
	if polar_ax is None: polar_ax=plt.gca(polar=True)
	else: polar_ax=ax_0N(polar_ax)
	if ax is None: ax=plt.gca(polar=False)
	#self.polar_boundary_plot(polar_ax)[0].set_zorder(-1)
	pos=int(p*a2l/100)
	self.m1_extent_process_polar_frame(pos, polar_ax)
	#show_m1_regions(self, polar_ax=polar_ax, active=True, angle=360*p/100)
	self.m1_extent_process_ratio_frame(pos, ax)
	polar_ax.set_yticks([])

def m1_flux_process_plot_frame(self, p, polar_ax=None, ax=None, outer=True):
	if polar_ax is None: polar_ax=plt.gca(polar=True)
	else: polar_ax=ax_0N(polar_ax)
	if ax is None: ax=plt.gca(polar=False)
	#self.polar_boundary_plot(polar_ax)[0].set_zorder(-1)
	pos=int(p*a2l/100)
	self.m1_flux_process_polar_frame(pos, polar_ax, outer=outer)
	#polar_ax.plot(*self.m1interior_edge.T, c='k', lw=1)
	self.m1_flux_process_ratio_frame(pos, ax, outer=outer)

def m1_extent_process_plot(self):
	plt.gcf().set_size_inches(12, 16)
	nsteps=4
	gs=gridspec.GridSpec(nsteps, 4, hspace=.1, wspace=0)
	for i, p in enumerate(np.linspace(100/nsteps, 100, nsteps)):
		self.m1_extent_process_plot_frame(p, 
			polar_ax=plt.subplot(gs[i, :2], polar=True), 
			ax=plt.subplot(gs[i, 2:])
			)
	#plt.show()
	paper_tex_save(
		self.filename+' m1 extent process', 2, 
		link_folder=('m1 process', self.filename), resize=False)

def m1_flux_process_plot(self, outer=True):
	plt.gcf().set_size_inches(12, 16)
	nsteps=4
	gs=gridspec.GridSpec(nsteps, 4, hspace=.1, wspace=0)
	for i, p in enumerate(np.linspace(100/nsteps, 100, nsteps)):
		self.m1_flux_process_plot_frame(p, 
			polar_ax=plt.subplot(gs[i, :2], polar=True), 
			ax=plt.subplot(gs[i, 2:]), 
			outer=outer
			)
	#plt.show()
	paper_tex_save(
		self.filename+' m1 %s flux process'%('outer' if outer else 'global'), 2, 
		link_folder=('m1 process', self.filename), resize=False)
	

def m2_extent_process_polar_frame(self, pos, polar_ax, arrow_kw={}, boundary = True, deprojected=False):
	if boundary:
		self.polar_boundary_plot(polar_ax, projected=not deprojected,
								 deprojected=deprojected)[0].set_zorder(-1)  #
	if deprojected: ext_array=self.extentlist_graph_deproject
	else: ext_array=self.extentlist_graph
	if pos is None: pos = np.argmax(self.m2score_ar[:,1])
	t, r = ext_array[pos%a2l,:2]
	t*=deg_to_rad
	arrowprops = {**dict(arrowstyle="->", color='r', lw=2, shrinkA=0, shrinkB=0), **arrow_kw}
	polar_ax.annotate(
		"", xytext=[t, 0], xy=[t, r], zorder=-1,
		arrowprops=arrowprops
	)
	color = arrowprops['color']
	polar_ax.plot([t+tau/2]*2, [0, ext_array[(pos+al)%a2l, 1]], lw=2, c=color, zorder=-1)
	for a in (-1, 1):
		polar_ax.plot([0, t+a*tau/4], [0, ext_array[(pos+a*ahl)%a2l, 1]], c=color, ls=':', lw=2, zorder=-1)
	polar_center_scatter(polar_ax)
	polar_ax.set_xticks([])
	"""for j, ind in enumerate(range(0, a2l, ahl)):
		i = (pos+ind)%a2l
		c = [0, 1, 0] if j%2==0 else 'b'
		t, r=self.extentlist_graph_deproject[i, :2]; t*=deg_to_rad
		polar_ax.plot([t, t], [0, r], c=c, lw=2)
		polar_ax.scatter(t, r, c=c, s=80)"""

"""def m2_flux_process_polar_frame(self, pos, polar_ax):
	for j, ind in enumerate(range(0, a2l, ahl)):
		i=(pos+ind-aql)%a2l
		#i1, i2 = i, (i+aql)%a2l
		c = 'lightblue' if j%2==0 else 'violet'
		inds=np.arange(i, i+ahl+1)%a2l
		t=self.extentlist_graph_deproject[inds, 0]*deg_to_rad
		r=self.extentlist_graph_deproject[inds, 1]
		r0=self.m2interior_edge[inds, 1]
		polar_ax.fill_between(t, 0, r0, color='lightgreen')#
		polar_ax.fill_between(t, r0, r, color=c)#, lw=2
		
		c='b' if j%2==0 else 'purple'
		wrongside_inds=get_region_inds(r0>r)
		for inds in wrongside_inds:
			polar_ax.fill_between(t[inds], r0[inds], r[inds], color=c)"""

def m2_flux_process_polar_frame(self, pos, polar_ax, arrow_kw={}):
	show_m2_regions(self, polar_ax=polar_ax, active=True, angle=pos * index_to_deg, arrow_kw=arrow_kw)

	

def m2_extent_process_ratio_frame(self, pos, ax, unsmoothed=True):
	#data=self.m2_ext_ratios[:, 1]
	data = self.m2score_ar[:,1]
	keepax(data, ax=ax)
	if unsmoothed:
		ax.plot(np.arange(pos)*index_to_deg, data[:pos],
				label='m=2 extent ratio (unsmoothed)')
	if pos==a2l:
		ax.plot(*self.m2score_ar.T,
				label='m=2 extent ratio',c='r')
		tmax,rmax = self.m2score_ar[np.argmax(self.m2score_ar[:,1]),:2]
		vline(np.array((0,180))+tmax, mod=360, ls=':', ax=ax, lw=2)
		ax.scatter((np.array([0,180])+tmax)%360, (rmax,rmax),
				   c='b', s=100, marker='x', zorder=max_plot_prop(ax,'zorder')+1)
		ax.set_title(f'm=2: max={rmax:.2} (m1::m2 = {(self.ER-1)/(rmax-1):.2})',size=16)
	ax.set_xlim(0, 360)

def m2_flux_process_ratio_frame(self, pos, ax):
	#data=asy_funcs.get_m2_flux_arrays(self)[2]
	data = self.m2fluxscore_ar[:, 1]
	ax.plot(data, c=[1, 1, 1, 0])
	ax.plot(np.arange(pos)*index_to_deg, data[:pos])
	#if pos==a2l:
	#	ax.plot(range_a2l*index_to_deg, rolling_gmean_wrap(data, aql))
	ax.set_xlim(0, 360)

def m2_extent_process_plot_frame(self, p, polar_ax=None, ax=None):
	if polar_ax is None: polar_ax=plt.gca(polar=True)
	else: polar_ax=ax_0N(polar_ax)
	if ax is None: ax=plt.gca(polar=False)
	#self.polar_boundary_plot(polar_ax)[0].set_zorder(-1)#, projected=False, deprojected=True
	pos=int(p*a2l/100)
	self.m2_extent_process_polar_frame(pos, polar_ax)
	self.m2_extent_process_ratio_frame(pos, ax)
	polar_ax.set_yticks([])

def m2_flux_process_plot_frame(self, p, polar_ax=None, ax=None):
	if polar_ax is None: polar_ax=plt.gca(polar=True)
	else: polar_ax=ax_0N(polar_ax)
	if ax is None: ax=plt.gca(polar=False)
	#self.polar_boundary_plot(polar_ax, projected=False, deprojected=True)[0].set_zorder(-1)
	pos=int(p*a2l/100)
	self.m2_flux_process_ratio_frame(pos, ax)
	self.m2_flux_process_polar_frame(pos, polar_ax)
	#polar_ax.plot(*self.m2interior_edge.T, c='k', lw=1)
	#polar_ax.set_yticks([])

def m2_extent_process_plot(self):
	plt.gcf().set_size_inches(12, 16)
	nsteps=4
	gs=gridspec.GridSpec(nsteps, 4, hspace=.1, wspace=0)
	for i, p in enumerate(np.linspace(100/nsteps, 100, nsteps)):
		self.m2_extent_process_plot_frame(p, 
			polar_ax=plt.subplot(gs[i, :2], polar=True), 
			ax=plt.subplot(gs[i, 2:])
			)
	#plt.show()
	paper_tex_save(
		self.filename+' m2 extent process', 2, 
		link_folder=('m2 process', self.filename), resize=False)

def m2_flux_process_plot(self):
	plt.gcf().set_size_inches(12, 16)
	nsteps=4
	gs=gridspec.GridSpec(nsteps, 4, hspace=.1, wspace=0)
	for i, p in enumerate(np.linspace(100/nsteps, 100, nsteps)):
		self.m2_flux_process_plot_frame(p, 
			polar_ax=plt.subplot(gs[i, :2], polar=True), 
			ax=plt.subplot(gs[i, 2:])
			)
	#plt.show()
	paper_tex_save(
		self.filename+' m2 flux process', 2, 
		link_folder=('m2 process', self.filename), resize=False)

def create_m_overview_axes(polar_space = 2, ratio_space = 4):
	g = gridspec.GridSpec(polar_space + ratio_space, 4)
	polar_axes = np.array([ax_0N(plt.subplot(g[:polar_space, i], polar=True)) for i in range(4)])
	ax = plt.subplot(g[polar_space:, :])
	return polar_axes, ax

def m1_extent_overview_plot(self):
	polar_axes, ax = create_m_overview_axes()

	locs = np.linspace(ahl,a2l,4,dtype=int)
	colors = ['r','y','g','b']
	for pax,p,c in zip(reindex(polar_axes,1),locs,colors):
		self.m1_extent_process_polar_frame(p, pax, arrow_kw = {'color':c})

	self.m1_extent_process_ratio_frame(a2l, ax)
	for l,c in zip(locs%a2l,colors):
		ax.scatter(ax.lines[3].get_data()[0][l],ax.lines[3].get_data()[1][l],c=c,marker=(3,0,l*index_to_deg),s=200)
	ax.set_xlim(-5,365)
	paper_tex_save(
		self.filename + ' m1 extent overview', 2,
		link_folder=('m1 process', self.filename), resize=2)

def m1_flux_overview_plot(self):
	polar_axes, ax = create_m_overview_axes()

	locs = np.linspace(ahl,a2l,4,dtype=int)
	colors = ['r','y','g','b']
	for pax,p,c in zip(reindex(polar_axes,1),locs,colors):
		self.m1_flux_process_polar_frame(p, pax, arrow_kw = {'color':c})

	self.m1_flux_process_ratio_frame(a2l, ax)
	for l,c in zip(locs%a2l,colors):
		ax.scatter(ax.lines[1].get_data()[0][l],ax.lines[1].get_data()[1][l],c=c,marker=(3,0,l*index_to_deg),s=200)
	ax.set_xlim(-5,365)
	paper_tex_save(
		self.filename + ' m1 flux overview', 2,
		link_folder=('m1 process', self.filename), resize=2)

def m2_extent_overview_plot(self):
	polar_axes, ax = create_m_overview_axes()

	locs = np.arange(0,5*aql,aql)
	colors = ['r','y','g','b']
	for pax,p,c in zip(polar_axes,locs,colors):
		self.m2_extent_process_polar_frame(p, pax, arrow_kw = {'color':c})

	self.m2_extent_process_ratio_frame(a2l, ax)
	for l,c in zip(locs%a2l,colors):
		ax.scatter(ax.lines[1].get_data()[0][l],ax.lines[1].get_data()[1][l],c=c,marker=(3,0,l*index_to_deg),s=200)
	ax.set_xlim(-5,365)
	paper_tex_save(
		self.filename + ' m2 extent overview', 2,
		link_folder=('m2 process', self.filename), resize=2)

def m2_flux_overview_plot(self):
	polar_axes, ax = create_m_overview_axes()

	locs = np.arange(0,5*aql,aql)
	colors = ['r','y','g','b']
	for pax,p,c in zip(polar_axes,locs,colors):
		self.m2_flux_process_polar_frame(p, pax, arrow_kw = {'color':c})

	self.m2_flux_process_ratio_frame(a2l, ax)
	for l,c in zip(locs%a2l,colors):
		ax.scatter(ax.lines[1].get_data()[0][l],ax.lines[1].get_data()[1][l],c=c,marker=(3,0,l*index_to_deg),s=200)
	ax.set_xlim(-5,365)
	paper_tex_save(
		self.filename + ' m2 flux overview', 2,
		link_folder=('m2 process', self.filename), resize=2
	)

def m1_extent_maximizing_angle_plot(self):
	g = gridspec.GridSpec(5,5)
	polar_ax=ax_0N(plt.subplot(g[:, :2], polar=True))
	self.m1_extent_process_polar_frame(int(np.round(self.EA*deg_to_index)), polar_ax)
	ax = plt.subplot(g[1:4, 2:])
	self.m1_extent_process_ratio_frame(a2l, ax)
	paper_tex_save(
		self.filename + ' m1 maximizing extent', 2,
		link_folder=('m1 process', self.filename), resize=2
	)

def deprojection_plot(self):
	fig=plt.figure()
	fig.set_size_inches(13, 8)
	gs=gridspec.GridSpec(5, 6)#, hspace=0, wspace=0
	ax=plt.subplot(gs[:3, :3], polar=True)
	ax2=plt.subplot(gs[:3, 3:], polar=True)
	for a in (ax, ax2): a.set_theta_zero_location('N')
	self.polar_boundary_plot(PA=True, polar_ax=ax)
	self.polar_boundary_plot(deprojected=True, PA=True, polar_ax=ax2)
	
	ax3=plt.subplot(gs[3:, :])
	ax3.plot(range_a2l*index_to_deg, self.extentlist_graph_deproject[:, 2], 
			 ls='-.', label='deprojection factor')
	vline((self.PA, self.PA+180), ax=ax3, c=PA_color, ls='--', lw=2, mod=360, 
		   label='major-axis PA')
	ax3.set_xticks(np.linspace(0, 360, 8+1, dtype=int))
	ax3.set_xlim(0, 360)
	
	ax3.legend()
	fig.legend(
		ax2.lines, ('projected', 'deprojected'), 
		loc='lower center', 
		bbox_to_anchor=(
			(ax3.get_position().x0+ax3.get_position().x1)/2, 
			ax3.get_position().y1+.01), 
		fontsize=16, title='boundaries:', title_fontsize=16)
	print((ax3.get_position().x0+ax3.get_position().x1)/2)
	paper_tex_save(self.filename+' deprojection_plot', 2, link_folder=('m2 process', self.filename), bbox_inches=None)

def show_m1_regions(g, polar_ax=None, active=False, angle=30, outer=True, arrow_kw = {}):
	#if isinstance(g, str): g=Galaxy(g, save=False)
	polar_ax=ax_0N(polar_ax)
	polar_ax.axis('off')
	if outer: polar_ax.plot(*g.iet_ar.T, c='k', ls=(0, (3, 1)), lw=3)
	polar_ax.plot(*g.pl_ar.T, c=outer_HI_color)
	if outer:
		polar_fill(g.iet_ar[:, 0], g.iet_ar[:, 1], None, 
					color=center_shading, ax=polar_ax,zorder=1)
	#polar_ax.scatter(0, 0, c='b', s=200, marker='+',zorder=2)
	if active:
		r=keepax(ax=polar_ax)[-1]#*.5
		t=angle*deg_to_rad
		tpos=int(np.round(angle*deg_to_index))
		kw={**dict(arrowstyle="->", color='r', lw=2, shrinkA=0, shrinkB=0), **arrow_kw}#color=[0, .5, 1]
		polar_ax.annotate("", xytext=[t+tau/2, r], xy=[t, r], arrowprops=kw)
		polar_ax.plot([t-tau/4, t+tau/4], (r, )*2, c=kw['color'], ls='--')
		pl_view=reindex(g.pl_ar, tpos-ahl, extend=1)
		if outer:
			iet_view=reindex(g.iet_ar, tpos-ahl, extend=1)
			polar_fill(
				iet_view[:al, 0], iet_view[:al, 1], pl_view[:al, 1], 
				pos_shading, color_darken(pos_shading), zorder=0,
				ax=polar_ax, hatch='||' if REGION_HATCHING else None)
			polar_fill(
				iet_view[al:, 0], iet_view[al:, 1], pl_view[al:, 1], 
				neg_shading, color_darken(neg_shading), zorder=0,
				ax=polar_ax, hatch='--' if REGION_HATCHING else None)
		else:
			polar_fill(
				pl_view[:al, 0], pl_view[:al, 1], None, 
				color_lighten(ht_shading_active, alpha=.5), ht_shading_active, 
				ax=polar_ax, hatch='oo' if REGION_HATCHING else None)
			polar_fill(
				pl_view[al:, 0], pl_view[al:, 1], None, 
				ht_shading_opp, 'y', 
				ax=polar_ax, hatch='++' if REGION_HATCHING else None)
	else:
		polar_fill(g.iet_ar[:, 0], g.iet_ar[:, 1], g.pl_ar[:, 1], 
					color='orange', alpha=.75, ax=polar_ax)
	#plt.show()

def show_m2_regions(g, polar_ax=None, active=False, angle=30, arrow_kw = {}):
	#if isinstance(g, str): g=Galaxy(g, save=False)
	polar_ax=ax_0N(polar_ax)
	polar_ax.axis('off')
	polar_ax.plot(*g.m2interior_edge.T, c='k', ls=(0, (3, 1)), lw=3)
	polar_ax.plot(*g.pl_ar.T, c=outer_HI_color)
	
	pl_ar=np.concatenate((g.pl_ar, g.pl_ar[[0]]), axis=0)
	fill_r=np.copy(g.m2interior_edge)
	cond=fill_r[:, 1]>pl_ar[:, 1]
	fill_r[cond, 1]=pl_ar[cond, 1]
	
	if active:
		r=keepax(ax=polar_ax)[-1]#*.5
		t=angle*deg_to_rad
		tpos=int(angle*deg_to_index)
		kw={**dict(arrowstyle="->", color='r', lw=2, shrinkA=0, shrinkB=0), **arrow_kw}#, ls='--'
		polar_ax.annotate("", xytext=[t+tau/2, r], xy=[t, r], arrowprops=kw)
		for a in (-tau/8, tau/8): polar_ax.plot([t+a, t+tau/2+a], (r, )*2, c=[0, .5, 1], ls='--')
		extend=2
		iet_view=reindex(g.m2interior_edge[:-1], tpos-aql, extend=extend)
		pl_view=reindex(g.pl_ar, tpos-aql, extend=extend)
		fill_r_view=reindex(fill_r, tpos-aql, extend=extend)
		for p, c, h in zip((0, ahl), (pos_shading, neg_shading), (('||', '--'), ('.', 'o'))):
			for i in (p+0, p+al):
				try: polar_fill(
					fill_r_view[i:i+ahl+extend, 0], fill_r_view[i:i+ahl+extend, 1], pl_view[i:i+ahl+extend, 1], 
					color_lighten(c), c, ax=polar_ax, hatch=h[0] if REGION_HATCHING else None)
				except ValueError: pass
				try: polar_fill(
					fill_r_view[i:i+ahl+extend, 0], fill_r_view[i:i+ahl+extend, 1], iet_view[i:i+ahl+extend, 1], 
					c, color_darken(c), ax=polar_ax, hatch=h[1] if REGION_HATCHING else None)
				except ValueError: pass
				
	else:
		polar_fill(
			fill_r[:, 0], fill_r[:, 1], g.m2interior_edge[:, 1], 
			color=color_lighten('r'), ax=polar_ax)
		polar_fill(g.m2interior_edge[:-1, 0], g.m2interior_edge[:-1, 1], g.pl_ar[:, 1],
					color='orange', alpha=.75, ax=polar_ax)
		g.polar_boundary_plot(deprojected=False, polar_ax=polar_ax)
	
	polar_fill(
		fill_r[:, 0], fill_r[:, 1], None, 
		color=center_shading, ax=polar_ax, zorder=2)


__all__ = ('m1_m2_regions','polar_center_scatter','m1_extent_process_polar_frame',
'm1_flux_process_polar_frame','m1_flux_process_polar_frame',
'm1_extent_process_ratio_frame','m1_flux_process_ratio_frame',
'm1_extent_process_plot_frame','m1_flux_process_plot_frame',
'm1_extent_process_plot','m1_flux_process_plot','m2_extent_process_polar_frame',
'm2_flux_process_polar_frame','m2_flux_process_polar_frame',
'm2_extent_process_ratio_frame','m2_flux_process_ratio_frame',
'm2_extent_process_plot_frame','m2_flux_process_plot_frame','m2_extent_process_plot',
'm2_flux_process_plot','create_m_overview_axes','m1_extent_overview_plot',
'm1_flux_overview_plot','m2_extent_overview_plot','m2_flux_overview_plot',
'm1_extent_maximizing_angle_plot','deprojection_plot','show_m1_regions',
'show_m2_regions',)
