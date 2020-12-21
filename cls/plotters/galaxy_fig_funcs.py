from functools import partialmethod

from common.arrays.roll_wrap_funcs import *
from comp.computation_functions import *
from comp.array_functions import get_region_inds, reindex, minmax
import comp.asymmetry_functions as asy_funcs
from asy_io.asy_paths import EXAMPLES_DIRECTORY

import numpy as np, pandas as pd
from matplotlib import gridspec, pyplot as plt

from plot.plotting_functions import *
from plot.asy_figplot_base import polar_plotter

from prop.asy_prop import *
from prop.asy_prop_plot import *
from prop.asy_defaults import *

REGION_HATCHING = False







def m1_extent_process_polar_frame(self):
	self.ratio_process_plotter.m1_extent_process_polar_frame()
def m1_extent_process_ratio_frame(self):
	self.ratio_process_plotter.m1_extent_process_ratio_frame()
def m1_flux_process_polar_frame(self):
	self.ratio_process_plotter.m1_flux_process_polar_frame()
def m1_ht_process_polar_frame(self):
	self.ratio_process_plotter.m1_ht_process_polar_frame()
def m1_flux_process_ratio_frame(self):
	self.ratio_process_plotter.m1_flux_process_ratio_frame()
def m1_ht_process_ratio_frame(self):
	self.ratio_process_plotter.m1_ht_process_ratio_frame()
def m1_extent_overview_plot(self):
	self.ratio_process_plotter.m1_extent_overview_plot()
def m1_flux_overview_plot(self):
	self.ratio_process_plotter.m1_flux_overview_plot()
def m1_ht_overview_plot(self):
	self.ratio_process_plotter.m1_ht_overview_plot()
def m2_extent_overview_plot(self):
	self.ratio_process_plotter.m2_extent_overview_plot()
def m2_flux_overview_plot(self):
	self.ratio_process_plotter.m2_flux_overview_plot()


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
		self.filename+' asy regions and active panels', 'methods', 
		link_folder='m1 m2 regions')#, resize=False

def m1_extent_maximizing_angle_plot(self):
	g = gridspec.GridSpec(5,5)
	polar_ax=ax_0N(plt.subplot(g[:, :2], polar=True))
	self.m1_extent_process_polar_frame(int(np.round(self.EA*deg_to_index)), polar_ax)
	ax = plt.subplot(g[1:4, 2:])
	self.m1_extent_process_ratio_frame(a2l, ax)
	paper_tex_save(
		self.filename + ' m1 maximizing extent', 'methods',
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
	paper_tex_save(self.filename+' deprojection_plot', 'methods', link_folder=('m2 process', self.filename), bbox_inches=None)

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

@makeax
def raw_zmap(self, ax=None, save=False):
	self.zshow(cmap='Greys_r', remove_background=False,
			   inout=False, center_lines=False, border=False)
	ax.axis(self.get_imshow_lims())
	if save:
		fig_size_save(f'{EXAMPLES_DIRECTORY}{self.filename} raw zmap', (8,8), fmt='png')

# @makeax
def angle_map_plot(self, polar_ax=None, save=False):
	if polar_ax is None: polar_ax = ax_0N()
	polar_plotter(self, polar_ax)
	self.zshow(ax=create_hidden_axis(polar_ax), border=False, inout=False)
	if save:
		fig_size_save(f'{EXAMPLES_DIRECTORY}{self.filename} angle map', (8,8), fmt='png')

def raw_to_angles_plot(self, save=True):
	self.raw_zmap(plt.subplot(121))
	self.angle_map_plot(plt.subplot(122, polar=True))
	if save:
		fig_size_save(f'{EXAMPLES_DIRECTORY}{self.filename} raw to angle map', fmt='png')

__all__ = ('m1_extent_process_polar_frame','m1_flux_process_polar_frame',
'm1_extent_process_ratio_frame','m1_flux_process_ratio_frame',
'm1_ht_process_polar_frame','m1_ht_process_ratio_frame',
'm1_extent_overview_plot','m1_flux_overview_plot','m1_ht_overview_plot',
'm2_extent_overview_plot','m2_flux_overview_plot','deprojection_plot',
'm1_extent_maximizing_angle_plot','angle_map_plot','raw_to_angles_plot',
'show_m1_regions','show_m2_regions','m1_m2_regions','raw_zmap')
