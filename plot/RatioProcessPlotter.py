"""RatioProcessPlotter provides a convenient interface to a certain kind of plot
used to highlight the processes for calculating m=1/m=2 asymmetry ratios & angles.

Simplified by the existence of Python's `functools.partialmethod` in Python 3.4+.
"""

from functools import partialmethod

from common import consume
from common.arrays.roll_wrap_funcs import rolling_gmean_wrap
# from comp.computation_functions import *
from comp.array_functions import get_region_inds, reindex, minmax
import comp.asymmetry_functions as asy_funcs

from matplotlib import gridspec, pyplot as plt
import numpy as np

from .plotting_functions import *

from prop.asy_prop import *
from prop.asy_prop_plot import *
from prop.asy_defaults import *


class RatioProcessPlotter:
	func_template = 'm{m}_{qtype}_process_{ptype}_frame'
	label_map = {'extent':'extent', 'flux':'outer head-tail flux',
				 'ht':'global head-tail flux'}
	ratio_map = {('extent',1):'ER', ('extent',2):'m2ER', ('ht',1):'HTR',
				 ('flux',1):'FR', ('flux',2):'m2FR'}
	text_size = 24
	xlabel = f'active angle on sky ({deg})'
	labelsize = 30
	def __init__(self, Ginstance):
		self.g = Ginstance
	
	def polar_center_scatter(self, polar_ax):
		polar_ax.scatter(0, 0, marker=(6,2,30), s=200, linewidths=2, c='g')
	
	def __get_process_frame_func(self, m, qtype, ptype):
		"""get the relevant function for plotting a specific frame
		in the process of calculating a particular asymmetry quantity.
		By design these functions take arguments (index, Axes instance)"""
		return getattr(self, self.func_template.format(m=m, qtype=qtype, ptype=ptype))
	
	def create_m_overview_axes(self, polar_space = 2, ratio_space = 4):
		"""create four polar axes arrange horizontally above a larger ratio axis"""
		g = gridspec.GridSpec(polar_space + ratio_space, 4)
		polar_axes = np.array([
			ax_0N(plt.subplot(g[:polar_space, i], polar=True, yticks=[]))
			for i in range(4)
		])
		ax = plt.subplot(g[polar_space:, :])
		return polar_axes, ax
	
	def overview_plots(self):
		"""All m=1/m=2 overview plots, i.e., show polar frames at particular
		angles in the calculation process, arranged horizontally,
		and mark the corresponding angles in a ratio plot beneath these."""
		self.m1_extent_overview_plot()
		self.m1_flux_overview_plot()
		self.m1_ht_overview_plot()
		self.m2_extent_overview_plot()
		self.m2_flux_overview_plot()
	
	def __overview_plot(self, m, qtype):
		"""Underlying method that provides for the logic in each derivative
		method reference in `overview_plots`."""
		polar_axes, ax = self.create_m_overview_axes()
		consume(p.text(.025,.975, i+1, ha='right', va='top',
				transform=p.transAxes, size=self.text_size)
				for i,p in enumerate(polar_axes))
		# indices corresponding to angles where we will focus on
		locs = np.linspace(ahl/m, a2l/m, 4, dtype=int)
		colors = 'rygb'
		# function that draws on the polar axes
		polar_plotter = self.__get_process_frame_func(m, qtype, 'polar')
		# draw on each polar axis
		for pax, p, c in zip(reindex(polar_axes,1), locs, colors):
			polar_plotter(p, pax, arrow_kw = {'color':c})
		
		# the following line calls the function that draws on the ratio axis
		# and retrieves from the call an array containing the x/y data
		# of the ratio of interest plotted as `rline_data`
		rline_data = self.__get_process_frame_func(m, qtype, 'ratio')(a2l, ax)
		# for each angle, mark the ratio curve with a color-coded symbol
		ratio = .5 * getattr(self.g, self.ratio_map[(qtype,m)])
		scale = np.diff(ax.axis()[-2:])
		for i,(l,c) in enumerate(zip(locs % a2l,colors)):
			x,y = rline_data[:, l]
			yv = rline_data[1, np.arange(l-2,l+3)%a2l].mean() > y
			ax.scatter(x,y, c=c, marker=(3,0,l*index_to_deg), s=200)
			va,yadj = (('top',-1) if yv else ('bottom',1))
			ax.text(x, y+.025*scale*yadj, (i+1)%4 + 1, ha='center', va=va,
					size=self.text_size)
		ax.set_xlim((-5,365) if m==1 else (0,360))
		ax.tick_params(axis='both',which='major',labelsize=16)
		axlabels(self.xlabel, f'{self.label_map[qtype]} ratio', size=self.labelsize)
		paper_tex_save(
			f'{self.g.filename} m{m} {qtype} overview', 'methods',
			link_folder=(f'm{m} process', self.g.filename),
			dump = 2)
	
	def m1_extent_process_polar_frame(self, pos, polar_ax, arrow_kw={}, boundary = True):
		if boundary: self.g.polar_boundary_plot(polar_ax)[0].set_zorder(-1)
		theta, radius=self.g.extentlist_graph[pos%a2l]; theta*=deg_to_rad
		polar_ax.annotate(
			"", xytext=[theta, 0], xy=[theta, radius], zorder=-1,
			arrowprops={**dict(arrowstyle="->", color='r', lw=2, shrinkA=0, shrinkB=0, zorder=-1), **arrow_kw})
		polar_ax.plot(
			[0, theta+tau/2], [0, self.g.extentlist_graph[(pos+al)%a2l, 1]],
			c=arrow_kw.get('color','r'), ls=':', lw=2,zorder=-1)
		self.polar_center_scatter(polar_ax)
		polar_ax.set_xticks([])
		polar_ax.set_yticklabels([])
		
	def _m1_flux_process_polar_frame(self, pos, polar_ax, outer, arrow_kw = {}):
		"""
		
		:param self: Galaxy instance
		:param pos: varies from 0 - a2l
		:param polar_ax: polar axis
		:param outer: show outer flux vs. global flux
		:param arrow_kw: gets passes to arrow plotting in self.g.show_m1_regions
		:return:
		"""
		self.g.show_m1_regions(polar_ax=polar_ax, active=True,
						angle = pos * index_to_deg, outer = outer, arrow_kw = arrow_kw)
	
	def m1_extent_process_ratio_frame(self, pos, ax, unsmoothed=True):
		"""
	
		:param self: Galaxy instance
		:param pos: varies from 0 - a2l
		:param ax: standard cartesian axis
		:return:
		"""
		data = self.g.extentratio[:, 1]
		if unsmoothed:
			ax.plot(np.arange(pos) * index_to_deg, data[:pos],
					ls = '--', zorder = -1, label = 'extent ratio (unsmoothed)')
		
		rline_data = None
		if pos==a2l:
			rline_data = np.array(
				ax.plot(range_a2l*index_to_deg, rolling_gmean_wrap(data, ahl),
						c=ext_color, zorder=0, label='extent ratio')
				[0].get_data()
			)
			vline(self.g.EA+180, mod=360, ls=':', ax=ax, lw=2)
			ax.scatter((self.g.EA+180)%360, self.g.ER, c='r', s=100, marker='x', zorder=1)
			ax.set_title(f'm=1: max={self.g.ER:.2}', size=16)
		pushax(ax,y = minmax(data))
# 		ax.set_xlim(0, 360)
		return rline_data
	
	def _m1_flux_process_ratio_frame(self, pos, ax, outer):
		"""
	
		:param self: Galaxy instance
		:param pos: varies from 0 - a2l
		:param ax: standard cartesian axis
		:param outer: outer flux?
		:return:
		"""
		if outer:
			smooth, ratio, angle=self.g.fluxscore_ar[:, 1], self.g.FR, self.g.FA
			color=flux_color
		else:
			smooth, ratio, angle=self.g.htscores_ar[:, 1], self.g.HTR, self.g.HTA
			color=ht_color
		
		ax.plot(range_a2l, smooth, c=[1, 1, 1, 0], zorder=-1)
		rline_data = np.array(ax.plot(np.arange(pos)*index_to_deg,
							  smooth[:pos], c=color, zorder=-1)[0].get_data())
# 		ax.set_xlim(0, 360)
		if pos==a2l:
			angle = (angle+180)%360
			vline(angle, ax=ax)
			ax.scatter(angle, ratio, c='b', s=100, marker='x', zorder=1)
		
		return rline_data
	
	def m2_extent_process_polar_frame(self, pos, polar_ax, arrow_kw={},
									  boundary = True, deprojected=False):
		if boundary:
			self.g.polar_boundary_plot(polar_ax, projected=not deprojected,
									   deprojected=deprojected)[0].set_zorder(-1)  #
		
		ext_array = self.g.extentlist_graph_deproject \
					if deprojected \
					else self.g.extentlist_graph
		
		if pos is None:
			pos = np.argmax(self.g.m2score_ar[:,1])
		
		theta, radius = ext_array[pos%a2l,:2]
		theta*=deg_to_rad
		
		arrowprops = dict(arrowstyle="->", color='r', lw=2, shrinkA=0, shrinkB=0)
		arrowprops.update(arrow_kw)
		
		polar_ax.annotate("", xytext=[theta, 0], xy=[theta, radius],
						  zorder=-1, arrowprops=arrowprops)
		polar_ax.set_yticklabels([])
		
		color = arrowprops['color']
		polar_ax.plot([theta+tau/2]*2, [0, ext_array[(pos+al)%a2l, 1]],
					  lw=2, c=color, zorder=-1)
		for sign in (-1, 1):
			polar_ax.plot([0, theta+sign*tau/4],
						  [0, ext_array[(pos+sign*ahl)%a2l, 1]],
						  c=color, ls=':', lw=2, zorder=-1)
		
		self.polar_center_scatter(polar_ax)
		polar_ax.set_xticks([])
	
	def m2_flux_process_polar_frame(self, pos, polar_ax, arrow_kw={}):
		self.g.show_m2_regions(polar_ax=polar_ax, active=True,
						angle=pos * index_to_deg, arrow_kw=arrow_kw)
	
	def m2_extent_process_ratio_frame(self, pos, ax, unsmoothed=True):
		data = self.g.m2score_ar[:,1]
		keepax(data, ax=ax)
		if unsmoothed:
			ax.plot(np.arange(pos)*index_to_deg, data[:pos],
					label='m=2 extent ratio (unsmoothed)')
		rline_data = None
		if pos==a2l:
			rline_data = np.array(
							ax.plot(*self.g.m2score_ar.T,
							label='m=2 extent ratio',c='r')[0].get_data()
						 )
			tmax,rmax = self.g.m2score_ar[np.argmax(self.g.m2score_ar[:,1]),:2]
			vline(np.array((0,180))+tmax, mod=360, ls='--', ax=ax, lw=1)
			ax.scatter((np.array([0,180])+tmax)%360, (rmax,rmax),
					   c='b', s=100, marker='x', zorder=max_plot_prop(ax,'zorder')+1)
			ax.set_title(f'm=2: max={rmax:.2} (m1::m2 = {(self.g.ER-1)/(rmax-1):.2})',size=16)
# 		ax.set_xlim(0, 360)
		
		return rline_data
	
	def m2_flux_process_ratio_frame(self, pos, ax):
		data = self.g.m2fluxscore_ar[:, 1]
		ax.plot(data, c=[1, 1, 1, 0])
		rline_data = np.array(
			ax.plot(np.arange(pos)*index_to_deg, data[:pos])
			[0].get_data()
		)
# 		if pos==a2l:
# 			ax.plot(range_a2l*index_to_deg, rolling_gmean_wrap(data, aql))
# 		ax.set_xlim(0, 360)
		
		return rline_data
	
	m1_flux_process_polar_frame = partialmethod(_m1_flux_process_polar_frame, outer=True)
	m1_ht_process_polar_frame = partialmethod(_m1_flux_process_polar_frame, outer=False)
	
	m1_flux_process_ratio_frame = partialmethod(_m1_flux_process_ratio_frame, outer=True)
	m1_ht_process_ratio_frame = partialmethod(_m1_flux_process_ratio_frame, outer=False)
	
	m1_extent_overview_plot = partialmethod(__overview_plot, m=1, qtype='extent')
	m1_flux_overview_plot = partialmethod(__overview_plot, m=1, qtype='flux')
	m1_ht_overview_plot = partialmethod(__overview_plot, m=1, qtype='ht')
	
	m2_extent_overview_plot = partialmethod(__overview_plot, m=2, qtype='extent')
	m2_flux_overview_plot = partialmethod(__overview_plot, m=2, qtype='flux')

RatioProcessPlotter.__doc__ = __doc__

__all__ = ('RatioProcessPlotter',)