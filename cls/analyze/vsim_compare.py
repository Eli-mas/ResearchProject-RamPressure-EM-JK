from copy import copy, deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from asy_io.asy_io import sanitize_path

from comp.array_functions import minmax
from comp import polar_functions
from cls.adc import VollmerSeries, R0Series, R90Series
from plot.plotting_functions import (multi_axis_labels, make_common_axlims, 
	misc_fig_save, hline, autoscale, vline, sorted_legend)
from prop import asy_prop_plot
from prop.simprop import vollmer_disk_wind_inclinations
from prop.asy_prop import deg, nan
from prop.simprop import vollmer_data, vollmer_ranges
from math import floor

from common import EmptyDict, Struct

__prop__ = Struct(
	colors=('r', 'y', 'g', 'b'),
	markers = ('+', '^', 'x'),
	ls = (':', '--', '-'),
	incgroups = ['low', 'medium', 'high'],
	galaxies = vollmer_ranges.keys()
)
inclabels = {'low' : 'i=30', 'medium' : 'i=60', 'high' : r'i$\geq$80'}
ls_lw={':':1, '--':.5, '-':0}
MEAN_OVERLAP_ALPHA = 0.25
VARIANCE_SHADING = False
HLINE_PROP = dict(c='grey',alpha=.8)
VLINE_PROP = HLINE_PROP
DEFAULT_LW = 1
meantypes = ('incmeans','galmeans') #
PLOT_STDEV = True
FILL_STDEV = True
INC_ALT_COLORS = ('cyan','magenta','orange')

def _line_averager(pseries, average_where_possible = True, **pkw):
	data = [l[0].get_data() for l in pseries if l]
	size = len(data)
	X,Y = ([l[i] for l in data] for i in (0,1))
	xspan = minmax(X, flatten=True)
	xdelt = X[0][1]-X[0][0]
	
	try:
		df = pd.DataFrame(index = range(xspan[0],xspan[1]+xdelt,xdelt),columns = range(size),dtype=float)
	except TypeError:
		df = pd.DataFrame(index = np.arange(xspan[0],xspan[1]+xdelt,xdelt),columns = range(size),dtype=float)
	
	for i,(x,y) in enumerate(zip(X,Y)):
		df[i].loc[x[0]:x[-1]] = y
	#print(type(df.values))
	
	if average_where_possible:
		try:
			df.values[np.sum(~np.isnan(df.values),axis=1) < 2] = nan
			# set rows with only one non-missing value to be all nan
			# this way other rows will have averages of present values computed
			#	by setting skipna = True below
			
			# print('_line_averager:',type(df.values), df.values.dtype)
		except:
			print('_line_averager: ERROR',type(df.values), df.values.dtype)
			print(df.values)
			raise
	
	# if skipna, rows have values computed even with nans present
	# if not, rows with at least one nan have nan averages computed
	"""mean = df.mean(axis = 1, skipna = average_where_possible)
	std = df.std(axis = 1, skipna = average_where_possible)"""
	if average_where_possible: meanfunc,stdfunc = np.nanmean, np.nanstd
	else: meanfunc,stdfunc = np.mean, np.std
	mean = polar_functions.polar_offset(
		polar_functions.polar_reduction(meanfunc,df.values,360,axis = 1,squeeze=True),
		0,
		360
	)
	std = polar_functions.polar_reduction(stdfunc,df.values,360,axis = 1, start=False,squeeze=True)
	#print('mean:',mean)
	#print('df mean:',df.mean(axis = 1, skipna = average_where_possible).values)
	#print('std:',std)
	#print('df std:',df.std(axis = 1, skipna = average_where_possible).values)
	mean_line = Line2D(df.index, mean, **pkw)
	pkw = deepcopy(pkw)
	pkw.setdefault('lw',1)
	pkw['lw'] /= 2
	pkw.setdefault('zorder',0)
	pkw['zorder'] -= 1
	pkw['label'] = None
	std_line_1 = Line2D(df.index, mean-std, **pkw)
	std_line_2 = Line2D(df.index, mean+std, **pkw)
	return mean_line, std_line_1, std_line_2

def _meanplot(lines_tuples, ax, attr=None, typefolder=None, savetype=None, scale=False):
	# print(savetype)
	handles = [lines[0] for lines in lines_tuples]
	
	print(f'~ ~ ~ ~ ~ ~ ~ ~ _meanplot: lines_tuples: ~ ~ ~ ~ ~ ~ ~ ~\n{lines_tuples}')
	
	fills = []
	
	colors = __prop__.colors if len(lines_tuples)==4 else INC_ALT_COLORS
	print(f'vsim_compare: savetype={savetype}, lines_tuples:\n{lines_tuples}')
	#print('_meanplot: len(lines_tuples) =',len(lines_tuples))
	for lines,c in zip(lines_tuples,colors):
		for line in lines:
			line.set_color(c)
		if FILL_STDEV:
			print(f'filling lines with color {c}:\n{lines}\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
			fills.append(
				ax.fill_between(
					lines[1].get_data()[0],
					lines[1].get_data()[1],
					lines[2].get_data()[1],
					color=c, alpha=.4
				)
			)
	lines_tuples = [ax.add_line(copy(line)) for lines in lines_tuples for line in lines]
	autoscale(ax, 1, scale)

	try:
		if ax.relegend: sorted_legend(ax, handles=handles)
		else: ax.legend().remove()
	except AttributeError:
		sorted_legend(ax, handles=handles)

	if savetype:
		misc_fig_save(f'series/vsim_compare/1d/{typefolder}/single/{savetype}/vsim_compare_{attr}_{savetype}',
				  2, clf=False, close=False)
		for line in lines_tuples: line.remove()
		for fill in fills: fill.remove()
	return lines_tuples, fills

def vsim_compare(
	attr, attr2 = None, *, aplot=True, ax=None, series_kw=None, plot_kw=None, averaging=True,
	axes=None, save=True, label_gal=True, label_inc=True, text=True, name=None,
	timeplot_kw=EmptyDict, qtype=None
):
	typefolder = 'other' if qtype is not None else 'angles' if aplot else 'ratios'
	if series_kw is None: series_kw = {}
	if plot_kw is None: plot_kw = {}
	if axes is not None:
		_vsim_compare_multi_1d(attr, aplot, axes.ravel(), series_kw, plot_kw)
		return
	print('vsim_compare:',attr)
	_1d = not attr2

	if ax is None: ax=plt.subplot()

	plots, incs, *extra = _vsim_compare_plotter(
		attr, attr2, aplot, averaging, ax, plot_kw, series_kw, label_gal,
		timeplot_kw=timeplot_kw)

	if label_inc:
		for i,(m, il) in enumerate(zip(__prop__.ls, plots.index)):
			ax.plot([], [], c='k', ls=m, label=inclabels[il])

	ncol = label_inc + label_gal
	try:
		if ncol: sorted_legend(ax, fontsize=12, ncol = ncol if ncol else None)#\
		# .set_title(
		# 	getattr(asy_prop_plot.LABELS_LONG, attr),
		# 	prop=dict(size=14)
		# )
	except Exception as e:
		print(f'vsim_compare: cannot set legend title: {repr(e)}')
	
	ax.attr = attr
	if text:
		if name is None:
			try: ax.quantity = getattr(asy_prop_plot.LABELS_LONG, attr)
			except AttributeError: ax.quantity = attr
			except TypeError: print(f'unnamed quantity passed to vsim_compare')
		else:
			ax.quantity = name
		
		ax.text(
			.025, .05 if aplot else .5, ax.quantity,
			ha='left',va='bottom',transform=ax.transAxes,size=14)

	if _1d: path = (f'series/vsim_compare/1d/{typefolder}/single/raw/'
					f'vsim_compare_{sanitize_path(name) if name else attr}_raw')
	else:
		path = f'series/vsim_compare/2d/single/vsim_compare_{"+".join(sorted((attr,attr2)))}'
	
	if save:
		ax.set_xlabel('time (Myr)',size=16),
		if not name:
			ax.set_ylabel(
				f'offset from wind angle ({deg})' if aplot
				else qtype if qtype is not None
				else 'ratio value',
				size=16)
		else:
			ax.set_ylabel(name, size=16)

		if _1d:
			misc_fig_save(path, 2, clf=False, close=False)

			if aplot:
				for plot in plots.values.flat:
					if not plot: continue
					reformat_plotted_line(plot[0])

				for lines_tuples,savetype in zip(extra,meantypes):
					_meanplot(lines_tuples, ax, attr=attr, typefolder=typefolder, savetype=savetype)

			plt.clf()

		else:
			misc_fig_save(path, 2)
	"""
	if aplot and _1d:
		for p in plots['4522'].loc['low']: p.remove()
		autoscale(ax, 1)
		if save: misc_fig_save(f'series/vsim_compare/1d/{typefolder}/single/vsim_compare_{attr} sans v4522_30_30', 2)
	"""
	"""
	print(ax.dataLim)
	lowhigh=minmax([minmax(l.get_data()[1]) for l in ax.lines if len(l.get_data()[1]) > 0])
	print(lowhigh)
	for l in ax.lines:
		a = l.get_data()[1]==lowhigh[0]
		if np.any(a):
			ax.scatter(l.get_data()[0][a], l.get_data()[1][a], marker='o', facecolor='none', edgecolor='b', s=200)
	"""
	return plots, incs, extra, path

def _vsim_compare_1d_plot(S, attr, aplot, ax, **pkw):
	if aplot:
		if pkw.get('sig_asy') is None:
			try:
				mask = S.sig_asy_full_mask()
			except AttributeError:
				mask = S.sig_asy_mask()
			pkw['sig_asy'] = mask

	try:	
		if callable(attr):
			attr = attr(S)
		else:
			attribute = getattr(S,attr)
			if callable(attribute):
				attr = attribute()

		p = S.timeplot(attr, ax=ax, aplot=aplot, label=None, **pkw)
		return p
	except AttributeError as e:
		print(f'\t{S.directory}:',repr(e))

def _vsim_compare_2d_plot(S, attr, attr2, ax, **pkw):
	return S.plot(attr, attr2, ax=ax, label=None, **pkw)

def _vsim_compare_plotter(attr, attr2, aplot, averaging, ax, plot_kw, series_kw, label_gal,
						  timeplot_kw = EmptyDict
):
	plots = pd.DataFrame(index=__prop__.incgroups, columns=vollmer_ranges.keys())
	incs = pd.DataFrame(index=__prop__.incgroups, columns=vollmer_ranges.keys())

	_1d = not attr2

	gallines = pd.Series(index = __prop__.galaxies)
	inclines = pd.Series(index = __prop__.incgroups)

	if _1d and aplot:
		plot_kw.setdefault('center_on_0', True)
		plot_kw.setdefault('zoom', True)
	for i, (galaxy, c) in enumerate(zip(vollmer_ranges, __prop__.colors)):
		# galaxy_plots={}
		inc_plots, inc_labels = [], []
		for j, (inc, ls, m) in enumerate(zip(sorted(vollmer_data[galaxy].keys()), __prop__.ls, __prop__.markers)):  # markers
			if galaxy=='4522' and inc==30:
				inc_plots.append(None)
				inc_labels.append(nan)
				continue
			#print(galaxy,inc)
			S = VollmerSeries(galaxy, inc, **series_kw)
			pkw = dict()
			if _1d:
				plot = _vsim_compare_1d_plot(
					S, attr, aplot, ax, c=c, ls=ls, lw=DEFAULT_LW+ls_lw[ls],
					**plot_kw, **timeplot_kw)
			else:
				plot = _vsim_compare_2d_plot(S, attr, attr2, ax,
											 c=c, marker=m, ls='none', **plot_kw)
			inc_plots.append(plot)
			inc_labels.append(inc)

		# print(galaxy, inc, (ax.dataLim.y0, ax.dataLim.y1), minmax([minmax(l.get_data()[1]) for l in ax.lines if len(l.get_data()[1])>0]))

		if label_gal:
			ax.plot([], [], c + 'o', label=galaxy)

		plots[galaxy], incs[galaxy] = inc_plots, inc_labels
		try:
			gallines[galaxy] = _line_averager(plots[galaxy], c=c, lw=3, label=f'{vollmer_disk_wind_inclinations[galaxy]}{deg} ({galaxy})')
		except ValueError:
			gallines[galaxy] = None

	for inc,ls,m in zip(__prop__.incgroups, __prop__.ls,__prop__.markers):
		try: inclines[inc] = _line_averager(plots.loc[inc], label=f'{inclabels[inc]} mean',
									  c='k', ls=ls, lw = 3 if ls == ':' else 2)
		except ValueError:
			inclines[inc] = None

	if _1d:
		if aplot:
			hline(0, ax=ax, **HLINE_PROP)
			autoscale(ax, 1)
		ax.vline = vline(0, ax=ax, **VLINE_PROP)#, ls='none',marker='D',markersize=5,c='b'

		return plots, incs, inclines, gallines
	else:
		return plots, incs

def reformat_plotted_line(line: Line2D) -> None:
	if PLOT_STDEV:
		line.remove()
	else:
		line.set_alpha(MEAN_OVERLAP_ALPHA)
		line.set_lw(1 if line.get_ls() != ':' else 2)
		line.set_label(f'_{line.get_label()}')

def scale_axes(axes,aplot):
	if aplot:
		make_common_axlims(*axes)
	else:
		ext_axes, flux_axes = [], []
		for ax in axes:
			if ax.dataLim.y1 > 1:
				ext_axes.append(ax)
			else:
				flux_axes.append(ax)
		if ext_axes: make_common_axlims(*ext_axes)
		if flux_axes: make_common_axlims(*flux_axes)


# for ax in axes:
#	keepax(ax=ax)

def _vsim_compare_multi_1d(attrs, aplot	, axes, series_kw, plot_kw, roediger=True):
	print('_vsim_compare_multi_1d:',attrs)

	typefolder = 'angles' if aplot else 'ratios'
	returns = []

	for i, (ax, a) in enumerate(zip(axes, attrs)):
		returns.append(vsim_compare(
			a, aplot=aplot, ax=ax, series_kw=series_kw, plot_kw=plot_kw, save=False,
			label_gal=i == 0, label_inc=i == 1
		)[:3])
		ax.relegend = not bool(i)
		if i%2: ax.yaxis.tick_right()
	plots,_,extras = ([r[i] for r in returns] for i in range(3))
	plots = [p for plot in plots for p in plot.values.ravel() if p]
	multi_axis_labels(
		axes, 'time (Myr)', 
		f'offset from wind angle ({deg})' if aplot else 'ratio value', 
		size=16
	)

	unscaled_axlims = np.array([ax.axis() for ax in axes])
	for s,scaling in zip((False, True),("unscaled","scaled")):
		for p in plots: p[0].set_alpha(1)
		if s: scale_axes(axes,aplot)
		for ax in axes:
			ax.vline[0].set_ydata(ax.axis()[2:])
		misc_fig_save(
			f'series/vsim_compare/1d/{typefolder}/multi/raw/vsim_compare_{"+".join(sorted(attrs))}_raw_{scaling}',
			2, clf=False, close=False
		)
	
	
	if aplot:
		s0 = R0Series()
		s90 = R90Series()
		
		for p in plots: reformat_plotted_line(p[0])
		# for ax in axes:
		# 	for p in ax.lines: p.set_alpha(1); p.set_color('b')
		for s,scaling in zip((False, True),("unscaled","scaled")):
			print(f'>  >  >  >  scaling:{scaling}  <  <  <  <')
			if s: scale_axes(axes,aplot)
			for i,savetype in enumerate(meantypes):
				lines = []
				fills=[]
				print(f'_vsim_compare_multi_1d ({attrs}): iterating over extras')
				for ax,extra in zip(axes,extras):
					print(f'****    extra    ****:\n{extra}')
					plotted_lines,plotted_fills=_meanplot(extra[i], ax, scale=not s)
					lines.extend(plotted_lines)
					fills.extend(plotted_fills)
					
					spans = (f.get_datalim(f.get_transform()) for f in plotted_fills)
					spans = np.array([(bbox.x0, bbox.x1) for bbox in spans])
					xspan = (spans[:,0].min(), spans[:,1].max())
					r0_attr = s0.attr_sig(s0.offset(ax.attr)).squeeze()
					r90_attr = s90.attr_sig(s90.offset(ax.attr)).squeeze()
					lines.extend(
						ax.plot(
							s0.attr_sig(s0.time_index).squeeze() + (xspan[0]-s0.time_index[0]),
							r0_attr, c='k', ls='--')
					)
					lines.extend(
						ax.plot(
							s90.attr_sig(s90.time_index).squeeze() + (xspan[0]-s90.time_index[0]),
							r90_attr, c='k', ls=':')
					)
				
				for ax in axes:
					ax.vline[0].set_ydata(ax.axis()[2:])
				
				axes[1].legend(labels=('r0_90 (flipped)','r_90_0'),handles=lines[-2:])
				
				if s:
					pass
				else:
					for ax, lim in zip(axes, unscaled_axlims):
						ax.axis(lim)
						if ax.get_ylim()[0]>-30:
							ax.set_ylim(bottom=-30)
				
				misc_fig_save(
					f'series/vsim_compare/1d/{typefolder}/multi/{savetype}/vsim_compare_{"+".join(sorted(attrs))}_{savetype}_{scaling}',
					2, clf=False, close=False
				)

				for line in lines:
					line.remove()
				for fill in fills:
# 					print(f'removing fill with color {fill.get_facecolor()}')
					fill.remove()
				#for ax in axes:
				#	for coll in ax.collections:
				#		coll.remove()
				#		#if isinstance(coll,PolyCollection): coll.remove()
	plt.clf()