import pandas as pd, numpy as np
from prop.asy_prop import *
from prop.asy_prop_plot import *
from plot.plotting_functions import *
from functools import partial
from asy_io.asy_io import *
from plot.voll_rp_profile import add_rp_profile

pindex=['c', 'marker', 'ls']

plotting_kw=pd.DataFrame(
	data=np.array((
		(ext_color, 	flux_color, 	ht_color, 	qext_color, 	qflux_color, 	ext_trig_color	, 	flux_trig_color, 	[0, .8, 0]), 
		('+', 		'D'		, 	(6, 2)	, 	'o'		, 	's'			, 	'x'				, 	'p'				, 	'^'		), 
		('-', 		'-'		, 	'-'		, 	'--'	, 	'--'		, 	'-.'			, 	'-.'			, 	':'		)), 
		dtype=object	
	), 

	columns=['EA', 'FA', 'HTA', 'qEA', 'qFA', 'EA_trig', 'FA_trig', 'tail_A'], 
	index=pindex
	
)

plotting_kw[[k.replace('A', 'R') for k in plotting_kw.columns[:-1]]]=plotting_kw[plotting_kw.columns[:-1]]

# INITIAL=

@partial(paperplot2, show=False)
@makeax
def timeplot(self, attr, ax=None, aplot=False, alter=None, mode='plot', 
	pcode=True, diff=False, start=None, end=None, slice=None, sig_asy = None,
	zoom = True, adjust = 0, plot_sig_asy=False,  offsets=False,
	**pkwargs):#pfunc=plt.plot,

	if diff: tplot, res=self.time_diff, np.diff(self.get_attr(attr), 1)
	else:
		#if not isinstance(attr, str):
		if slice is None:
			if start is None: start=0
			if end is None: end=self.count
			
			try: res=getattr(self,attr)#self.get_attr(attr)
			except AttributeError as e:
				print('timeplot error:',repr(e))
				return
			except TypeError:
				# print('timeplot: the entered attr is not a string: treating as data')
				res = attr
			tplot=self.time_index[start:end]
			
			if res.shape[0]!=(end-start):
				raise ValueError(
				'`timeplot`: if named attribute not passed, '
				' data must match dimension of Collection, '
				' or be compatible with `start` and `end` kwargs; '
				f'received object of shape {res.shape}')
		else:
			try: res=self.get_attr(attr)
			except AttributeError as e:
				print('timeplot error:',repr(e))
				return
			except TypeError:
				print('timeplot: the entered attr is not a string: treating as data')
				res = attr
			res=res[slice]
			tplot=self.time_index[slice]
	if alter is not None: res=alter(res)

	# print('timeplot: to be plotted:',res)
	
	pkw=get_attribute_plot_kw(attr, pcode, mode)
	if isinstance(attr, str):
		if diff:
			pkwargs.setdefault('label',attr+' diff')
		else:
			try: pkwargs.setdefault('label', getattr(LABELS,attr))
			except AttributeError: pkwargs.setdefault('label',attr)
	pkw.update(pkwargs)

	if sig_asy is not None:
		res[~sig_asy] = nan

	#pfunc(self.time_index, res, **pkw)
	if attr=='tail_A': adjust = 180
	if adjust: res = res+adjust
	#print(f'timeplot: tplot[0]={tplot[0]}, aplot={aplot}')
	if aplot:
		if self.galaxy == '4522': res = self.reverse(res)
		if offsets:
			pkw.setdefault('center', 0)
			res = self.offset(res)
		else:
			pkw.setdefault('center', self.WA)
		plot = stabilize_plot(res, ax=ax, X_array=tplot, return_plots=True, set_ylim = not zoom, **pkw)
	else:
		plot = ax.plot(tplot, res, **pkw)

	if plot_sig_asy and (sig_asy is not None):
		try:
			sig_asy_ax = sig_m1_m2_plot(self.ER, self.FR, self.m2ER, self.m2FR, ax=ax, X_array = tplot)
			#sig_asy_plot(self.ER, self.FR, ax=ax)
			ax.sig_asy_ax = sig_asy_ax
		except AttributeError as e:
			print(f'error in timeplot -> sig_m1_m2_plot: <{repr(e)}>')#: calling sig_asy_plot instead
			# sig_asy_plot(self.ER, self.FR, ax=ax)

	#ax.plot([tplot[0],tplot[-1]], [np.average(ax.get_ylim())]*2, c=[1, 1, 1, 0])

	pushax(ax,x=(tplot[0],tplot[-1]))
	try:
		pushax(sig_asy_ax,x=(tplot[0],tplot[-1]))
		#print(f'timeplot: called pushax on sig_asy_ax: limits are {sig_asy_ax.dataLim}')
		sig_asy_ax.axis('on')
		sig_asy_ax.set_xticks([]); sig_asy_ax.set_yticks([])
		sig_asy_ax.set_position(ax.get_position())
	except Exception as e:
		pass
		#print(f'timeplot: could not perform pushax on sig_asy_ax: {repr(e)}')

	return plot

def get_attribute_plot_kw(attr, pcode, mode='plot'):
	if pcode:
		try:
			pkw = dict(list(zip(pindex, plotting_kw[attr])))
		except KeyError:
			pkw = {}
	else:
		pkw = {}
	if mode == 'plot':
		pkw['marker'] = None
	elif mode == 'scatter':
		pkw['ls'] = 'none'

	return pkw

@makeax
def ratio_plot(self, ax=None, title=False, title_append='',
			   which=set(range(7)), weights=False,
			   save=True, m2=False, show_rp = True,
			   **kw):
	ax2=ax.twinx()
	
	for i, (attr, axis) in enumerate(zip(
		('ER', 'FR', 'qER', 'qFR', 'HTR', 'ER_trig', 'FR_trig'), 
		( ax,  ax2, ax,   ax2,  ax,   ax,       ax2)
	)):
		if i not in which: continue
		
		if weights and attr[:-4]=='trig': continue
		
		initial_plot=self.timeplot(attr, ax=axis, label=getattr(LABELS,attr), **kw)
		
		if weights:
			for line in initial_plot: line.set_color(mpl_colors[0])
			for j, extension in enumerate(('_trig', '_rw', '_trig_rw')):
				try:
					attr_new=attr+extension
					self.timeplot(attr_new, ax=axis, color=mpl_colors[j+1], label=attr_new, pcode=False)
				except AttributeError: pass
	
	if m2 and not weights:
		self.timeplot('m2ER', ax=ax, c=ext_color, marker='+')#, c='b'
		#self.timeplot('m2ER', ax=ax, c=ext_color, slice=np.s_[::2], marker='o')
		self.timeplot('m2FR', ax=ax2, c=flux_color, marker='+')#, c='r', marker=(6, 2)
		#self.timeplot('m2FR', ax=ax2, c=flux_color, slice=np.s_[::2], marker='o')
	
	if kw.setdefault('diff', False):
		hline(0, ax=ax, c='b', zorder=-1)
		hline(0, ax=ax2, c='purple', zorder=-1, ls=':')
	#ax.set_xlabel('t')
	#if which-{1, 3}: ax.set_ylabel('Extent ratios')
	#if which-{0, 2}: ax2.set_ylabel('Flux ratios')
	if title: ax.set_title('Asymmetry ratios by time'+title_append)
	full_legend(ax, ax2, ncol=2)
	diff=kw.setdefault('diff', False)
	print('aplot: self.directory =', self.directory)

	if show_rp and self.directory[0]=='v':
		add_rp_profile(host_ax=ax, diff=diff, galaxy=self.galaxy)

	savefile=link_folder='rplot'
	if diff: savefile+=' diff'
	if weights:
		if which=={0, }: savefile+=' EA weights'
		elif which=={1, }: savefile+=' FA weights'
		else: raise ValueError('a weighted aplot with which=%s is not currently permitted'%(which))
		link_folder='rplot weights'
	if save: self.figsave(savefile, link_folder=link_folder)
	return ax2

@makeax
def aplot(self, 
	ax=None, 
	title=False, title_append='', 
	which=set(range(8)), zoom=False, 
	weights=False, save=True, real_comparison=False,
	show_rp = True,
	**kw
):#	sig_asy=False,

# 	kw.setdefault('center', self.WA)
	which=set(listify2(which))
	attrs=('EA', 'FA', 'qEA', 'qFA', 'tail_A', 'HTA', 'EA_trig', 'FA_trig')
	plot=[False]*len(attrs)
	for i, attr in enumerate(attrs):
		if (i not in which) and (which!=attr) and (attr not in which): continue
		
		if weights and attr[-4:]=='trig': continue

		plot[i]=True

		if attr[:4]=='tail':
			self.timeplot('tail_A', ax=ax, label=LABELS.tail_A, aplot=True, zoom = zoom, **kw)
			continue
		
		initial_plot=self.timeplot(attr, ax=ax, label=getattr(LABELS,attr), aplot=True, zoom = zoom, **kw)
		
		if weights:
			#for line in initial_plot: line.set_color(mpl_colors[0])
			for j, extension in enumerate(('_trig', )):#'_rw', '_trig_rw'
				try:
					attr_new=attr+extension
					self.timeplot(
						attr_new, aplot=True, 
						ax=ax, ls='--', lw=2, 
						color=color_lighten(initial_plot[0].get_color()), 
						label=attr_new
					)
					#lines=
					#for l in lines[1:]: l.set_color(lines[0].get_color())
				except AttributeError: pass
	
	offsets = kw.get('offsets',False)
	hline(self.WA*(not offsets), zorder=-1, ax=ax, lw=2)
	ylim=ax.get_ylim()
	if real_comparison and not zoom:
		for i, attr in enumerate(attrs):
			if not plot[i]: continue
			y=self.EWA[attr]
			kw = get_attribute_plot_kw(attr, True, kw)
			#kw['ls'] = 1
			hline(y, lw=1, zorder=-.5, alpha = .5, ax=ax, **kw)
			ax.scatter([.001, .999], [(y-ylim[0])/(ylim[1]-ylim[0])]*2, c=kw.get('c', kw.get('color')), transform=ax.transAxes)
	#ax.set_xlabel('t')
	ax.set_ylabel('offset from wind angle' if offsets else 'Angle from N (\N{DEGREE SIGN})')
	ax.legend(ncol=2 if weights else 3 if len(which)<5 else 4)
	
	if show_rp and self.directory[0]=='v':
		add_rp_profile(host_ax=ax, galaxy=self.galaxy)
	
	savefile=link_folder='aplot'
	if zoom:
		try:
			#ax.set_ylim(self.aplot_ylim)
			savefile+=' zoom'
		except AttributeError: pass
	if weights:
		if 0 in which: savefile+=' EA weights'
		if 1 in which: savefile+=' FA weights'
		#else: raise ValueError('a weighted aplot with which=%s is not currently permitted'%(which))
		link_folder='aplot weights'
	"""if sig_asy:
		sig_asy_plot(self.ER, self.FR, ax=ax)
		if self.inc != 90: sig_m1_m2_plot(self.ER, self.FR, self.m2ER, self.m2FR, ax=ax)"""
	if save: self.figsave(savefile, link_folder=link_folder)
	#if title: ax.set_title('Asymmetry angles by time'+title_append)
	"""print('aplot finished')
	for i, ax in enumerate(plt.gcf().axes):
		print('axis %i has %i lines:'%(0, len(ax.lines)), [l.get_label() for l in ax.lines])"""

__all__ = ('timeplot', 'get_attribute_plot_kw', 'ratio_plot', 'aplot')